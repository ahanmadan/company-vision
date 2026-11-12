import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from ctypes import sizeof, string_at
from getpass import getpass
from threading import Event, Lock

from NetSDK.NetSDK import NetClient
from NetSDK.SDK_Callback import fDecCBFun, fRealDataCallBackEx2
from NetSDK.SDK_Enum import (
    EM_LOGIN_SPAC_CAP_TYPE,
    EM_REALDATA_FLAG,
    SDK_RealPlayType,
)
from NetSDK.SDK_Struct import (
    NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY,
    NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY,
)

CAMERA_IP = "5.202.196.201"
CAMERA_PORT = 37777
CHANNEL = 2
DURATION_SECONDS = 60.0
OUTPUT_FPS = 25.0

desktop = Path.home() / "Desktop"
desktop.mkdir(parents=True, exist_ok=True)

output = desktop / (
    "CompanyVision_Dahua_Raw_"
    + datetime.now().strftime("%Y%m%d_%H%M%S")
    + ".mp4"
)

sdk = NetClient()
lock = Lock()
finished = Event()

state = {
    "writer": None,
    "started": None,
    "frames": 0,
    "width": 0,
    "height": 0,
    "error": None,
}

initialized = False
login_id = 0
play_id = 0
free_port = None
stream_open = False
player_started = False

def on_decoded(
    port, buffer, buffer_size, frame_info,
    user_data, reserved
):
    try:
        info = frame_info.contents

        if info.nType != 3:
            return

        width = int(info.nWidth)
        height = int(info.nHeight)

        if width <= 0 or height <= 0:
            return

        if width % 2 or height % 2:
            return

        expected = width * height * 3 // 2

        if buffer_size < expected:
            return

        raw = string_at(buffer, expected)

        yuv = np.frombuffer(
            raw, dtype=np.uint8
        ).reshape((height * 3 // 2, width))

        frame = cv2.cvtColor(
            yuv, cv2.COLOR_YUV2BGR_I420
        )

        now = time.monotonic()

        with lock:

            if finished.is_set():
                return

            if state["started"] is None:

                state["started"] = now
                state["width"] = width
                state["height"] = height

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                writer = cv2.VideoWriter(
                    str(output),
                    fourcc,
                    OUTPUT_FPS,
                    (width, height),
                )

                if not writer.isOpened():
                    state["error"] = "VideoWriter initialization failed."
                    finished.set()
                    return

                state["writer"] = writer

                print(
                    f"RECORDING STARTED: {width}x{height}"
                )

            elapsed = now - state["started"]

            if elapsed >= DURATION_SECONDS:
                finished.set()
                return

            state["writer"].write(frame)
            state["frames"] += 1

            if state["frames"] % 100 == 0:
                print(
                    f"RECORDED: {elapsed:.1f}s "
                    f"| FRAMES: {state['frames']}"
                )

    except Exception as exc:
        with lock:
            state["error"] = repr(exc)
            finished.set()


def on_real_data(
    real_handle, data_type, buffer,
    buffer_size, param, user_data
):
    if buffer_size <= 0:
        return

    try:
        accepted = sdk.InputData(
            free_port, buffer, buffer_size
        )

        if not accepted:
            with lock:
                state["error"] = "SDK InputData failed."
                finished.set()

    except Exception as exc:
        with lock:
            state["error"] = repr(exc)
            finished.set()


decoding_callback = fDecCBFun(on_decoded)
data_callback = fRealDataCallBackEx2(on_real_data)

username = input("Camera Username: ").strip()
password = getpass("Camera Password: ")

try:
    print("\nInitializing Dahua SDK...")

    if not sdk.InitEx():
        raise RuntimeError("SDK initialization failed")

    initialized = True

    login_in = NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY()
    login_in.dwSize = sizeof(login_in)
    login_in.szIP = CAMERA_IP.encode("ascii")
    login_in.nPort = CAMERA_PORT
    login_in.szUserName = username.encode("utf-8")
    login_in.szPassword = password.encode("utf-8")
    login_in.emSpecCap = EM_LOGIN_SPAC_CAP_TYPE.TCP
    login_in.pCapParam = None

    login_out = NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY()
    login_out.dwSize = sizeof(login_out)

    login_id, device_info, error = (
        sdk.LoginWithHighLevelSecurity(
            login_in, login_out
        )
    )

    if not login_id:
        raise RuntimeError(f"Login failed: {error}")

    print("LOGIN: PASS")

    if not 1 <= CHANNEL <= int(device_info.nChanNum):
        raise RuntimeError("Invalid camera channel")

    success, free_port = sdk.GetFreePort()

    if not success:
        raise RuntimeError("GetFreePort failed")

    if not sdk.OpenStream(free_port):
        raise RuntimeError("OpenStream failed")

    stream_open = True

    if not sdk.Play(free_port, 0):
        raise RuntimeError("Decoder Play failed")

    player_started = True

    if not sdk.SetDecCallBack(
        free_port, decoding_callback
    ):
        raise RuntimeError("Decoder callback failed")

    play_id = sdk.RealPlayEx(
        login_id,
        CHANNEL - 1,
        0,
        SDK_RealPlayType.Realplay,
    )

    if not play_id:
        raise RuntimeError("RealPlay failed")

    if not sdk.SetRealDataCallBackEx2(
        play_id,
        data_callback,
        None,
        EM_REALDATA_FLAG.RAW_DATA,
    ):
        raise RuntimeError("Real data callback failed")

    print("CAMERA STREAM: CONNECTED")
    print("WAITING FOR FIRST FRAME...")

    startup_deadline = time.monotonic() + 20

    while time.monotonic() < startup_deadline:
        if state["started"] is not None:
            break
        if state["error"]:
            raise RuntimeError(state["error"])
        time.sleep(0.1)

    if state["started"] is None:
        raise RuntimeError("No camera frame received.")

    while not finished.wait(0.1):
        if state["error"]:
            raise RuntimeError(state["error"])

    if state["error"]:
        raise RuntimeError(state["error"])

finally:

    print("\nStopping camera capture...")

    if play_id:
        sdk.StopRealPlayEx(play_id)

    if free_port is not None:

        try:
            sdk.SetDecCallBack(free_port, None)
        except Exception:
            pass

        if player_started:
            sdk.Stop(free_port)

        if stream_open:
            sdk.CloseStream(free_port)

        sdk.ReleasePort(free_port)

    if login_id:
        sdk.Logout(login_id)

    if initialized:
        sdk.Cleanup()

    with lock:
        if state["writer"] is not None:
            state["writer"].release()

    if state["frames"] > 0 and output.exists():
        print("\nRECORDING COMPLETE")
        print("FILE:", output)
        print("FRAMES:", state["frames"])
        print("RESOLUTION:", state["width"], "x", state["height"])
        print("DURATION TARGET:", DURATION_SECONDS)
        print("SIZE MB:", round(output.stat().st_size / 1048576, 2))
    else:
        print("NO VIDEO SAVED")