import argparse
import time
from ctypes import sizeof, string_at
from getpass import getpass
from threading import Lock

import cv2
import numpy as np
from NetSDK.NetSDK import NetClient
from NetSDK.SDK_Callback import (
    fDecCBFun,
    fRealDataCallBackEx2,
)
from NetSDK.SDK_Enum import (
    EM_LOGIN_SPAC_CAP_TYPE,
    EM_REALDATA_FLAG,
    SDK_RealPlayType,
)
from NetSDK.SDK_Struct import (
    NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY,
    NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="5.202.196.201")
    parser.add_argument("--port", type=int, default=37777)
    parser.add_argument("--channel", type=int, default=2)
    parser.add_argument("--seconds", type=int, default=20)
    parser.add_argument(
        "--output",
        default="data/outputs/dahua_channel_2.jpg",
    )
    args = parser.parse_args()

    username = input("Camera Username: ").strip()
    password = getpass("Camera Password: ")

    sdk = NetClient()

    initialized = False
    login_id = 0
    play_id = 0
    free_port = None
    stream_open = False
    player_started = False

    stats = {
        "packets": 0,
        "decoded": 0,
        "frame": None,
        "width": 0,
        "height": 0,
        "decode_errors": 0,
    }

    lock = Lock()

    def on_decoded(
        port,
        buffer,
        buffer_size,
        frame_info,
        user_data,
        reserved,
    ):
        try:
            info = frame_info.contents

            # Video frame; other types are audio.
            if info.nType != 3:
                return

            width = int(info.nWidth)
            height = int(info.nHeight)

            if width <= 0 or height <= 0:
                return

            expected = width * height * 3 // 2

            if buffer_size < expected:
                return

            raw = string_at(buffer, expected)

            with lock:
                stats["decoded"] += 1
                stats["frame"] = raw
                stats["width"] = width
                stats["height"] = height

        except Exception:  # noqa: BLE001
            with lock:
                stats["decode_errors"] += 1

    def on_real_data(
        real_handle,
        data_type,
        buffer,
        buffer_size,
        param,
        user_data,
    ):
        if buffer_size <= 0:
            return

        try:
            sdk.InputData(free_port, buffer, buffer_size)

            with lock:
                stats["packets"] += 1
        except Exception:  # noqa: BLE001
            with lock:
                stats["decode_errors"] += 1

    decoding_callback = fDecCBFun(on_decoded)
    data_callback = fRealDataCallBackEx2(on_real_data)

    try:
        print("Initializing Dahua NetSDK...")

        if not sdk.InitEx():
            raise RuntimeError("SDK Init failed")

        initialized = True

        login_in = NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY()
        login_in.dwSize = sizeof(login_in)
        login_in.szIP = args.ip.encode()
        login_in.nPort = args.port
        login_in.szUserName = username.encode()
        login_in.szPassword = password.encode()
        login_in.emSpecCap = EM_LOGIN_SPAC_CAP_TYPE.TCP
        login_in.pCapParam = None

        login_out = NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY()
        login_out.dwSize = sizeof(login_out)

        login_id, info, error = (
            sdk.LoginWithHighLevelSecurity(
                login_in,
                login_out,
            )
        )

        if not login_id:
            raise RuntimeError(f"Login failed: {error}")

        print("LOGIN: PASS")
        print("CHANNEL COUNT:", info.nChanNum)

        if not 1 <= args.channel <= int(info.nChanNum):
            raise ValueError("Invalid channel number")

        channel_index = args.channel - 1

        # Allocate Dahua PlaySDK decoder port.
        success, free_port = sdk.GetFreePort()

        if not success:
            raise RuntimeError("GetFreePort failed")

        if not sdk.OpenStream(free_port):
            raise RuntimeError("OpenStream failed")

        stream_open = True

        # Decode without a display window.
        if not sdk.Play(free_port, 0):
            raise RuntimeError("PlaySDK Play failed")

        player_started = True

        if not sdk.SetDecCallBack(
            free_port,
            decoding_callback,
        ):
            raise RuntimeError("SetDecCallBack failed")

        play_id = sdk.RealPlayEx(
            login_id,
            channel_index,
            0,
            SDK_RealPlayType.Realplay,
        )

        if not play_id:
            raise RuntimeError(
                "RealPlayEx failed: "
                + sdk.GetLastErrorMessage()
            )

        if not sdk.SetRealDataCallBackEx2(
            play_id,
            data_callback,
            None,
            EM_REALDATA_FLAG.RAW_DATA,
        ):
            raise RuntimeError("Real data callback failed")

        print("REALPLAY: STARTED")
        print("DECODER: STARTED")
        print("WAITING FOR DECODED FRAME...")

        deadline = time.monotonic() + args.seconds

        while time.monotonic() < deadline:
            with lock:
                decoded = stats["decoded"]

            if decoded > 0:
                break

            time.sleep(0.1)

        with lock:
            frame_raw = stats["frame"]
            width = stats["width"]
            height = stats["height"]
            decoded_count = stats["decoded"]
            packet_count = stats["packets"]
            decode_errors = stats["decode_errors"]

        print("PACKETS:", packet_count)
        print("DECODED FRAMES:", decoded_count)
        print("DECODE ERRORS:", decode_errors)

        if frame_raw is None:
            raise RuntimeError(
                "Video packets received but no decoded YUV frame. "
                "Check PlaySDK codec support and callback output."
            )

        yuv = np.frombuffer(
            frame_raw,
            dtype=np.uint8,
        ).reshape((height * 3 // 2, width))

        # Official sample describes decoded output as IYUV/YUV420.
        bgr = cv2.cvtColor(
            yuv,
            cv2.COLOR_YUV2BGR_I420,
        )

        from pathlib import Path

        output = Path(args.output)
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not cv2.imwrite(str(output), bgr):
            raise RuntimeError("Could not save JPEG")

        print("FRAME WIDTH:", width)
        print("FRAME HEIGHT:", height)
        print("OUTPUT:", output)
        print("DAHUA OPENCV FRAME TEST: PASS")

    except Exception as exc:
        print(
            "DAHUA OPENCV FRAME TEST: FAIL",
            type(exc).__name__,
            str(exc),
        )
        raise

    finally:
        if play_id:
            sdk.StopRealPlayEx(play_id)

        if free_port is not None:
            try:
                sdk.SetDecCallBack(free_port, None)
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: SDK callback cleanup failed: {exc}")

            if player_started:
                sdk.Stop(free_port)

            if stream_open:
                sdk.CloseStream(free_port)

            sdk.ReleasePort(free_port)

        if login_id:
            sdk.Logout(login_id)

        if initialized:
            sdk.Cleanup()

        print("SDK CLEANUP: COMPLETE")


if __name__ == "__main__":
    main()
