import argparse
import time
from ctypes import sizeof, string_at
from getpass import getpass
from threading import Lock

import cv2
import numpy as np
import torch
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
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ip", default="5.202.196.201")
    parser.add_argument("--port", type=int, default=37777)
    parser.add_argument("--channel", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Loading YOLO Pose...")

    model = YOLO("yolo26n-pose.pt")

    username = input("Camera Username: ").strip()
    password = getpass("Camera Password: ")

    sdk = NetClient()

    initialized = False
    login_id = 0
    play_id = 0
    free_port = None

    stream_open = False
    player_started = False

    lock = Lock()

    state = {
        "latest": None,
        "sequence": 0,
        "packets": 0,
        "decoded": 0,
        "errors": 0,
    }

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

            with lock:
                state["sequence"] += 1
                state["decoded"] += 1

                state["latest"] = (
                    state["sequence"],
                    raw,
                    width,
                    height,
                )

        except Exception:  # noqa: BLE001
            with lock:
                state["errors"] += 1

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
            accepted = sdk.InputData(
                free_port,
                buffer,
                buffer_size,
            )

            with lock:
                if accepted:
                    state["packets"] += 1
                else:
                    state["errors"] += 1

        except Exception:  # noqa: BLE001
            with lock:
                state["errors"] += 1

    decoding_callback = fDecCBFun(on_decoded)
    data_callback = fRealDataCallBackEx2(on_real_data)

    window_name = "Company Vision - Dahua Live YOLO Pose"

    last_sequence = 0
    processed_frames = 0
    inference_times = []

    started_at = None
    last_report = time.monotonic()

    try:
        print("\nInitializing Dahua SDK...")

        if not sdk.InitEx():
            raise RuntimeError("SDK initialization failed")

        initialized = True

        login_in = NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY()
        login_in.dwSize = sizeof(login_in)
        login_in.szIP = args.ip.encode("ascii")
        login_in.nPort = args.port
        login_in.szUserName = username.encode("utf-8")
        login_in.szPassword = password.encode("utf-8")
        login_in.emSpecCap = EM_LOGIN_SPAC_CAP_TYPE.TCP
        login_in.pCapParam = None

        login_out = NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY()
        login_out.dwSize = sizeof(login_out)

        login_id, device_info, error = (
            sdk.LoginWithHighLevelSecurity(
                login_in,
                login_out,
            )
        )

        if not login_id:
            raise RuntimeError(f"Login failed: {error}")

        print("LOGIN: PASS")
        print("CHANNELS:", device_info.nChanNum)

        if not 1 <= args.channel <= int(device_info.nChanNum):
            raise ValueError("Invalid channel")

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
            free_port,
            decoding_callback,
        ):
            raise RuntimeError("Decoder callback failed")

        play_id = sdk.RealPlayEx(
            login_id,
            args.channel - 1,
            0,
            SDK_RealPlayType.Realplay,
        )

        if not play_id:
            raise RuntimeError(
                "RealPlay failed: "
                + sdk.GetLastErrorMessage()
            )

        if not sdk.SetRealDataCallBackEx2(
            play_id,
            data_callback,
            None,
            EM_REALDATA_FLAG.RAW_DATA,
        ):
            raise RuntimeError("Real data callback failed")

        print("REALPLAY: PASS")
        print("YOLO POSE: READY")
        print("Waiting for camera frame...")

        startup_deadline = time.monotonic() + 15

        while time.monotonic() < startup_deadline:
            with lock:
                available = state["latest"] is not None

            if available:
                break

            time.sleep(0.05)

        if not available:
            raise RuntimeError("No camera frame received")

        cv2.namedWindow(
            window_name,
            cv2.WINDOW_NORMAL,
        )

        cv2.resizeWindow(window_name, 1280, 720)

        started_at = time.monotonic()

        print("\nLIVE YOLO POSE STARTED")
        print("Press Q to stop.")

        while True:

            with lock:
                item = state["latest"]

            if item is None:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            sequence, raw, width, height = item

            if sequence == last_sequence:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                time.sleep(0.005)
                continue

            last_sequence = sequence

            yuv = np.frombuffer(
                raw,
                dtype=np.uint8,
            ).reshape((height * 3 // 2, width))

            frame = cv2.cvtColor(
                yuv,
                cv2.COLOR_YUV2BGR_I420,
            )

            inference_start = time.perf_counter()

            results = model.predict(
                source=frame,
                device="cuda:0",
                imgsz=args.imgsz,
                conf=args.conf,
                verbose=False,
            )

            inference_ms = (
                time.perf_counter() - inference_start
            ) * 1000

            inference_times.append(inference_ms)

            result = results[0]

            # Draw only YOLO Pose results.
            annotated = result.plot(
                boxes=True,
                labels=True,
                conf=True,
            )

            person_count = (
                len(result.boxes)
                if result.boxes is not None
                else 0
            )

            elapsed = time.monotonic() - started_at
            processed_frames += 1

            processing_fps = (
                processed_frames / max(elapsed, 0.001)
            )

            cv2.putText(
                annotated,
                f"DAHUA LIVE | CHANNEL {args.channel}",
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                annotated,
                f"PERSONS: {person_count}",
                (25, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                annotated,
                f"FPS: {processing_fps:.2f} | INFERENCE: {inference_ms:.1f} ms",
                (25, 116),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            cv2.imshow(window_name, annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            now = time.monotonic()

            if now - last_report >= 5:
                with lock:
                    decoded = state["decoded"]
                    packets = state["packets"]

                print(
                    f"FPS={processing_fps:.2f} "
                    f"DECODED={decoded} "
                    f"PROCESSED={processed_frames} "
                    f"PACKETS={packets} "
                    f"PERSONS={person_count} "
                    f"INFERENCE={inference_ms:.1f}ms"
                )

                last_report = now

    except KeyboardInterrupt:
        print("\nStopped by user")

    finally:
        print("\nStopping live processing...")

        cv2.destroyAllWindows()

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

        elapsed = (
            time.monotonic() - started_at
            if started_at is not None
            else 0
        )

        average_inference = (
            sum(inference_times) / len(inference_times)
            if inference_times
            else 0
        )

        print("\n=== LIVE YOLO POSE REPORT ===")
        print("PROCESSED FRAMES:", processed_frames)
        print("ELAPSED:", round(elapsed, 2))
        print(
            "AVERAGE FPS:",
            round(processed_frames / max(elapsed, 0.001), 2),
        )
        print(
            "AVERAGE INFERENCE MS:",
            round(average_inference, 2),
        )

        print("SDK CLEANUP: COMPLETE")


if __name__ == "__main__":
    main()
