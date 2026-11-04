import argparse
import time
from ctypes import sizeof, string_at
from datetime import datetime
from getpass import getpass
from pathlib import Path
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
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--output-fps", type=float, default=5.0)
    args = parser.parse_args()

    if args.seconds <= 0 or args.output_fps <= 0:
        parser.error("seconds and output-fps must be positive")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    print("=== COMPANY VISION / DAHUA LIVE TEST ===")
    print("GPU:", torch.cuda.get_device_name(0))

    print("Loading YOLO Pose...")
    model = YOLO("yolo26n-pose.pt")

    username = input("Camera Username: ").strip()
    password = getpass("Camera Password: ")

    sdk = NetClient()

    login_id = 0
    play_id = 0
    free_port = None

    initialized = False
    stream_open = False
    player_started = False
    writer = None

    lock = Lock()

    state = {
        "latest": None,
        "sequence": 0,
        "packets": 0,
        "decoded": 0,
        "processed": 0,
        "decode_errors": 0,
        "input_failures": 0,
    }

    output_dir = Path("data/outputs/dahua_live")
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_video = output_dir / f"channel_{args.channel}_{stamp}.mp4"
    output_image = output_dir / f"channel_{args.channel}_{stamp}.jpg"

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

            # Keep only the newest frame.
            # Never queue unlimited frames.
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
                state["decode_errors"] += 1

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
                state["packets"] += 1

                if not accepted:
                    state["input_failures"] += 1

        except Exception:  # noqa: BLE001
            with lock:
                state["input_failures"] += 1

    decoding_callback = fDecCBFun(on_decoded)
    data_callback = fRealDataCallBackEx2(on_real_data)

    started_at = None
    last_sequence = 0
    last_report = 0
    frames_written = 0
    inference_times = []
    success = False

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

        login_id, info, error = sdk.LoginWithHighLevelSecurity(
            login_in,
            login_out,
        )

        if not login_id:
            raise RuntimeError(f"Login failed: {error}")

        print("LOGIN: PASS")
        print("CHANNEL COUNT:", info.nChanNum)

        if not 1 <= args.channel <= int(info.nChanNum):
            raise ValueError("Invalid channel number")

        channel_index = args.channel - 1

        success_port, free_port = sdk.GetFreePort()

        if not success_port:
            raise RuntimeError("GetFreePort failed")

        if not sdk.OpenStream(free_port):
            raise RuntimeError("OpenStream failed")

        stream_open = True

        if not sdk.Play(free_port, 0):
            raise RuntimeError("PlaySDK could not start")

        player_started = True

        if not sdk.SetDecCallBack(
            free_port,
            decoding_callback,
        ):
            raise RuntimeError("Decoder callback failed")

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
        print("YOLO: READY")
        print("Waiting for first frame...")

        # Allow a short startup window for the first decoded frame.
        startup_deadline = time.monotonic() + 15

        while time.monotonic() < startup_deadline:
            with lock:
                has_frame = state["latest"] is not None

            if has_frame:
                break

            time.sleep(0.05)

        if not has_frame:
            raise RuntimeError(
                "No decoded frame arrived during startup"
            )

        started_at = time.monotonic()
        deadline = started_at + args.seconds

        print("\nLIVE PROCESSING: STARTED")

        while time.monotonic() < deadline:
            with lock:
                item = state["latest"]

            if item is None:
                time.sleep(0.01)
                continue

            sequence, raw, width, height = item

            if sequence == last_sequence:
                time.sleep(0.01)
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

            infer_start = time.perf_counter()

            results = model.predict(
                source=frame,
                device="cuda:0",
                imgsz=args.imgsz,
                conf=args.conf,
                verbose=False,
            )

            inference_ms = (
                time.perf_counter() - infer_start
            ) * 1000

            inference_times.append(inference_ms)

            result = results[0]
            annotated = result.plot()

            person_count = (
                len(result.boxes)
                if result.boxes is not None
                else 0
            )

            elapsed = time.monotonic() - started_at

            cv2.putText(
                annotated,
                f"LIVE | CH {args.channel} | {elapsed:.1f}s",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                annotated,
                f"Persons: {person_count} | Infer: {inference_ms:.0f} ms",
                (25, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            if writer is None:
                height_out, width_out = annotated.shape[:2]

                writer = cv2.VideoWriter(
                    str(output_video),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    args.output_fps,
                    (width_out, height_out),
                )

                if not writer.isOpened():
                    raise RuntimeError("MP4 VideoWriter failed")

            writer.write(annotated)
            frames_written += 1

            with lock:
                state["processed"] += 1

            if not cv2.imwrite(str(output_image), annotated):
                raise RuntimeError("Failed to save latest JPG")

            current_second = int(elapsed)

            if current_second >= last_report + 5:
                last_report = current_second

                with lock:
                    decoded = state["decoded"]
                    processed = state["processed"]

                processing_fps = processed / max(elapsed, 0.001)

                print(
                    f"TIME={elapsed:.1f}s "
                    f"DECODED={decoded} "
                    f"PROCESSED={processed} "
                    f"FPS={processing_fps:.2f} "
                    f"PERSONS={person_count}"
                )

        success = frames_written > 0

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    except Exception as exc:  # noqa: BLE001
        print(
            "LIVE TEST ERROR:",
            type(exc).__name__,
            str(exc),
        )

    finally:
        if started_at is not None:
            elapsed_total = time.monotonic() - started_at
        else:
            elapsed_total = 0

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

        if writer is not None:
            writer.release()

        with lock:
            packets = state["packets"]
            decoded = state["decoded"]
            processed = state["processed"]
            decode_errors = state["decode_errors"]
            input_failures = state["input_failures"]

        avg_infer = (
            sum(inference_times) / len(inference_times)
            if inference_times
            else 0
        )

        actual_fps = (
            processed / elapsed_total
            if elapsed_total > 0
            else 0
        )

        skipped = max(0, decoded - processed)

        print("\n=== FINAL LIVE TEST REPORT ===")
        print("ELAPSED SECONDS:", round(elapsed_total, 2))
        print("PACKETS:", packets)
        print("DECODED FRAMES:", decoded)
        print("PROCESSED FRAMES:", processed)
        print("SKIPPED FRAMES:", skipped)
        print("PROCESSING FPS:", round(actual_fps, 2))
        print("AVG INFERENCE MS:", round(avg_infer, 2))
        print("DECODE ERRORS:", decode_errors)
        print("INPUT FAILURES:", input_failures)
        print("VIDEO FRAMES WRITTEN:", frames_written)

        if frames_written:
            print("VIDEO OUTPUT:", output_video)
            print("IMAGE OUTPUT:", output_image)

        if success and decode_errors == 0 and input_failures == 0:
            print("DAHUA CONTINUOUS LIVE TEST: PASS")
        elif frames_written:
            print("DAHUA CONTINUOUS LIVE TEST: PARTIAL")
        else:
            print("DAHUA CONTINUOUS LIVE TEST: FAIL")

        print("SDK CLEANUP: COMPLETE")


if __name__ == "__main__":
    main()
