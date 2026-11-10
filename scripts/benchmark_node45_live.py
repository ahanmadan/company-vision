
from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from ctypes import sizeof, string_at
from datetime import datetime
from getpass import getpass
from pathlib import Path
from threading import Lock
from time import monotonic, perf_counter

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

from company_vision.vision.async_pose_pipeline import (
    AsyncPosePipeline,
    PoseResult,
)
from company_vision.vision.stable_pose_manager import StablePoseManager
from company_vision.vision.track_lifecycle_guard import (
    TrackLifecycleGuard,
)


def percentile(values, p):
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def infer_sync(model, frame, sequence, captured_at, args):
    started = perf_counter()

    results = model.predict(
        source=frame,
        device="cuda:0",
        quantize=(16 if args.gpu_precision == "fp16" else 32),
        imgsz=args.imgsz,
        conf=args.conf,
        classes=[0],
        verbose=False,
    )

    if not results:
        raise RuntimeError("YOLO returned no results")

    result = results[0]
    boxes = result.boxes
    keypoints = result.keypoints

    if (
        boxes is not None
        and keypoints is not None
        and keypoints.data is not None
        and len(boxes) > 0
    ):
        coordinates = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()

        coco = keypoints.data.detach().cpu().numpy()

        if coco.ndim != 3 or coco.shape[1:] != (17, 3):
            raise RuntimeError("Invalid COCO pose shape")

        poses = coco[:, list(range(9)), :].astype(
            np.float32, copy=True
        )

        if not (
            len(coordinates) == len(scores) == len(poses)
        ):
            raise RuntimeError("Detection/pose mismatch")

    else:
        coordinates = np.empty((0, 4), dtype=np.float32)
        scores = np.empty((0,), dtype=np.float32)
        poses = np.empty((0, 9, 3), dtype=np.float32)

    return PoseResult(
        sequence=sequence,
        frame=frame,
        captured_at=captured_at,
        completed_at=monotonic(),
        coordinates=coordinates,
        poses=poses,
        scores=scores,
        inference_ms=(perf_counter() - started) * 1000,
    )


def warmup(model, frame, args):
    print("CUDA WARMUP...")

    for index in range(args.warmup_runs):
        infer_sync(
            model,
            frame,
            index,
            monotonic(),
            args,
        )

    torch.cuda.synchronize()
    print("WARMUP: PASS")


def draw_preview(frame, persons, mode, fps, latency):
    image = frame.copy()

    for person in persons:
        if not person.get("observed", False):
            continue

        box = np.asarray(person["bbox"], dtype=np.float32)

        if box.shape != (4,) or not np.isfinite(box).all():
            continue

        x1, y1, x2, y2 = np.round(box).astype(int)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 220, 0),
            2,
        )

        points = np.asarray(
            person["keypoints"], dtype=np.float32
        )

        if points.shape != (9, 3):
            continue

        for point in points:
            if (
                np.isfinite(point).all()
                and point[2] >= 0.25
            ):
                cv2.circle(
                    image,
                    tuple(np.round(point[:2]).astype(int)),
                    4,
                    (0, 255, 255),
                    -1,
                )

    cv2.putText(
        image,
        f"MODE: {mode.upper()}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        image,
        f"FPS: {fps:.2f} | LATENCY: {latency:.1f} ms",
        (20, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    cv2.imshow("NODE 4.5 LIVE BENCHMARK", image)


def run_phase(
    mode,
    repetition,
    model,
    state,
    lock,
    args,
    window_name,
):
    print("\n" + "=" * 60)
    print(f"PHASE: {mode.upper()} | REPETITION: {repetition}")
    print("=" * 60)

    with lock:
        item = state["latest"]

    if item is None:
        raise RuntimeError("No live camera frame")

    _, raw, width, height, _ = item

    frame = cv2.cvtColor(
        np.frombuffer(raw, dtype=np.uint8).reshape(
            height * 3 // 2, width
        ),
        cv2.COLOR_YUV2BGR_I420,
    )

    warmup(model, frame, args)

    pose_manager = StablePoseManager(
        confirm_hits=args.confirm_frames,
        max_missed=args.max_missed,
        min_keypoints=3,
        keypoint_conf=args.keypoint_conf,
        ema_alpha=args.ema_alpha,
        match_distance=0.85,
        pose_filter_mode="one_euro",
        new_track_min_conf=0.50,
    )

    lifecycle_guard = TrackLifecycleGuard(
        max_hidden_seconds=0.75,
        min_confirmed_observations=2,
    )

    pipeline = None

    if mode == "async":
        pipeline = AsyncPosePipeline(
            model=model,
            imgsz=args.imgsz,
            conf=args.conf,
            gpu_precision=args.gpu_precision,
        )
        pipeline.start()

    inference_samples = []
    completion_latency = []
    delivery_latency = []

    processed = 0
    skipped_sequences = 0
    detected_total = 0
    observed_total = 0
    predicted_total = 0
    valid_keypoints = 0
    possible_keypoints = 0

    start = monotonic()
    deadline = start + args.duration
    torch.cuda.reset_peak_memory_stats()

    stopped_by_user = False

    torch.cuda.synchronize()

    # Exclude camera frames received during model warmup.
    with lock:
        initial_sequence = state["sequence"]
        decoded_start = state["decoded"]

    last_sequence = initial_sequence

    def consume(result):
        nonlocal processed
        nonlocal detected_total
        nonlocal observed_total
        nonlocal predicted_total
        nonlocal valid_keypoints
        nonlocal possible_keypoints

        persons = pose_manager.update(
            result.coordinates,
            result.poses,
            result.scores,
        )

        persons = lifecycle_guard.update(
            persons,
            timestamp=monotonic(),
        )

        processed += 1
        detected_total += len(result.scores)

        observed_total += sum(
            1 for p in persons if p.get("observed", False)
        )

        predicted_total += sum(
            1 for p in persons if p.get("predicted", False)
        )

        valid_keypoints += int(
            np.count_nonzero(result.poses[:, :, 2] >= args.keypoint_conf)
        )

        possible_keypoints += result.poses.shape[0] * 9

        now = monotonic()

        inference_samples.append(result.inference_ms)

        completion_latency.append(
            max(0.0, (result.completed_at - result.captured_at) * 1000)
        )

        delivery_latency.append(
            max(0.0, (now - result.captured_at) * 1000)
        )

        elapsed = max(now - start, 0.001)
        current_fps = processed / elapsed

        if args.display:
            draw_preview(
                result.frame,
                persons,
                mode,
                current_fps,
                delivery_latency[-1],
            )

    try:
        while monotonic() < deadline:
            if pipeline is not None:
                pipeline.check_error()

            with lock:
                item = state["latest"]

            if item is not None:
                sequence, raw, width, height, captured_at = item

                if sequence > last_sequence:
                    skipped_sequences += max(
                        0,
                        sequence - last_sequence - 1,
                    )

                    last_sequence = sequence

                    frame = cv2.cvtColor(
                        np.frombuffer(
                            raw, dtype=np.uint8
                        ).reshape(
                            height * 3 // 2, width
                        ),
                        cv2.COLOR_YUV2BGR_I420,
                    )

                    if mode == "sync":
                        result = infer_sync(
                            model,
                            frame,
                            sequence,
                            captured_at,
                            args,
                        )
                        consume(result)

                    else:
                        pipeline.submit(
                            sequence,
                            frame,
                            captured_at,
                        )

            if pipeline is not None:
                result = pipeline.poll()

                if result is not None:
                    consume(result)

            if args.display:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stopped_by_user = True
                    break
            else:
                time.sleep(0.001)

    finally:
        if pipeline is not None:
            pipeline.stop(timeout=30)

            # Include a completed final result if it belongs
            # to the measurement window.
            final_result = pipeline.poll()

            if (
                final_result is not None
                and final_result.completed_at <= deadline
            ):
                consume(final_result)

    torch.cuda.synchronize()

    elapsed = monotonic() - start

    with lock:
        decoded_end = state["decoded"]

    decoded = decoded_end - decoded_start

    pipeline_stats = (
        pipeline.stats()
        if pipeline is not None
        else {
            "submitted": processed,
            "processed": processed,
            "dropped_input": 0,
            "dropped_output": 0,
            "average_inference_ms": (
                float(np.mean(inference_samples))
                if inference_samples else 0.0
            ),
        }
    )

    keypoint_ratio = (
        valid_keypoints / possible_keypoints
        if possible_keypoints else 0.0
    )

    row = {
        "mode": mode,
        "repetition": repetition,
        "duration_seconds": round(elapsed, 3),
        "decoded_frames": decoded,
        "submitted_frames": pipeline_stats["submitted"],
        "processed_frames": pipeline_stats["processed"],
        "worker_processed": pipeline_stats["processed"],
        "tracking_updates": processed,
        "processed_fps": round(
            pipeline_stats["processed"] / max(elapsed, 0.001), 3
        ),
        "tracking_fps": round(
            processed / max(elapsed, 0.001), 3
        ),
        "source_fps": round(decoded / max(elapsed, 0.001), 3),
        "skipped_sequences": skipped_sequences,
        "queue_dropped_input": pipeline_stats["dropped_input"],
        "queue_dropped_output": pipeline_stats["dropped_output"],
        "inference_mean_ms": round(
            float(np.mean(inference_samples))
            if inference_samples else 0.0, 3
        ),
        "inference_p95_ms": round(
            percentile(inference_samples, 95), 3
        ),
        "completion_latency_mean_ms": round(
            float(np.mean(completion_latency))
            if completion_latency else 0.0, 3
        ),
        "completion_latency_p95_ms": round(
            percentile(completion_latency, 95), 3
        ),
        "delivery_latency_mean_ms": round(
            float(np.mean(delivery_latency))
            if delivery_latency else 0.0, 3
        ),
        "delivery_latency_p95_ms": round(
            percentile(delivery_latency, 95), 3
        ),
        "detection_total": detected_total,
        "observed_total": observed_total,
        "predicted_total": predicted_total,
        "keypoint_valid_ratio": round(keypoint_ratio, 4),
        "gpu_peak_memory_mb": round(
            torch.cuda.max_memory_allocated() / (1024 ** 2), 2
        ),
    }

    print("\nPHASE RESULT")

    for key, value in row.items():
        print(f"{key}: {value}")

    return row, stopped_by_user


def main():
    parser = argparse.ArgumentParser(
        description="Node 4.5 direct Dahua Sync/Async benchmark"
    )

    parser.add_argument("--ip", default="5.202.196.201")
    parser.add_argument("--port", type=int, default=37777)
    parser.add_argument("--channel", type=int, default=2)

    parser.add_argument("--model", default="yolo26x-pose.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.30)

    parser.add_argument(
        "--gpu-precision",
        choices=("fp16", "fp32"),
        default="fp16",
    )

    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup-runs", type=int, default=5)

    parser.add_argument("--confirm-frames", type=int, default=3)
    parser.add_argument("--max-missed", type=int, default=60)
    parser.add_argument("--ema-alpha", type=float, default=0.40)
    parser.add_argument("--keypoint-conf", type=float, default=0.25)

    parser.add_argument("--no-display", action="store_true")

    args = parser.parse_args()
    args.display = not args.no_display

    if args.duration < 10 or args.repeats < 1:
        parser.error("duration must be >= 10 and repeats >= 1")

    if args.warmup_runs < 1:
        parser.error("warmup-runs must be >= 1")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    print("GPU:", torch.cuda.get_device_name(0))
    print("MODEL:", args.model)

    model = YOLO(args.model)

    if model.task != "pose":
        raise RuntimeError("Model must be a pose model")

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
        "decoded": 0,
        "packets": 0,
        "errors": 0,
    }

    def on_decoded(
        port, buffer, buffer_size, frame_info,
        user_data, reserved,
    ):
        try:
            info = frame_info.contents

            if info.nType != 3:
                return

            width = int(info.nWidth)
            height = int(info.nHeight)

            if (
                width <= 0 or height <= 0
                or width % 2 or height % 2
            ):
                return

            expected = width * height * 3 // 2

            if buffer_size < expected:
                return

            raw = string_at(buffer, expected)
            captured_at = monotonic()

            with lock:
                state["sequence"] += 1
                state["decoded"] += 1

                state["latest"] = (
                    state["sequence"],
                    raw,
                    width,
                    height,
                    captured_at,
                )

        except Exception:  # noqa: BLE001
            with lock:
                state["errors"] += 1

    def on_real_data(
        real_handle, data_type, buffer,
        buffer_size, param, user_data,
    ):
        if buffer_size <= 0:
            return

        try:
            accepted = sdk.InputData(
                free_port,
                buffer,
                buffer_size,
            )

            if accepted:
                with lock:
                    state["packets"] += 1
            else:
                with lock:
                    state["errors"] += 1

        except Exception:  # noqa: BLE001
            with lock:
                state["errors"] += 1

    decoding_callback = fDecCBFun(on_decoded)
    data_callback = fRealDataCallBackEx2(on_real_data)

    rows = []

    try:
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

        if not 1 <= args.channel <= int(device_info.nChanNum):
            raise ValueError("Invalid camera channel")

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
            args.channel - 1,
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

        print("\nDAHUA LIVE STREAM: ACTIVE")

        startup_deadline = monotonic() + 20

        while monotonic() < startup_deadline:
            with lock:
                available = state["latest"] is not None

            if available:
                break

            time.sleep(0.05)

        if not available:
            raise RuntimeError("No live camera frames")

        if args.display:
            cv2.namedWindow(
                "NODE 4.5 LIVE BENCHMARK",
                cv2.WINDOW_NORMAL,
            )
            cv2.resizeWindow(
                "NODE 4.5 LIVE BENCHMARK",
                1280,
                720,
            )

        order = []

        for index in range(args.repeats):
            pair = (
                ("sync", "async")
                if index % 2 == 0
                else ("async", "sync")
            )
            order.extend(pair)

        print("\nEXPERIMENT ORDER:", order)
        print("LIVE CAMERA TEST ONLY")

        for repetition, mode in enumerate(order, start=1):
            row, stopped = run_phase(
                mode,
                repetition,
                model,
                state,
                lock,
                args,
                "NODE 4.5 LIVE BENCHMARK",
            )

            rows.append(row)

            if stopped:
                break

        if rows:
            reports = Path("reports")
            reports.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            output = reports / (
                f"node45_live_benchmark_{timestamp}.csv"
            )

            with output.open(
                "w",
                newline="",
                encoding="utf-8-sig",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=list(rows[0].keys()),
                )
                writer.writeheader()
                writer.writerows(rows)

            print("\nREPORT:", output)

            grouped = defaultdict(list)

            for row in rows:
                grouped[row["mode"]].append(row)

            print("\n=== SYNC / ASYNC COMPARISON ===")

            for mode in ("sync", "async"):
                values = grouped[mode]

                if not values:
                    continue

                fps = np.mean([
                    row["processed_fps"] for row in values
                ])

                latency = np.mean([
                    row["delivery_latency_p95_ms"]
                    for row in values
                ])

                print(
                    f"{mode.upper()}: "
                    f"FPS={fps:.2f} "
                    f"DELIVERY_P95={latency:.2f}ms"
                )

            if grouped["sync"] and grouped["async"]:
                sync_fps = np.mean([
                    r["processed_fps"] for r in grouped["sync"]
                ])
                async_fps = np.mean([
                    r["processed_fps"] for r in grouped["async"]
                ])

                improvement = (
                    (async_fps / sync_fps - 1) * 100
                    if sync_fps else 0
                )

                print(
                    f"\nASYNC FPS CHANGE: {improvement:+.2f}%"
                )

    except KeyboardInterrupt:
        print("\nBenchmark interrupted")

    finally:
        cv2.destroyAllWindows()

        if play_id:
            sdk.StopRealPlayEx(play_id)

        if free_port is not None:
            try:
                sdk.SetDecCallBack(free_port, None)
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: Decoder callback cleanup failed: {exc}")

            if player_started:
                sdk.Stop(free_port)

            if stream_open:
                sdk.CloseStream(free_port)

            sdk.ReleasePort(free_port)

        if login_id:
            sdk.Logout(login_id)

        if initialized:
            sdk.Cleanup()

        print("\nDAHUA SDK CLEANUP: COMPLETE")


if __name__ == "__main__":
    main()
