
import argparse
import json
import time
from ctypes import string_at
from getpass import getpass
from threading import Lock
from uuid import uuid4

import cv2
import numpy as np
import torch
from NetSDK.NetSDK import NetClient
from NetSDK.SDK_Callback import fDecCBFun, fRealDataCallBackEx2
from NetSDK.SDK_Enum import (
    EM_REALDATA_FLAG,
    SDK_RealPlayType,
)
from ultralytics import YOLO

from company_vision.behaviors.live_behavior_enhancements import (
    WeightedStillness,
    detect_precise_objects,
    render_head_status,
)
from company_vision.behaviors.live_behavior_evidence import (
    match_wrists,
)
from company_vision.behaviors.live_gpu_integration import (
    LiveBehaviorIntegrator,
    draw_behavior_labels,
)
from company_vision.vision.async_pose_pipeline import AsyncPosePipeline
from company_vision.vision.stable_pose_manager import StablePoseManager
from company_vision.vision.track_lifecycle_guard import TrackLifecycleGuard

# ============================================================
# IMPORTANT COCO KEYPOINTS
# ============================================================

# Head: nose, eyes, ears
# Shoulders: left and right
# Elbows: left and right

IMPORTANT_KEYPOINTS = (
    0,  # Nose
    1,  # Left Eye
    2,  # Right Eye
    3,  # Left Ear
    4,  # Right Ear
    5,  # Left Shoulder
    6,  # Right Shoulder
    7,  # Left Elbow
    8,  # Right Elbow
)

IMPORTANT_EDGES = (
    # Head
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),

    # Shoulders
    (5, 6),

    # Arms
    (5, 7),
    (6, 8),
)


# ============================================================
# DRAW STABLE POSE
# ============================================================

def draw_stable_pose(
    image,
    track_id,
    bbox,
    keypoints,
    alpha,
    confidence_threshold,
    states,
):
    """Draw only head, shoulder and elbow keypoints."""

    x1, y1, x2, y2 = map(int, bbox)

    diagonal = max(
        1.0,
        ((x2-x1)**2 + (y2-y1)**2)**0.5,
    )

    state = states.setdefault(
        track_id,
        {"points": None, "hits": 0, "missed": 0},
    )

    current = keypoints.copy()

    if state["points"] is not None:
        previous = state["points"].copy()

        for index in range(len(current)):
            if current[index, 2] < confidence_threshold:
                current[index] = previous[index]
                continue

            if previous[index, 2] >= confidence_threshold:
                distance = (
                    (current[index, 0] - previous[index, 0]) ** 2
                    + (current[index, 1] - previous[index, 1]) ** 2
                ) ** 0.5

                if distance > max(45.0, diagonal * 0.30):
                    current[index] = previous[index]
                    continue

                current[index, :2] = (
                    alpha * current[index, :2]
                    + (1.0 - alpha) * previous[index, :2]
                )

    state["points"] = current.copy()
    state["missed"] = 0

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 220, 0),
        2,
    )

    cv2.putText(
        image,
        f"ID {track_id}",
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 220, 0),
        2,
    )

    for first, second in IMPORTANT_EDGES:
        a = current[first]
        b = current[second]

        if (
            a[2] >= confidence_threshold
            and b[2] >= confidence_threshold
        ):
            p1 = tuple(np.round(a[:2]).astype(int))
            p2 = tuple(np.round(b[:2]).astype(int))

            cv2.line(
                image,
                p1,
                p2,
                (0, 255, 255),
                2,
            )

    for index in IMPORTANT_KEYPOINTS:
        point = current[index]

        if point[2] >= confidence_threshold:
            center = tuple(np.round(point[:2]).astype(int))

            cv2.circle(
                image,
                center,
                4,
                (255, 100, 0),
                -1,
            )

    return image


# ============================================================
# DRAW MANAGER TRACKS
# ============================================================


def draw_manager_tracks(image, person, confidence_threshold):
    """Draw observed person box and nine pose keypoints, without ID."""
    if not person.get("observed", False):
        return image

    height, width = image.shape[:2]
    box = np.asarray(person["bbox"], dtype=np.float32)

    if box.shape != (4,) or not np.isfinite(box).all():
        return image

    x1, y1, x2, y2 = np.round(box).astype(int)
    x1 = int(np.clip(x1, 0, width - 1))
    x2 = int(np.clip(x2, 0, width - 1))
    y1 = int(np.clip(y1, 0, height - 1))
    y2 = int(np.clip(y2, 0, height - 1))

    if x2 <= x1 or y2 <= y1:
        return image

    cv2.rectangle(
        image, (x1, y1), (x2, y2), (0, 220, 0), 2
    )

    points = np.asarray(person["keypoints"], dtype=np.float32)
    if points.shape != (9, 3):
        return image

    valid = (
        np.isfinite(points).all(axis=1)
        & (points[:, 2] >= confidence_threshold)
    )

    for first, second in IMPORTANT_EDGES:
        if valid[first] and valid[second]:
            a = tuple(np.round(points[first, :2]).astype(int))
            b = tuple(np.round(points[second, :2]).astype(int))
            cv2.line(image, a, b, (0, 255, 255), 2)

    for index in IMPORTANT_KEYPOINTS:
        if valid[index]:
            point = tuple(
                np.round(points[index, :2]).astype(int)
            )
            cv2.circle(image, point, 4, (255, 100, 0), -1)

    return image


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--ip", default="5.202.196.201")
    parser.add_argument("--port", type=int, default=37777)
    parser.add_argument("--channel", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=1280)

    parser.add_argument(
        "--annotation-copy",
        action="store_true",
        help="Copy the decoded frame before drawing (A/B baseline)",
    )
    parser.add_argument(
        "--gpu-precision",
        choices=("fp16", "fp32"),
        default="fp16",
        help="CUDA inference precision",
    )
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument(
        "--async-inference",
        action="store_true",
        help="Enable optional Node 4.5 asynchronous inference.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolo26x-pose.pt",
    )

    parser.add_argument(
        "--tracker",
        default=None,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--confirm-frames",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.40,
    )

    parser.add_argument(
        "--keypoint-conf",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--max-missed",
        type=int,
        default=60,
    )

    args = parser.parse_args()

    if args.confirm_frames < 1:
        parser.error("--confirm-frames must be >= 1")

    if not 0.0 < args.ema_alpha <= 1.0:
        parser.error("--ema-alpha must be in (0, 1]")

    if not 0.0 < args.keypoint_conf <= 1.0:
        parser.error("--keypoint-conf must be in (0, 1]")

    if not 0.0 < args.conf <= 1.0:
        parser.error("--conf must be in (0, 1]")

    if args.imgsz < 32:
        parser.error("--imgsz must be >= 32")

    if args.max_missed < 0:
        parser.error("--max-missed must be >= 0")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Loading YOLO Pose...")

    model = YOLO(args.model)

    if model.task != "pose":
        raise RuntimeError(
            "Selected model is not a Pose model."
        )

    print("POSE MODEL:", args.model)

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
        "capture_age_sum_ms": 0.0,
        "capture_age_max_ms": 0.0,
        "capture_samples": 0,
        "skipped_frames": 0,
    }

    # ========================================================
    # DECODE CALLBACK
    # ========================================================

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
            captured_at = time.monotonic()

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

    # ========================================================
    # REAL DATA CALLBACK
    # ========================================================

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
    last_processed_sequence = 0
    processed_frames = 0

    inference_times = []
    async_pipeline = None

    started_at = None
    last_report = time.monotonic()

    try:

        # ====================================================
        # SDK INITIALIZATION
        # ====================================================

        print("\nInitializing Dahua SDK...")

        if not sdk.InitEx():
            raise RuntimeError("SDK initialization failed")

        initialized = True


        # LEGACY_LOGIN_5102
        # Use the same NetSDK authentication method that passed
        # the real camera validation in Node 5.10.1.
        #
        # NetClient.InitEx() has already initialized the SDK.
        # WinDLL loads the same native SDK module and login
        # handle is subsequently used by NetClient.RealPlayEx.

        import ctypes
        from pathlib import Path

        native_dll = (
            Path(__file__).resolve().parents[1]
            / ".venv"
            / "Lib"
            / "site-packages"
            / "NetSDK"
            / "Libs"
            / "win64"
            / "dhnetsdk.dll"
        )

        if not native_dll.is_file():
            raise RuntimeError(
                f"Native Dahua SDK missing: {native_dll}"
            )


        sdk_native = ctypes.WinDLL(str(native_dll))

        sdk_native.CLIENT_LoginEx2.argtypes = [
            ctypes.c_char_p,
            ctypes.c_ushort,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        sdk_native.CLIENT_LoginEx2.restype = ctypes.c_longlong

        sdk_native.CLIENT_GetLastError.argtypes = []
        sdk_native.CLIENT_GetLastError.restype = ctypes.c_uint

        device_buffer = ctypes.create_string_buffer(4096)
        login_error = ctypes.c_int(0)

        # One authentication attempt; never retry locked accounts.
        login_id = int(
            sdk_native.CLIENT_LoginEx2(
                args.ip.encode("utf-8"),
                args.port,
                username.encode("utf-8"),
                password.encode("utf-8"),
                0,
                None,
                ctypes.byref(device_buffer),
                ctypes.byref(login_error),
            )
        )

        if not login_id:
            native_error = sdk_native.CLIENT_GetLastError()
            raise RuntimeError(
                "Dahua CLIENT_LoginEx2 failed: "
                f"device_error={login_error.value}, "
                f"sdk_error={native_error}"
            )

        print("LOGIN: PASS - CLIENT_LoginEx2")
        print("CHANNEL: 2 / SDK INDEX: 1")

        if args.channel != 2:
            raise ValueError(
                "Node 5.10.2 requires physical Channel 2"
            )


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
        print("ACTIVE KEYPOINTS: HEAD + SHOULDERS + ELBOWS")
        print("ACTIVE KEYPOINT COUNT: 9")
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

        cv2.resizeWindow(
            window_name,
            1280,
            720,
        )

        started_at = time.monotonic()

        track_lifecycle_guard = TrackLifecycleGuard(
            max_hidden_seconds=0.75,
            min_confirmed_observations=2,
        )

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

        if args.async_inference:
            async_pipeline = AsyncPosePipeline(
                model=model,
                imgsz=args.imgsz,
                conf=args.conf,
                gpu_precision=args.gpu_precision,
            )
            async_pipeline.start()
            print("NODE 4.5: ASYNC INFERENCE ACTIVE")
        else:
            print("NODE 4.5: ORIGINAL SYNC PATH ACTIVE")

        behavior_integrator = LiveBehaviorIntegrator(
            camera_id="dahua-channel-2",
            session_id=str(uuid4()),
        )
        behavior_processed = 0
        trace_path = Path(
            "artifacts/local_validation/node_5_10_2/"
            f"behavior_trace_{uuid4().hex[:10]}.jsonl"
        )
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        print("BEHAVIOR TRACE:", trace_path)
        print("NODE 5.10.2: FOUR BEHAVIOR PLUGINS ACTIVE")
        # NODE5102_OBJECT_DETECTION
        # NODE_5102_PRECISION_V2
        object_model = YOLO("yolo26l.pt")
        if object_model.task != "detect":
            raise RuntimeError("Not an object detection model")
        print("OBJECT DETECTOR: YOLO26n")
        print("GPU PRECISION:", args.gpu_precision.upper())
        print("ANNOTATION BUFFER:", "COPY" if args.annotation_copy else "IN_PLACE")

        print("\nLIVE YOLO POSE STARTED")
        print("Press Q to stop.")

        # ====================================================
        # LIVE PROCESSING LOOP
        # ====================================================

        stillness_tracker = WeightedStillness()
        wrist_coco = None
        wrist_sequence = -1

        while True:

            pose_batch = None

            if async_pipeline is not None:
                async_pipeline.check_error()

                async_result = async_pipeline.poll()

                if async_result is not None:
                    pose_batch = (
                        async_result.sequence,
                        async_result.frame,
                        async_result.coordinates,
                        async_result.poses,
                        async_result.scores,
                        async_result.inference_ms,
                    )

            with lock:
                item = state["latest"]

            if item is not None:
                sequence, raw, width, height, captured_at = item

                if sequence > last_sequence:

                    frame_age_ms = max(
                        0.0,
                        (time.monotonic() - captured_at) * 1000.0,
                    )

                    with lock:
                        state["skipped_frames"] += max(
                            0, sequence - last_sequence - 1
                        )
                        state["capture_samples"] += 1
                        state["capture_age_sum_ms"] += frame_age_ms
                        state["capture_age_max_ms"] = max(
                            state["capture_age_max_ms"],
                            frame_age_ms,
                        )

                    last_sequence = sequence

                    yuv = np.frombuffer(
                        raw,
                        dtype=np.uint8,
                    ).reshape((height * 3 // 2, width))

                    frame = cv2.cvtColor(
                        yuv,
                        cv2.COLOR_YUV2BGR_I420,
                    )

                    if async_pipeline is not None:

                        async_pipeline.submit(
                            sequence,
                            frame,
                            captured_at,
                        )

                    else:

                        inference_start = time.perf_counter()

                        results = model.predict(
                            source=frame,
                            device="cuda:0",
                            quantize=(
                                16 if args.gpu_precision == "fp16" else 32
                            ),
                            imgsz=args.imgsz,
                            conf=args.conf,
                            classes=[0],
                            verbose=False,
                        )

                        if not results:
                            raise RuntimeError(
                                "YOLO returned no results."
                            )

                        result = results[0]
                        boxes = result.boxes
                        keypoints_object = result.keypoints

                        if (
                            boxes is not None
                            and keypoints_object is not None
                            and keypoints_object.data is not None
                            and len(boxes) > 0
                        ):

                            coordinates = (
                                boxes.xyxy.detach().cpu().numpy()
                            )

                            scores = (
                                boxes.conf.detach().cpu().numpy()
                            )

                            coco_poses = (
                                keypoints_object.data.detach()
                                .cpu().numpy()
                            )

                            if (
                                coco_poses.ndim != 3
                                or coco_poses.shape[1:] != (17, 3)
                            ):
                                raise RuntimeError(
                                    "Expected COCO-17 keypoint structure."
                                )

                            wrist_coco = coco_poses.copy()
                            wrist_sequence = sequence

                            poses = coco_poses[
                                :, IMPORTANT_KEYPOINTS, :
                            ].astype(np.float32, copy=True)

                            if not (
                                len(coordinates)
                                == len(scores)
                                == len(poses)
                            ):
                                raise RuntimeError(
                                    "Detection/pose count mismatch."
                                )

                            if poses.shape[1:] != (9, 3):
                                raise RuntimeError(
                                    "Expected selected 9-keypoint structure."
                                )

                        else:

                            coordinates = np.empty(
                                (0, 4), dtype=np.float32
                            )
                            poses = np.empty(
                                (0, 9, 3), dtype=np.float32
                            )
                            scores = np.empty(
                                (0,), dtype=np.float32
                            )

                        inference_ms = (
                            time.perf_counter() - inference_start
                        ) * 1000.0

                        pose_batch = (
                            sequence,
                            frame,
                            coordinates,
                            poses,
                            scores,
                            inference_ms,
                        )

            # Poll again to avoid unnecessary frame-delivery delay.
            if async_pipeline is not None and pose_batch is None:

                async_result = async_pipeline.poll()

                if async_result is not None:
                    pose_batch = (
                        async_result.sequence,
                        async_result.frame,
                        async_result.coordinates,
                        async_result.poses,
                        async_result.scores,
                        async_result.inference_ms,
                    )

            if pose_batch is None:

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                time.sleep(0.005)
                continue

            (
                sequence,
                frame,
                coordinates,
                poses,
                scores,
                inference_ms,
            ) = pose_batch

            if sequence <= last_processed_sequence:
                continue

            last_processed_sequence = sequence
            inference_times.append(inference_ms)

            # =================================================
            # TEMPORAL TRACKING
            # =================================================

            tracked_persons = pose_manager.update(
                coordinates,
                poses,
                scores,
            )

            tracked_persons = track_lifecycle_guard.update(
                tracked_persons,
                timestamp=time.monotonic(),
            )

            # Detect visible objects for four behavior plugins.
            # COCO: laptop=63 mouse=64 keyboard=66 phone=67.
            detected_objects = detect_precise_objects(
                object_model, frame, tracked_persons,
                imgsz=1280,
            )

            # Match only wrists from this exact inference frame.
            wrist_evidence = match_wrists(
                tracked_persons,
                coordinates,
                (
                    wrist_coco
                    if wrist_sequence == sequence
                    else None
                ),
                frame.shape,
            )

            behavior_output = behavior_integrator.process(
                frame_id=processed_frames + 1,
                monotonic_timestamp=time.monotonic(),
                frame=frame,
                tracked_persons=tracked_persons,
                supplemental_wrists=wrist_evidence,
                objects=detected_objects,
            )
            behavior_processed += 1
            stillness_state = stillness_tracker.update(
                tracked_persons, time.monotonic(),
            )

            with trace_path.open('a', encoding='utf-8') as log:
                log.write(json.dumps({
                    "frame_id": behavior_processed,
                    "objects": detected_objects,
                    "wrist_track_ids": sorted(wrist_evidence),
                    "weighted_stillness": stillness_state,
                    "results": [
                        item.to_payload()
                        for item in behavior_output.results
                    ],
                }, ensure_ascii=False) + '\n')
            if len(behavior_output.results) != len(tracked_persons):
                raise RuntimeError(
                    "Behavior output track count mismatch"
                )

            annotated = (
                frame.copy() if args.annotation_copy else frame
            )

            person_count = sum(
                1
                for person in tracked_persons
                if person["observed"]
            )

            predicted_count = sum(
                1
                for person in tracked_persons
                if person["predicted"]
            )

            # =================================================
            # DRAW ONLY 9 IMPORTANT KEYPOINTS
            # =================================================

            for person in tracked_persons:

                annotated = draw_manager_tracks(
                    annotated,
                    person,
                    args.keypoint_conf,
                )

            annotated = render_head_status(
                annotated, behavior_output,
                detected_objects, tracked_persons, stillness_state,
            )

            annotated = draw_behavior_labels(
                annotated,
                tracked_persons,
                behavior_output,
            )

            # =================================================
            # PERFORMANCE METRICS
            # =================================================

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
                f"PERSONS: {person_count} | "
                f"PREDICTED: {predicted_count}",
                (25, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                annotated,
                f"FPS: {processing_fps:.2f} | "
                f"INFERENCE: {inference_ms:.1f} ms",
                (25, 116),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            cv2.imshow(
                window_name,
                annotated,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            now = time.monotonic()

            # =================================================
            # TERMINAL REPORT
            # =================================================

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

        if behavior_processed < 1:
            raise RuntimeError("No integrated GPU frames")
        print("NODE 5.10.2 LIVE GPU: PASS")
        print("BEHAVIOR FRAMES:", behavior_processed)
        print("BEHAVIOR OUTPUT:", behavior_integrator.summary())

    except KeyboardInterrupt:
        print("\nStopped by user")

    finally:

        print("\nStopping live processing...")

        if async_pipeline is not None:
            try:
                async_pipeline.stop(timeout=15)
                print("ASYNC WORKER: STOPPED")
                print("ASYNC STATS:", async_pipeline.stats())
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: Async shutdown issue: {exc}")

        cv2.destroyAllWindows()

        if play_id:
            sdk.StopRealPlayEx(play_id)

        if free_port is not None:

            try:
                sdk.SetDecCallBack(free_port, None)

            except Exception as exc:  # noqa: BLE001
                print(
                    f"WARNING: SDK callback cleanup failed: {exc}"
                )

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
            round(
                processed_frames / max(elapsed, 0.001),
                2,
            ),
        )

        print(
            "AVERAGE INFERENCE MS:",
            round(average_inference, 2),
        )
        with lock:
            samples = state["capture_samples"]
            age_sum = state["capture_age_sum_ms"]
            age_max = state["capture_age_max_ms"]
            skipped = state["skipped_frames"]
            decoded_total = state["decoded"]
            errors_total = state["errors"]

        print("\n=== NODE 4.2 CAPTURE REPORT ===")
        print("DECODED FRAMES:", decoded_total)
        print("SKIPPED SEQUENCES:", skipped)
        print("CAPTURE SAMPLES:", samples)
        print("FRAME AGE MEAN MS:", round(
            age_sum / max(samples, 1), 3
        ))
        print("FRAME AGE MAX MS:", round(age_max, 3))
        print("SDK INPUT ERRORS:", errors_total)
        print("SDK CLEANUP: COMPLETE")


if __name__ == "__main__":
    main()
