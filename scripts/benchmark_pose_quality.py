
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from ctypes import sizeof, string_at
from datetime import datetime
from getpass import getpass
from pathlib import Path
from threading import Lock

import cv2
import matplotlib
import matplotlib.pyplot as plt
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

from company_vision.vision.stable_pose_manager import (
    KEYPOINT_COUNT,
    KEYPOINT_NAMES,
    StablePoseManager,
)

matplotlib.use("Agg")

# ============================================================
# PROJECT CONTRACT: EXACTLY 9 KEYPOINTS
# ============================================================

KEYPOINTS = {
    index: name
    for index, name in enumerate(KEYPOINT_NAMES)
}

EDGES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (6, 8),
)

assert KEYPOINT_COUNT == 9
assert len(KEYPOINTS) == 9


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Dahua YOLO26x Pose Benchmark "
            "with 9-Keypoint Dynamic Weighting"
        )
    )

    parser.add_argument("--ip", default="5.202.196.201")
    parser.add_argument("--port", type=int, default=37777)
    parser.add_argument("--channel", type=int, default=2)

    parser.add_argument(
        "--model",
        default="yolo26x-pose.pt",
    )

    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)

    parser.add_argument(
        "--keypoint-conf",
        type=float,
        default=0.25,
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
        "--max-missed",
        type=int,
        default=60,
    )

    parser.add_argument("--max-frames", type=int, default=0)

    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--video-fps",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--output",
        default="runs/pose_benchmark",
    )

    parser.add_argument("--device", default="0")

    parser.add_argument(
        "--no-display",
        action="store_true",
    )

    args = parser.parse_args()

    if args.channel < 1:
        parser.error("channel must be >= 1")

    if not 1 <= args.port <= 65535:
        parser.error("invalid port")

    if args.imgsz < 32:
        parser.error("invalid imgsz")

    if not 0 < args.conf <= 1:
        parser.error("invalid confidence")

    if not 0 < args.keypoint_conf <= 1:
        parser.error("invalid keypoint confidence")

    if args.confirm_frames < 1 or args.max_missed < 0:
        parser.error("invalid tracking limits")

    if not 0 < args.ema_alpha <= 1:
        parser.error("invalid ema_alpha")

    if args.duration < 0 or args.max_frames < 0:
        parser.error("invalid frame limit")

    if args.video_fps <= 0:
        parser.error("invalid video_fps")

    return args


# ============================================================
# STATISTICS
# ============================================================

def safe_mean(values):
    return float(np.mean(values)) if values else None


def safe_std(values):
    return float(np.std(values)) if values else None


def safe_percentile(values, percentile):
    return (
        float(np.percentile(values, percentile))
        if values else None
    )


def create_keypoint_storage():
    return {
        index: {
            "confidence": [],
            "displacement": [],
            "normalized_displacement": [],
            "acceleration": [],
            "normalized_acceleration": [],
            "missing": 0,
            "total": 0,
        }
        for index in range(KEYPOINT_COUNT)
    }


def create_track_storage():
    return defaultdict(
        lambda: {
            "observed": 0,
            "predicted": 0,
            "first_frame": None,
            "last_frame": None,
            "consecutive_observed": 0,
            "consecutive_predicted": 0,
            "max_consecutive_observed": 0,
            "max_consecutive_predicted": 0,
        }
    )


def body_scale(box):
    width = max(1.0, float(box[2] - box[0]))
    height = max(1.0, float(box[3] - box[1]))
    return math.hypot(width, height)


# ============================================================
# TEMPORAL METRICS
# ============================================================

def update_point_metric(
    storage,
    previous,
    track_id,
    index,
    x,
    y,
    confidence,
    scale,
    frame_number,
    threshold,
):
    values = storage[index]
    values["total"] += 1

    valid = confidence >= threshold

    result = {
        "valid": valid,
        "dx": None,
        "dy": None,
        "displacement": None,
        "normalized_displacement": None,
        "acceleration": None,
        "normalized_acceleration": None,
        "frame_gap": None,
    }

    key = (track_id, index)

    if not valid:
        values["missing"] += 1
        previous.pop(key, None)
        return result

    values["confidence"].append(confidence)

    old = previous.get(key)

    velocity_x = 0.0
    velocity_y = 0.0

    if old is not None:
        (
            old_x,
            old_y,
            old_frame,
            old_scale,
            old_vx,
            old_vy,
        ) = old

        frame_gap = frame_number - old_frame

        if frame_gap > 0:
            dx = x - old_x
            dy = y - old_y

            displacement = math.hypot(dx, dy)

            reference_scale = max(
                1.0,
                (scale + old_scale) / 2.0,
            )

            normalized = (
                displacement / reference_scale
            )

            velocity_x = dx / frame_gap
            velocity_y = dy / frame_gap

            acceleration = math.hypot(
                velocity_x - old_vx,
                velocity_y - old_vy,
            )

            normalized_acceleration = (
                acceleration / reference_scale
            )

            result.update({
                "dx": dx,
                "dy": dy,
                "displacement": displacement,
                "normalized_displacement": normalized,
                "acceleration": acceleration,
                "normalized_acceleration": (
                    normalized_acceleration
                ),
                "frame_gap": frame_gap,
            })

            values["displacement"].append(displacement)
            values["normalized_displacement"].append(
                normalized
            )
            values["acceleration"].append(acceleration)
            values["normalized_acceleration"].append(
                normalized_acceleration
            )

    previous[key] = (
        x,
        y,
        frame_number,
        scale,
        velocity_x,
        velocity_y,
    )

    return result


def summarize_keypoint(values):
    total = values["total"]
    missing = values["missing"]

    return {
        "total_observations": total,
        "valid_observations": total - missing,
        "missing_observations": missing,
        "valid_rate_percent": (
            round(100.0 * (total - missing) / total, 3)
            if total else None
        ),
        "mean_confidence": safe_mean(
            values["confidence"]
        ),
        "mean_displacement_pixels": safe_mean(
            values["displacement"]
        ),
        "p95_displacement_pixels": safe_percentile(
            values["displacement"],
            95,
        ),
        "mean_normalized_displacement": safe_mean(
            values["normalized_displacement"]
        ),
        "mean_acceleration_pixels": safe_mean(
            values["acceleration"]
        ),
        "p95_acceleration_pixels": safe_percentile(
            values["acceleration"],
            95,
        ),
        "mean_normalized_acceleration": safe_mean(
            values["normalized_acceleration"]
        ),
    }


# ============================================================
# TRACKING METRICS
# ============================================================

def update_tracking_metrics(
    storage,
    person,
    frame_number,
):
    track_id = int(person["track_id"])
    values = storage[track_id]

    if values["first_frame"] is None:
        values["first_frame"] = frame_number

    values["last_frame"] = frame_number

    if person["observed"]:
        values["observed"] += 1
        values["consecutive_observed"] += 1
        values["consecutive_predicted"] = 0

        values["max_consecutive_observed"] = max(
            values["max_consecutive_observed"],
            values["consecutive_observed"],
        )

    elif person["predicted"]:
        values["predicted"] += 1
        values["consecutive_predicted"] += 1
        values["consecutive_observed"] = 0

        values["max_consecutive_predicted"] = max(
            values["max_consecutive_predicted"],
            values["consecutive_predicted"],
        )


# ============================================================
# DRAWING
# ============================================================

def draw_person(frame, person, threshold):
    bbox = np.round(
        person["bbox"]
    ).astype(int)

    x1, y1, x2, y2 = bbox

    observed = bool(person["observed"])

    color = (
        (0, 220, 0)
        if observed
        else (0, 165, 255)
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    label = (
        f"ID:{person['track_id']} "
        f"PERSON:{person['confidence']:.2f}"
    )

    if not observed:
        label += " PREDICTED"

    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )

    if not observed:
        return

    points = person["keypoints"]
    weights = person["keypoint_weights"]

    valid = {}

    for index in range(KEYPOINT_COUNT):
        x, y, confidence = points[index]

        if confidence >= threshold:
            valid[index] = (
                round(float(x)),
                round(float(y)),
            )

    for first, second in EDGES:
        if first in valid and second in valid:
            cv2.line(
                frame,
                valid[first],
                valid[second],
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    # Exactly 9 keypoints.
    for index in KEYPOINTS:
        if index not in valid:
            continue

        weight = float(weights[index])

        if weight >= 0.65:
            point_color = (0, 220, 0)
        elif weight >= 0.40:
            point_color = (0, 200, 255)
        else:
            point_color = (0, 80, 255)

        center = valid[index]

        cv2.circle(
            frame,
            center,
            5,
            point_color,
            -1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"{index}:{weight:.2f}",
            (center[0] + 5, center[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            point_color,
            1,
            cv2.LINE_AA,
        )


# ============================================================
# PLOTS
# ============================================================

def save_plot(
    names,
    values,
    title,
    filename,
    output_dir,
):
    plt.figure(figsize=(13, 5))

    plt.bar(names, values)

    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        output_dir / filename,
        dpi=160,
    )

    plt.close()


def create_plots(
    raw_metrics,
    filtered_metrics,
    dynamic_weights,
    offsets,
    output_dir,
):
    names = list(KEYPOINT_NAMES)

    raw_acceleration = []
    filtered_acceleration = []
    reductions = []
    weights = []
    mean_offsets = []

    for index in range(KEYPOINT_COUNT):
        raw = (
            safe_mean(
                raw_metrics[index]["normalized_acceleration"]
            ) or 0.0
        )

        filtered = (
            safe_mean(
                filtered_metrics[index]["normalized_acceleration"]
            ) or 0.0
        )

        raw_acceleration.append(raw)
        filtered_acceleration.append(filtered)

        reductions.append(
            100.0 * (raw - filtered) / raw
            if raw > 0 else 0.0
        )

        weights.append(
            safe_mean(dynamic_weights[index]) or 0.0
        )

        mean_offsets.append(
            safe_mean(offsets[index]) or 0.0
        )

    save_plot(
        names,
        raw_acceleration,
        "Raw Normalized Acceleration",
        "raw_keypoint_acceleration.png",
        output_dir,
    )

    save_plot(
        names,
        filtered_acceleration,
        "Filtered Normalized Acceleration",
        "filtered_keypoint_acceleration.png",
        output_dir,
    )

    save_plot(
        names,
        reductions,
        "Acceleration Reduction (%)",
        "keypoint_acceleration_reduction.png",
        output_dir,
    )

    save_plot(
        names,
        weights,
        "Mean Dynamic Joint Weights",
        "dynamic_joint_weights.png",
        output_dir,
    )

    save_plot(
        names,
        mean_offsets,
        "Mean Filter Offset (Pixels)",
        "keypoint_filter_offset.png",
        output_dir,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available")

    device_index = (
        int(args.device)
        if args.device.isdigit()
        else 0
    )

    if device_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device {device_index} is unavailable"
        )

    print("=" * 65)
    print("DAHUA 9-KEYPOINT DYNAMIC WEIGHTING BENCHMARK")
    print("=" * 65)

    print("GPU:", torch.cuda.get_device_name(device_index))
    print("CHANNEL:", args.channel)
    print("MODEL:", args.model)
    print("KEYPOINT COUNT:", KEYPOINT_COUNT)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_dir = Path(args.output) / timestamp
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_video_path = output_dir / "raw_test.mp4"
    annotated_video_path = output_dir / "pose_annotated.mp4"

    pose_csv_path = output_dir / "pose_metrics.csv"
    tracking_csv_path = output_dir / "tracking_metrics.csv"
    report_path = output_dir / "pose_report.json"

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    print("\nLoading YOLO Pose model...")

    model = YOLO(args.model)

    if model.task != "pose":
        raise RuntimeError(
            "Selected model is not a Pose model"
        )

    print("YOLO POSE: READY")

    # --------------------------------------------------------
    # CAMERA CREDENTIALS
    # --------------------------------------------------------

    username = input("Camera Username: ").strip()
    password = getpass("Camera Password: ")

    if not username or not password:
        raise ValueError(
            "Camera credentials cannot be empty"
        )

    # --------------------------------------------------------
    # SDK STATE
    # --------------------------------------------------------

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

    raw_writer = None
    annotated_writer = None

    last_sequence = 0

    processed_frames = 0
    frames_with_people = 0
    total_detections = 0

    inference_times = []
    manager_times = []
    frame_times = []

    person_confidences = []

    raw_metrics = create_keypoint_storage()
    filtered_metrics = create_keypoint_storage()

    tracking_metrics = create_track_storage()

    previous_raw = {}
    previous_filtered = {}

    dynamic_weights = {
        index: []
        for index in range(KEYPOINT_COUNT)
    }

    filter_offsets = {
        index: []
        for index in range(KEYPOINT_COUNT)
    }

    pose_manager = StablePoseManager(
        confirm_hits=args.confirm_frames,
        max_missed=args.max_missed,
        min_keypoints=3,
        keypoint_conf=args.keypoint_conf,
        ema_alpha=args.ema_alpha,
        match_distance=0.85,
    )

    started_at = None
    last_report = time.monotonic()

    # --------------------------------------------------------
    # DECODE CALLBACK
    # --------------------------------------------------------

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

            if (
                width <= 0
                or height <= 0
                or width % 2
                or height % 2
            ):
                return

            expected = width * height * 3 // 2

            if buffer_size < expected:
                return

            raw = string_at(
                buffer,
                expected,
            )

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

    # --------------------------------------------------------
    # REAL DATA CALLBACK
    # --------------------------------------------------------

    def on_real_data(
        real_handle,
        data_type,
        buffer,
        buffer_size,
        param,
        user_data,
    ):
        if buffer_size <= 0 or free_port is None:
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

    window_name = "Dahua 9-Keypoint Benchmark"

    try:
        # ----------------------------------------------------
        # CAMERA INITIALIZATION
        # ----------------------------------------------------

        print("\nInitializing Dahua SDK...")

        if not sdk.InitEx():
            raise RuntimeError(
                "SDK initialization failed"
            )

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
            raise RuntimeError(
                f"Login failed: {error}"
            )

        print("LOGIN: PASS")
        print("CHANNELS:", device_info.nChanNum)

        if not 1 <= args.channel <= int(device_info.nChanNum):
            raise ValueError(
                "Invalid camera channel"
            )

        success, free_port = sdk.GetFreePort()

        if not success:
            raise RuntimeError(
                "GetFreePort failed"
            )

        if not sdk.OpenStream(free_port):
            raise RuntimeError(
                "OpenStream failed"
            )

        stream_open = True

        if not sdk.Play(free_port, 0):
            raise RuntimeError(
                "Decoder Play failed"
            )

        player_started = True

        if not sdk.SetDecCallBack(
            free_port,
            decoding_callback,
        ):
            raise RuntimeError(
                "SetDecCallBack failed"
            )

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
            raise RuntimeError(
                "SetRealDataCallBackEx2 failed"
            )

        print("REALPLAY: PASS")
        print("NETSDK RAW DATA: READY")

        startup_deadline = time.monotonic() + 20

        while time.monotonic() < startup_deadline:
            with lock:
                available = state["latest"] is not None

            if available:
                break

            time.sleep(0.05)

        else:
            raise RuntimeError(
                "No decoded camera frame received"
            )

        if not args.no_display:
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

        print("\nLIVE BENCHMARK STARTED")
        print("Press Q or Ctrl+C to stop.")

        # ----------------------------------------------------
        # CSV SCHEMAS
        # ----------------------------------------------------

        pose_columns = [
            "frame",
            "elapsed_seconds",
            "track_id",
            "keypoint_index",
            "keypoint_name",
            "raw_x",
            "raw_y",
            "raw_confidence",
            "filtered_x",
            "filtered_y",
            "filtered_confidence",
            "raw_valid",
            "filtered_valid",
            "raw_displacement_px",
            "filtered_displacement_px",
            "raw_normalized_displacement",
            "filtered_normalized_displacement",
            "raw_acceleration_px",
            "filtered_acceleration_px",
            "dynamic_joint_weight",
            "filter_offset_px",
        ]

        tracking_columns = [
            "frame",
            "elapsed_seconds",
            "track_id",
            "observed",
            "predicted",
            "missed",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "person_confidence",
        ]

        with pose_csv_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as pose_file, tracking_csv_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as tracking_file:

            pose_writer = csv.DictWriter(
                pose_file,
                fieldnames=pose_columns,
            )

            tracking_writer = csv.DictWriter(
                tracking_file,
                fieldnames=tracking_columns,
            )

            pose_writer.writeheader()
            tracking_writer.writeheader()

            # ------------------------------------------------
            # LIVE LOOP
            # ------------------------------------------------

            while True:
                frame_start = time.perf_counter()

                with lock:
                    item = state["latest"]

                if item is None:
                    time.sleep(0.005)
                    continue

                sequence, raw, width, height = item

                if sequence == last_sequence:
                    if (
                        not args.no_display
                        and cv2.waitKey(1) & 0xFF == ord("q")
                    ):
                        break

                    time.sleep(0.005)
                    continue

                last_sequence = sequence

                if (
                    args.max_frames
                    and processed_frames >= args.max_frames
                ):
                    break

                if (
                    args.duration
                    and time.monotonic() - started_at >= args.duration
                ):
                    break

                # --------------------------------------------
                # FRAME CONVERSION
                # --------------------------------------------

                yuv = np.frombuffer(
                    raw,
                    dtype=np.uint8,
                ).reshape(
                    (height * 3 // 2, width)
                )

                frame = cv2.cvtColor(
                    yuv,
                    cv2.COLOR_YUV2BGR_I420,
                )

                if raw_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                    raw_writer = cv2.VideoWriter(
                        str(raw_video_path),
                        fourcc,
                        args.video_fps,
                        (width, height),
                    )

                    annotated_writer = cv2.VideoWriter(
                        str(annotated_video_path),
                        fourcc,
                        args.video_fps,
                        (width, height),
                    )

                    if (
                        not raw_writer.isOpened()
                        or not annotated_writer.isOpened()
                    ):
                        raise RuntimeError(
                            "Could not create benchmark videos"
                        )

                raw_writer.write(frame)

                # --------------------------------------------
                # YOLO INFERENCE
                # --------------------------------------------

                inference_start = time.perf_counter()

                results = model.predict(
                    source=frame,
                    device=args.device,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    classes=[0],
                    verbose=False,
                )

                torch.cuda.synchronize(device_index)

                inference_ms = (
                    time.perf_counter() - inference_start
                ) * 1000.0

                inference_times.append(inference_ms)

                result = results[0]

                if (
                    result.boxes is not None
                    and result.keypoints is not None
                    and result.keypoints.data is not None
                    and len(result.boxes) > 0
                ):
                    boxes = (
                        result.boxes.xyxy
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )

                    scores = (
                        result.boxes.conf
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )

                    yolo_poses = (
                        result.keypoints.data
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )

                    if yolo_poses.ndim != 3:
                        raise RuntimeError(
                            "YOLO keypoints must be 3D"
                        )

                    if yolo_poses.shape[1] < KEYPOINT_COUNT:
                        raise RuntimeError(
                            "YOLO returned fewer than 9 keypoints"
                        )

                    if yolo_poses.shape[2] != 3:
                        raise RuntimeError(
                            "Expected keypoint x, y, confidence"
                        )

                    if not (
                        len(boxes)
                        == len(scores)
                        == len(yolo_poses)
                    ):
                        raise RuntimeError(
                            "YOLO output count mismatch"
                        )

                    # Only COCO indices 0 through 8 enter
                    # the Company Vision pipeline.
                    poses = yolo_poses[
                        :,
                        :KEYPOINT_COUNT,
                        :,
                    ].copy()

                    if poses.shape != (
                        len(boxes),
                        KEYPOINT_COUNT,
                        3,
                    ):
                        raise RuntimeError(
                            "9-keypoint extraction failed"
                        )

                else:
                    boxes = np.empty(
                        (0, 4),
                        dtype=np.float32,
                    )

                    scores = np.empty(
                        (0,),
                        dtype=np.float32,
                    )

                    poses = np.empty(
                        (0, KEYPOINT_COUNT, 3),
                        dtype=np.float32,
                    )

                # --------------------------------------------
                # TRACKING AND DYNAMIC FILTERING
                # --------------------------------------------

                manager_start = time.perf_counter()

                tracked_persons = pose_manager.update(
                    boxes,
                    poses,
                    scores,
                )

                manager_ms = (
                    time.perf_counter() - manager_start
                ) * 1000.0

                manager_times.append(manager_ms)

                observed_persons = [
                    person
                    for person in tracked_persons
                    if person["observed"]
                ]

                frames_with_people += int(
                    bool(observed_persons)
                )

                total_detections += len(boxes)

                annotated = frame.copy()

                elapsed = time.monotonic() - started_at
                frame_number = processed_frames + 1

                # --------------------------------------------
                # TRACKING AND KEYPOINT METRICS
                # --------------------------------------------

                for person in tracked_persons:
                    track_id = int(person["track_id"])

                    bbox = np.asarray(
                        person["bbox"],
                        dtype=np.float32,
                    )

                    update_tracking_metrics(
                        tracking_metrics,
                        person,
                        frame_number,
                    )

                    tracking_writer.writerow({
                        "frame": frame_number,
                        "elapsed_seconds": round(elapsed, 4),
                        "track_id": track_id,
                        "observed": person["observed"],
                        "predicted": person["predicted"],
                        "missed": person["missed"],
                        "bbox_x1": round(float(bbox[0]), 2),
                        "bbox_y1": round(float(bbox[1]), 2),
                        "bbox_x2": round(float(bbox[2]), 2),
                        "bbox_y2": round(float(bbox[3]), 2),
                        "person_confidence": round(
                            float(person["confidence"]),
                            5,
                        ),
                    })

                    draw_person(
                        annotated,
                        person,
                        args.keypoint_conf,
                    )

                    if not person["observed"]:
                        continue

                    person_confidences.append(
                        float(person["confidence"])
                    )

                    filtered = np.asarray(
                        person["keypoints"],
                        dtype=np.float32,
                    )

                    raw_pose = np.asarray(
                        person["raw_keypoints"],
                        dtype=np.float32,
                    )

                    weights = np.asarray(
                        person["keypoint_weights"],
                        dtype=np.float32,
                    )

                    if filtered.shape != (9, 3):
                        raise RuntimeError(
                            "Filtered pose is not (9, 3)"
                        )

                    if raw_pose.shape != (9, 3):
                        raise RuntimeError(
                            "Raw pose is not (9, 3)"
                        )

                    if weights.shape != (9,):
                        raise RuntimeError(
                            "Joint weights are not (9,)"
                        )

                    if (
                        not np.isfinite(filtered).all()
                        or not np.isfinite(raw_pose).all()
                        or not np.isfinite(weights).all()
                    ):
                        raise RuntimeError(
                            "Non-finite pose or weights"
                        )

                    if np.any(
                        (weights < 0)
                        | (weights > 1)
                    ):
                        raise RuntimeError(
                            "Dynamic weights outside [0, 1]"
                        )

                    scale = body_scale(bbox)

                    for index, name in KEYPOINTS.items():
                        rx = float(raw_pose[index, 0])
                        ry = float(raw_pose[index, 1])
                        rc = float(raw_pose[index, 2])

                        fx = float(filtered[index, 0])
                        fy = float(filtered[index, 1])
                        fc = float(filtered[index, 2])

                        weight = float(weights[index])

                        raw_metric = update_point_metric(
                            raw_metrics,
                            previous_raw,
                            track_id,
                            index,
                            rx,
                            ry,
                            rc,
                            scale,
                            frame_number,
                            args.keypoint_conf,
                        )

                        filtered_metric = update_point_metric(
                            filtered_metrics,
                            previous_filtered,
                            track_id,
                            index,
                            fx,
                            fy,
                            fc,
                            scale,
                            frame_number,
                            args.keypoint_conf,
                        )

                        offset = math.hypot(
                            fx - rx,
                            fy - ry,
                        )

                        if (
                            rc >= args.keypoint_conf
                            and fc >= args.keypoint_conf
                        ):
                            filter_offsets[index].append(
                                offset
                            )

                        dynamic_weights[index].append(
                            weight
                        )

                        pose_writer.writerow({
                            "frame": frame_number,
                            "elapsed_seconds": round(elapsed, 4),
                            "track_id": track_id,
                            "keypoint_index": index,
                            "keypoint_name": name,
                            "raw_x": round(rx, 3),
                            "raw_y": round(ry, 3),
                            "raw_confidence": round(rc, 5),
                            "filtered_x": round(fx, 3),
                            "filtered_y": round(fy, 3),
                            "filtered_confidence": round(fc, 5),
                            "raw_valid": raw_metric["valid"],
                            "filtered_valid": (
                                filtered_metric["valid"]
                            ),
                            "raw_displacement_px": (
                                raw_metric["displacement"]
                            ),
                            "filtered_displacement_px": (
                                filtered_metric["displacement"]
                            ),
                            "raw_normalized_displacement": (
                                raw_metric[
                                    "normalized_displacement"
                                ]
                            ),
                            "filtered_normalized_displacement": (
                                filtered_metric[
                                    "normalized_displacement"
                                ]
                            ),
                            "raw_acceleration_px": (
                                raw_metric["acceleration"]
                            ),
                            "filtered_acceleration_px": (
                                filtered_metric["acceleration"]
                            ),
                            "dynamic_joint_weight": round(
                                weight,
                                5,
                            ),
                            "filter_offset_px": round(
                                offset,
                                3,
                            ),
                        })

                # --------------------------------------------
                # PERFORMANCE
                # --------------------------------------------

                processed_frames += 1

                frame_ms = (
                    time.perf_counter() - frame_start
                ) * 1000.0

                frame_times.append(frame_ms)

                total_elapsed = max(
                    time.monotonic() - started_at,
                    0.001,
                )

                fps = processed_frames / total_elapsed

                cv2.putText(
                    annotated,
                    "COMPANY VISION | 9 KEYPOINT MODE",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    annotated,
                    (
                        f"PERSONS:{len(observed_persons)} "
                        f"FPS:{fps:.2f} "
                        f"YOLO:{inference_ms:.1f}ms "
                        f"MANAGER:{manager_ms:.1f}ms "
                        f"FRAME:{frame_ms:.1f}ms"
                    ),
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 255, 255),
                    2,
                )

                annotated_writer.write(annotated)

                if not args.no_display:
                    cv2.imshow(
                        window_name,
                        annotated,
                    )

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                now = time.monotonic()

                if now - last_report >= 5:
                    with lock:
                        decoded = state["decoded"]
                        packets = state["packets"]
                        errors = state["errors"]

                    print(
                        f"FPS={fps:.2f} "
                        f"DECODED={decoded} "
                        f"PROCESSED={processed_frames} "
                        f"PACKETS={packets} "
                        f"PERSONS={len(observed_persons)} "
                        f"YOLO={inference_ms:.1f}ms "
                        f"MANAGER={manager_ms:.1f}ms "
                        f"FRAME={frame_ms:.1f}ms "
                        f"ERRORS={errors}"
                    )

                    pose_file.flush()
                    tracking_file.flush()

                    last_report = now

    except KeyboardInterrupt:
        print("\nStopped by user")

    finally:
        print("\nStopping benchmark...")

        if not args.no_display:
            cv2.destroyAllWindows()

        if raw_writer is not None:
            raw_writer.release()

        if annotated_writer is not None:
            annotated_writer.release()

        # ----------------------------------------------------
        # SDK CLEANUP
        # ----------------------------------------------------

        if play_id:
            try:
                sdk.StopRealPlayEx(play_id)
            except Exception as exc:  # noqa: BLE001
                print("WARNING: StopRealPlayEx:", exc)

        if free_port is not None:
            try:
                sdk.SetDecCallBack(
                    free_port,
                    None,
                )
            except Exception as exc:  # noqa: BLE001
                print("WARNING: Decoder callback cleanup:", exc)

            if player_started:
                try:
                    sdk.Stop(free_port)
                except Exception as exc:  # noqa: BLE001
                    print("WARNING: Decoder stop:", exc)

            if stream_open:
                try:
                    sdk.CloseStream(free_port)
                except Exception as exc:  # noqa: BLE001
                    print("WARNING: CloseStream:", exc)

            try:
                sdk.ReleasePort(free_port)
            except Exception as exc:  # noqa: BLE001
                print("WARNING: ReleasePort:", exc)

        if login_id:
            try:
                sdk.Logout(login_id)
            except Exception as exc:  # noqa: BLE001
                print("WARNING: Logout:", exc)

        if initialized:
            try:
                sdk.Cleanup()
            except Exception as exc:  # noqa: BLE001
                print("WARNING: SDK Cleanup:", exc)

        # ----------------------------------------------------
        # FINAL REPORT
        # ----------------------------------------------------

        elapsed = (
            time.monotonic() - started_at
            if started_at is not None
            else 0.0
        )

        fps = processed_frames / max(elapsed, 0.001)

        report = {
            "project": "Company Vision",
            "model": args.model,

            "keypoint_contract": {
                "source_model_keypoints": 17,
                "processed_keypoints": KEYPOINT_COUNT,
                "indices": list(range(KEYPOINT_COUNT)),
                "names": list(KEYPOINT_NAMES),
                "manager_input_shape": "(N, 9, 3)",
                "manager_weight_shape": "(N, 9)",
            },

            "camera": {
                "ip": args.ip,
                "port": args.port,
                "channel": args.channel,
                "transport": "Dahua NetSDK RAW_DATA",
            },

            "configuration": {
                "imgsz": args.imgsz,
                "person_confidence": args.conf,
                "keypoint_confidence": args.keypoint_conf,
                "confirm_frames": args.confirm_frames,
                "ema_alpha": args.ema_alpha,
                "max_missed": args.max_missed,
                "device": args.device,
            },

            "performance": {
                "processed_frames": processed_frames,
                "elapsed_seconds": round(elapsed, 3),
                "average_fps": round(fps, 3),
                "mean_inference_ms": safe_mean(
                    inference_times
                ),
                "p95_inference_ms": safe_percentile(
                    inference_times,
                    95,
                ),
                "mean_manager_ms": safe_mean(
                    manager_times
                ),
                "p95_manager_ms": safe_percentile(
                    manager_times,
                    95,
                ),
                "mean_frame_ms": safe_mean(
                    frame_times
                ),
                "p95_frame_ms": safe_percentile(
                    frame_times,
                    95,
                ),
            },

            "camera_stream": {
                "decoded_frames": state["decoded"],
                "received_packets": state["packets"],
                "callback_errors": state["errors"],
                "processed_frames": processed_frames,
                "skipped_decoded_frames": max(
                    0,
                    state["decoded"] - processed_frames,
                ),
            },

            "detections": {
                "raw_detections": total_detections,
                "frames_with_people": frames_with_people,
                "mean_person_confidence": safe_mean(
                    person_confidences
                ),
            },

            "keypoints": {},
            "raw_vs_filtered": {},
            "dynamic_joint_weighting": {},
            "tracking": {
                "total_tracks": len(tracking_metrics),
                "tracks": {},
            },

            "interpretation": [
                "Exactly 9 keypoints are processed.",
                "YOLO output indices 0 through 8 are used.",
                "Indices 9 through 16 are excluded from Manager.",
                "Dynamic weights are trust scores, not probabilities.",
                "Predicted-only frames are excluded from pose metrics.",
                "Displacement includes real body motion and detector jitter.",
                "Acceleration is a temporal jitter proxy.",
                "Filter offset is the spatial difference from raw YOLO output.",
                "Benchmark uses latest-frame sampling.",
            ],
        }

        # ----------------------------------------------------
        # KEYPOINT REPORT
        # ----------------------------------------------------

        for index, name in KEYPOINTS.items():
            raw_summary = summarize_keypoint(
                raw_metrics[index]
            )

            filtered_summary = summarize_keypoint(
                filtered_metrics[index]
            )

            raw_acceleration = safe_mean(
                raw_metrics[index]["normalized_acceleration"]
            )

            filtered_acceleration = safe_mean(
                filtered_metrics[index]["normalized_acceleration"]
            )

            reduction = None

            if (
                raw_acceleration is not None
                and raw_acceleration > 0
                and filtered_acceleration is not None
            ):
                reduction = round(
                    100.0 * (
                        raw_acceleration - filtered_acceleration
                    ) / raw_acceleration,
                    3,
                )

            weights = dynamic_weights[index]

            report["keypoints"][name] = {
                "raw": raw_summary,
                "filtered": filtered_summary,
            }

            report["raw_vs_filtered"][name] = {
                "mean_filter_offset_px": safe_mean(
                    filter_offsets[index]
                ),
                "p95_filter_offset_px": safe_percentile(
                    filter_offsets[index],
                    95,
                ),
                "raw_normalized_acceleration": raw_acceleration,
                "filtered_normalized_acceleration": (
                    filtered_acceleration
                ),
                "acceleration_reduction_percent": reduction,
            }

            report["dynamic_joint_weighting"][name] = {
                "mean_weight": safe_mean(weights),
                "std_weight": safe_std(weights),
                "p05_weight": safe_percentile(weights, 5),
                "p50_weight": safe_percentile(weights, 50),
                "p95_weight": safe_percentile(weights, 95),
                "samples": len(weights),
            }

        # ----------------------------------------------------
        # TRACKING REPORT
        # ----------------------------------------------------

        for track_id, values in tracking_metrics.items():
            first = values["first_frame"]
            last = values["last_frame"]

            lifespan = (
                last - first + 1
                if first is not None and last is not None
                else 0
            )

            report["tracking"]["tracks"][str(track_id)] = {
                "first_frame": first,
                "last_frame": last,
                "lifespan_frames": lifespan,
                "observed_frames": values["observed"],
                "predicted_frames": values["predicted"],
                "max_consecutive_observed": (
                    values["max_consecutive_observed"]
                ),
                "max_consecutive_predicted": (
                    values["max_consecutive_predicted"]
                ),
                "observed_ratio_percent": (
                    round(
                        100.0 * values["observed"] / lifespan,
                        3,
                    )
                    if lifespan else None
                ),
            }

        # ----------------------------------------------------
        # SAVE REPORT AND PLOTS
        # ----------------------------------------------------

        report_path.write_text(
            json.dumps(
                report,
                indent=4,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        create_plots(
            raw_metrics,
            filtered_metrics,
            dynamic_weights,
            filter_offsets,
            output_dir,
        )

        # ----------------------------------------------------
        # CONSOLE REPORT
        # ----------------------------------------------------

        print("\n" + "=" * 65)
        print("9-KEYPOINT DYNAMIC WEIGHTING REPORT")
        print("=" * 65)

        print("PROCESSED FRAMES:", processed_frames)
        print("DECODED FRAMES:", state["decoded"])
        print("RAW DETECTIONS:", total_detections)
        print("UNIQUE TRACKS:", len(tracking_metrics))
        print("AVERAGE FPS:", round(fps, 2))

        print("\nDYNAMIC JOINT WEIGHTS")

        for name, values in (
            report["dynamic_joint_weighting"].items()
        ):
            print(
                f"{name:16s} "
                f"MEAN={values['mean_weight']} "
                f"P05={values['p05_weight']} "
                f"P95={values['p95_weight']} "
                f"SAMPLES={values['samples']}"
            )

        print("\nRAW VS FILTERED")

        for name, values in report["raw_vs_filtered"].items():
            print(
                f"{name:16s} "
                f"REDUCTION="
                f"{values['acceleration_reduction_percent']}% "
                f"OFFSET="
                f"{values['mean_filter_offset_px']}px"
            )

        print("\nOUTPUT DIRECTORY:", output_dir)
        print("RAW VIDEO:", raw_video_path)
        print("ANNOTATED VIDEO:", annotated_video_path)
        print("POSE CSV:", pose_csv_path)
        print("TRACKING CSV:", tracking_csv_path)
        print("JSON:", report_path)

        print("BENCHMARK FINISHED")


if __name__ == "__main__":
    main()
