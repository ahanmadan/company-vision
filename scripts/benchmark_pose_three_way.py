
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

# ============================================================
# CONTRACT: EXACTLY 9 KEYPOINTS
# ============================================================

assert KEYPOINT_COUNT == 9
assert len(KEYPOINT_NAMES) == 9

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

MODES = ("raw", "ema", "dynamic")


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Company Vision Three-Way Pose Benchmark"
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
    parser.add_argument("--keypoint-conf", type=float, default=0.25)

    parser.add_argument("--confirm-frames", type=int, default=3)
    parser.add_argument("--ema-alpha", type=float, default=0.40)
    parser.add_argument("--max-missed", type=int, default=60)

    parser.add_argument("--duration", type=float, default=120)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--video-fps", type=float, default=20.0)
    parser.add_argument("--device", default="0")

    parser.add_argument(
        "--output",
        default="runs/pose_three_way",
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
    )

    args = parser.parse_args()

    if args.channel < 1:
        parser.error("Invalid channel")

    if not 1 <= args.port <= 65535:
        parser.error("Invalid port")

    if args.imgsz < 32:
        parser.error("Invalid imgsz")

    if not 0 < args.conf <= 1:
        parser.error("Invalid confidence")

    if not 0 < args.keypoint_conf <= 1:
        parser.error("Invalid keypoint confidence")

    if not 0 < args.ema_alpha <= 1:
        parser.error("Invalid EMA alpha")

    if args.confirm_frames < 1 or args.max_missed < 0:
        parser.error("Invalid tracking parameters")

    if args.duration < 0 or args.max_frames < 0:
        parser.error("Invalid duration or frame limit")

    if args.video_fps <= 0:
        parser.error("Invalid video FPS")

    return args


# ============================================================
# STATISTICS HELPERS
# ============================================================

def mean(values):
    return float(np.mean(values)) if values else None


def percentile(values, value):
    return float(np.percentile(values, value)) if values else None


def body_scale(box):
    width = max(1.0, float(box[2] - box[0]))
    height = max(1.0, float(box[3] - box[1]))
    return math.hypot(width, height)


def empty_poses():
    return np.empty(
        (0, KEYPOINT_COUNT, 3),
        dtype=np.float32,
    )


def create_metric_storage():
    return {
        mode: {
            index: {
                "confidence": [],
                "displacement": [],
                "normalized_displacement": [],
                "acceleration": [],
                "normalized_acceleration": [],
                "valid": 0,
                "missing": 0,
                "total": 0,
            }
            for index in range(KEYPOINT_COUNT)
        }
        for mode in MODES
    }


def create_previous_storage():
    return {
        mode: {}
        for mode in MODES
    }


# ============================================================
# INDEPENDENT FIXED-ALPHA EMA
# ============================================================

class IndependentEMA:
    """
    Fixed-alpha EMA.

    This filter receives raw YOLO keypoints and does not
    use dynamic joint weights.

    Input shape: (9, 3)
    Output shape: (9, 3)
    """

    def __init__(self, alpha, confidence_threshold):
        self.alpha = alpha
        self.confidence_threshold = confidence_threshold
        self.states = {}

    def update(self, track_id, raw_points):
        raw_points = np.asarray(
            raw_points,
            dtype=np.float32,
        )

        if raw_points.shape != (KEYPOINT_COUNT, 3):
            raise ValueError("EMA input must have shape (9, 3)")

        key = int(track_id)

        if key not in self.states:
            self.states[key] = np.zeros(
                (KEYPOINT_COUNT, 3),
                dtype=np.float32,
            )

        state = self.states[key]
        output = raw_points.copy()

        for index in range(KEYPOINT_COUNT):
            x, y, confidence = raw_points[index]

            if confidence < self.confidence_threshold:
                state[index] = 0.0
                output[index] = raw_points[index]
                continue

            if state[index, 2] < self.confidence_threshold:
                state[index] = raw_points[index]
            else:
                state[index, 0] = (
                    self.alpha * x
                    + (1.0 - self.alpha) * state[index, 0]
                )

                state[index, 1] = (
                    self.alpha * y
                    + (1.0 - self.alpha) * state[index, 1]
                )

                state[index, 2] = confidence

            output[index] = state[index]

        return output

    def remove_track(self, track_id):
        self.states.pop(int(track_id), None)


# ============================================================
# KEYPOINT METRICS
# ============================================================

def update_metrics(
    mode,
    track_id,
    points,
    scale,
    frame_number,
    storage,
    previous,
    threshold,
):
    if points.shape != (KEYPOINT_COUNT, 3):
        raise ValueError(
            f"{mode} pose must have shape (9, 3)"
        )

    for index in range(KEYPOINT_COUNT):
        x, y, confidence = map(
            float,
            points[index],
        )

        values = storage[mode][index]
        values["total"] += 1

        key = (int(track_id), index)

        if confidence < threshold:
            values["missing"] += 1
            previous[mode].pop(key, None)
            continue

        values["valid"] += 1
        values["confidence"].append(confidence)

        old = previous[mode].get(key)

        velocity_x = 0.0
        velocity_y = 0.0

        displacement = None
        normalized_displacement = None
        acceleration = None
        normalized_acceleration = None

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

                normalized_displacement = (
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

                values["displacement"].append(displacement)

                values["normalized_displacement"].append(
                    normalized_displacement
                )

                values["acceleration"].append(acceleration)

                values["normalized_acceleration"].append(
                    normalized_acceleration
                )

        previous[mode][key] = (
            x,
            y,
            frame_number,
            scale,
            velocity_x,
            velocity_y,
        )

        yield {
            "keypoint_index": index,
            "keypoint_name": KEYPOINTS[index],
            "x": x,
            "y": y,
            "confidence": confidence,
            "displacement": displacement,
            "normalized_displacement": normalized_displacement,
            "acceleration": acceleration,
            "normalized_acceleration": normalized_acceleration,
        }


# ============================================================
# DRAWING
# ============================================================

def draw_pose(frame, points, threshold, weights=None):
    if points.shape != (KEYPOINT_COUNT, 3):
        raise ValueError("Invalid drawing pose dimensions")

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

    for index, center in valid.items():
        if weights is None:
            color = (255, 200, 0)
            label = str(index)
        else:
            weight = float(weights[index])

            if weight >= 0.65:
                color = (0, 220, 0)
            elif weight >= 0.40:
                color = (0, 200, 255)
            else:
                color = (0, 80, 255)

            label = f"{index}:{weight:.2f}"

        cv2.circle(
            frame,
            center,
            5,
            color,
            -1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            label,
            (center[0] + 5, center[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_person_box(frame, person, title):
    x1, y1, x2, y2 = np.round(
        person["bbox"]
    ).astype(int)

    color = (
        (0, 220, 0)
        if person["observed"]
        else (0, 165, 255)
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    cv2.putText(
        frame,
        f"{title} | ID:{person['track_id']}",
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
    )


# ============================================================
# REPORT
# ============================================================

def build_report(storage, tracking, performance, args):
    report = {
        "project": "Company Vision",
        "benchmark": "Three-Way Pose Comparison",
        "model": args.model,
        "keypoint_count": KEYPOINT_COUNT,
        "keypoint_names": list(KEYPOINT_NAMES),
        "configuration": {
            "imgsz": args.imgsz,
            "detection_confidence": args.conf,
            "keypoint_confidence": args.keypoint_conf,
            "ema_alpha": args.ema_alpha,
            "confirm_frames": args.confirm_frames,
            "max_missed": args.max_missed,
        },
        "methodology": {
            "raw": "Direct YOLO keypoints",
            "ema": "Fixed-alpha EMA applied to raw YOLO keypoints",
            "dynamic": (
                "StablePoseManager filtered keypoints "
                "and dynamic joint weights"
            ),
            "tracking": (
                "One shared StablePoseManager instance "
                "for all three methods"
            ),
            "input": (
                "Same frames, YOLO detections and track IDs"
            ),
            "important": (
                "Normalized displacement and acceleration "
                "are motion/jitter proxies, not ground-truth accuracy."
            ),
        },
        "performance": performance,
        "tracking": tracking,
        "modes": {},
        "comparison_vs_raw": {},
    }

    for mode in MODES:
        report["modes"][mode] = {}

        for index, name in enumerate(KEYPOINT_NAMES):
            values = storage[mode][index]
            total = values["total"]
            valid = values["valid"]

            report["modes"][mode][name] = {
                "total": total,
                "valid": valid,
                "missing": values["missing"],
                "valid_rate_percent": (
                    round(100 * valid / total, 3)
                    if total else None
                ),
                "mean_confidence": mean(values["confidence"]),
                "mean_displacement_px": mean(
                    values["displacement"]
                ),
                "p95_displacement_px": percentile(
                    values["displacement"],
                    95,
                ),
                "mean_normalized_displacement": mean(
                    values["normalized_displacement"]
                ),
                "mean_acceleration_px": mean(
                    values["acceleration"]
                ),
                "mean_normalized_acceleration": mean(
                    values["normalized_acceleration"]
                ),
                "p95_normalized_acceleration": percentile(
                    values["normalized_acceleration"],
                    95,
                ),
            }

    for name in KEYPOINT_NAMES:
        raw = report["modes"]["raw"][name][
            "mean_normalized_acceleration"
        ]

        report["comparison_vs_raw"][name] = {}

        for mode in ("ema", "dynamic"):
            current = report["modes"][mode][name][
                "mean_normalized_acceleration"
            ]

            reduction = None

            if raw is not None and raw > 0 and current is not None:
                reduction = round(
                    100.0 * (raw - current) / raw,
                    3,
                )

            report["comparison_vs_raw"][name][mode] = {
                "acceleration_reduction_percent": reduction,
            }

    return report


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
        raise RuntimeError("Invalid CUDA device")

    print("=" * 70)
    print("COMPANY VISION - THREE WAY POSE BENCHMARK")
    print("=" * 70)

    print("GPU:", torch.cuda.get_device_name(device_index))
    print("MODEL:", args.model)
    print("KEYPOINTS:", KEYPOINT_COUNT)
    print("EMA ALPHA:", args.ema_alpha)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(args.output) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "three_way_metrics.csv"
    json_path = output_dir / "three_way_report.json"
    video_path = output_dir / "three_way_comparison.mp4"

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print("\nLoading YOLO...")

    model = YOLO(args.model)

    if model.task != "pose":
        raise RuntimeError("Model must be a Pose model")

    print("MODEL READY")

    # --------------------------------------------------------
    # CAMERA CREDENTIALS
    # --------------------------------------------------------

    username = input("Camera Username: ").strip()
    password = getpass("Camera Password: ")

    if not username or not password:
        raise ValueError("Empty camera credentials")

    # --------------------------------------------------------
    # MANAGER AND EMA
    # --------------------------------------------------------

    manager = StablePoseManager(
        confirm_hits=args.confirm_frames,
        max_missed=args.max_missed,
        min_keypoints=3,
        keypoint_conf=args.keypoint_conf,
        ema_alpha=args.ema_alpha,
        match_distance=0.85,
    )

    ema_filter = IndependentEMA(
        alpha=args.ema_alpha,
        confidence_threshold=args.keypoint_conf,
    )

    storage = create_metric_storage()
    previous = create_previous_storage()

    tracking = defaultdict(
        lambda: {
            "observed": 0,
            "predicted": 0,
            "first_frame": None,
            "last_frame": None,
            "max_consecutive_predicted": 0,
            "current_consecutive_predicted": 0,
        }
    )

    inference_times = []
    manager_times = []
    ema_times = []
    frame_times = []

    person_confidences = []

    processed_frames = 0
    total_detections = 0

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
        "decoded": 0,
        "packets": 0,
        "errors": 0,
    }

    last_sequence = 0

    comparison_writer = None
    started_at = None

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

    window_name = "Company Vision Three-Way Benchmark"

    csv_columns = [
        "frame",
        "elapsed_seconds",
        "mode",
        "track_id",
        "keypoint_index",
        "keypoint_name",
        "x",
        "y",
        "confidence",
        "valid",
        "displacement_px",
        "normalized_displacement",
        "acceleration_px",
        "normalized_acceleration",
        "dynamic_weight",
    ]

    try:
        # ----------------------------------------------------
        # SDK INITIALIZATION
        # ----------------------------------------------------

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
            free_port,
            decoding_callback,
        ):
            raise RuntimeError("SetDecCallBack failed")

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

        print("CAMERA LOGIN: PASS")
        print("CAMERA STREAM: READY")

        deadline = time.monotonic() + 20

        while time.monotonic() < deadline:
            with lock:
                ready = state["latest"] is not None

            if ready:
                break

            time.sleep(0.05)

        else:
            raise RuntimeError("No decoded frames received")

        if not args.no_display:
            cv2.namedWindow(
                window_name,
                cv2.WINDOW_NORMAL,
            )

        started_at = time.monotonic()

        # ----------------------------------------------------
        # CSV AND LIVE LOOP
        # ----------------------------------------------------

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=csv_columns,
            )

            writer.writeheader()

            print("\nTHREE-WAY BENCHMARK STARTED")
            print("Press Q to stop.")

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

                    time.sleep(0.003)
                    continue

                last_sequence = sequence

                if (
                    args.max_frames
                    and processed_frames >= args.max_frames
                ):
                    break

                elapsed = time.monotonic() - started_at

                if args.duration and elapsed >= args.duration:
                    break

                frame = cv2.cvtColor(
                    np.frombuffer(
                        raw,
                        dtype=np.uint8,
                    ).reshape(
                        height * 3 // 2,
                        width,
                    ),
                    cv2.COLOR_YUV2BGR_I420,
                )

                # --------------------------------------------
                # ONE YOLO INFERENCE FOR ALL THREE METHODS
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

                    model_poses = (
                        result.keypoints.data
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )

                    if (
                        model_poses.ndim != 3
                        or model_poses.shape[1] < KEYPOINT_COUNT
                        or model_poses.shape[2] != 3
                    ):
                        raise RuntimeError(
                            "Invalid YOLO keypoint dimensions"
                        )

                    poses = model_poses[
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

                    if len(scores) != len(boxes):
                        raise RuntimeError(
                            "Detection count mismatch"
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

                    poses = empty_poses()

                total_detections += len(boxes)

                # --------------------------------------------
                # SHARED TRACKING AND DYNAMIC FILTER
                # --------------------------------------------

                manager_start = time.perf_counter()

                persons = manager.update(
                    boxes,
                    poses,
                    scores,
                )

                manager_ms = (
                    time.perf_counter() - manager_start
                ) * 1000.0

                manager_times.append(manager_ms)

                raw_view = frame.copy()
                ema_view = frame.copy()
                dynamic_view = frame.copy()

                frame_number = processed_frames + 1

                # IMPORTANT:
                # EMA time is measured across all people in
                # this frame, not separately averaged per person.
                ema_frame_ms = 0.0

                # --------------------------------------------
                # PROCESS EACH TRACK
                # --------------------------------------------

                for person in persons:
                    track_id = int(person["track_id"])

                    track = tracking[track_id]

                    if track["first_frame"] is None:
                        track["first_frame"] = frame_number

                    track["last_frame"] = frame_number

                    if person["observed"]:
                        track["observed"] += 1
                        track["current_consecutive_predicted"] = 0

                        raw_points = np.asarray(
                            person["raw_keypoints"],
                            dtype=np.float32,
                        )

                        dynamic_points = np.asarray(
                            person["keypoints"],
                            dtype=np.float32,
                        )

                        weights = np.asarray(
                            person["keypoint_weights"],
                            dtype=np.float32,
                        )

                        if raw_points.shape != (9, 3):
                            raise RuntimeError(
                                "raw_keypoints must be (9, 3)"
                            )

                        if dynamic_points.shape != (9, 3):
                            raise RuntimeError(
                                "keypoints must be (9, 3)"
                            )

                        if weights.shape != (9,):
                            raise RuntimeError(
                                "keypoint_weights must be (9,)"
                            )

                        if not (
                            np.isfinite(raw_points).all()
                            and np.isfinite(dynamic_points).all()
                            and np.isfinite(weights).all()
                        ):
                            raise RuntimeError(
                                "Non-finite keypoint values"
                            )

                        # MODE 1: RAW YOLO
                        raw_output = raw_points.copy()

                        # MODE 2: FIXED EMA
                        ema_start = time.perf_counter()

                        ema_output = ema_filter.update(
                            track_id,
                            raw_points,
                        )

                        ema_frame_ms += (
                            time.perf_counter() - ema_start
                        ) * 1000.0

                        # MODE 3: DYNAMIC WEIGHTING
                        dynamic_output = dynamic_points.copy()

                        scale = body_scale(person["bbox"])

                        outputs = {
                            "raw": raw_output,
                            "ema": ema_output,
                            "dynamic": dynamic_output,
                        }

                        # Same track ID, frame, scale and
                        # confidence threshold for all methods.
                        for mode, points in outputs.items():
                            rows = update_metrics(
                                mode,
                                track_id,
                                points,
                                scale,
                                frame_number,
                                storage,
                                previous,
                                args.keypoint_conf,
                            )

                            for metric in rows:
                                index = metric["keypoint_index"]

                                weight = (
                                    float(weights[index])
                                    if mode == "dynamic"
                                    else None
                                )

                                writer.writerow({
                                    "frame": frame_number,
                                    "elapsed_seconds": round(
                                        elapsed,
                                        4,
                                    ),
                                    "mode": mode,
                                    "track_id": track_id,
                                    "keypoint_index": index,
                                    "keypoint_name": (
                                        metric["keypoint_name"]
                                    ),
                                    "x": round(metric["x"], 3),
                                    "y": round(metric["y"], 3),
                                    "confidence": round(
                                        metric["confidence"],
                                        5,
                                    ),
                                    "valid": (
                                        metric["confidence"]
                                        >= args.keypoint_conf
                                    ),
                                    "displacement_px": (
                                        metric["displacement"]
                                    ),
                                    "normalized_displacement": (
                                        metric[
                                            "normalized_displacement"
                                        ]
                                    ),
                                    "acceleration_px": (
                                        metric["acceleration"]
                                    ),
                                    "normalized_acceleration": (
                                        metric[
                                            "normalized_acceleration"
                                        ]
                                    ),
                                    "dynamic_weight": weight,
                                })

                        person_confidences.append(
                            float(person["confidence"])
                        )

                        # DRAW SAME PERSON IN THREE PANELS
                        for view, title in (
                            (raw_view, "RAW"),
                            (ema_view, "EMA"),
                            (dynamic_view, "DYNAMIC"),
                        ):
                            draw_person_box(
                                view,
                                person,
                                title,
                            )

                        draw_pose(
                            raw_view,
                            raw_output,
                            args.keypoint_conf,
                        )

                        draw_pose(
                            ema_view,
                            ema_output,
                            args.keypoint_conf,
                        )

                        draw_pose(
                            dynamic_view,
                            dynamic_output,
                            args.keypoint_conf,
                            weights,
                        )

                    elif person["predicted"]:
                        track["predicted"] += 1

                        track["current_consecutive_predicted"] += 1

                        track["max_consecutive_predicted"] = max(
                            track["max_consecutive_predicted"],
                            track["current_consecutive_predicted"],
                        )

                        # Do not carry stale EMA coordinates
                        # across an unobserved tracking frame.
                        ema_filter.remove_track(track_id)

                        for view, title in (
                            (raw_view, "RAW"),
                            (ema_view, "EMA"),
                            (dynamic_view, "DYNAMIC"),
                        ):
                            draw_person_box(
                                view,
                                person,
                                f"{title} PREDICTED",
                            )

                # Record one EMA processing measurement per frame.
                ema_times.append(ema_frame_ms)

                # --------------------------------------------
                # THREE-PANEL VIDEO
                # --------------------------------------------

                panel_height = min(720, height)
                panel_width = int(
                    width * panel_height / height
                )

                panels = []

                for view, title in (
                    (raw_view, "1 - RAW YOLO"),
                    (ema_view, "2 - FIXED EMA"),
                    (dynamic_view, "3 - DYNAMIC WEIGHTING"),
                ):
                    resized = cv2.resize(
                        view,
                        (panel_width, panel_height),
                    )

                    cv2.putText(
                        resized,
                        title,
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 255),
                        2,
                    )

                    panels.append(resized)

                comparison = np.hstack(panels)

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
                    comparison,
                    (
                        f"FPS:{fps:.2f} "
                        f"YOLO:{inference_ms:.1f}ms "
                        f"MANAGER:{manager_ms:.1f}ms "
                        f"EMA:{ema_frame_ms:.2f}ms "
                        f"FRAME:{frame_ms:.1f}ms"
                    ),
                    (20, panel_height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

                if comparison_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                    comparison_writer = cv2.VideoWriter(
                        str(video_path),
                        fourcc,
                        args.video_fps,
                        (
                            comparison.shape[1],
                            comparison.shape[0],
                        ),
                    )

                    if not comparison_writer.isOpened():
                        raise RuntimeError(
                            "Cannot create comparison video"
                        )

                comparison_writer.write(comparison)

                if not args.no_display:
                    cv2.imshow(
                        window_name,
                        comparison,
                    )

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if processed_frames % 100 == 0:
                    csv_file.flush()

                    print(
                        f"FRAME={processed_frames} "
                        f"FPS={fps:.2f} "
                        f"YOLO={inference_ms:.1f}ms "
                        f"MANAGER={manager_ms:.1f}ms "
                        f"EMA={ema_frame_ms:.3f}ms "
                        f"TOTAL={frame_ms:.1f}ms"
                    )

    except KeyboardInterrupt:
        print("\nBenchmark stopped by user")

    finally:
        print("\nStopping camera and saving results...")

        if comparison_writer is not None:
            comparison_writer.release()

        if not args.no_display:
            cv2.destroyAllWindows()

        # ----------------------------------------------------
        # SDK CLEANUP
        # ----------------------------------------------------

        if play_id:
            try:
                sdk.StopRealPlayEx(play_id)
            except Exception as exc:  # noqa: BLE001
                print("StopRealPlayEx:", exc)

        if free_port is not None:
            try:
                sdk.SetDecCallBack(free_port, None)
            except Exception as exc:  # noqa: BLE001
                print("SetDecCallBack:", exc)

            if player_started:
                try:
                    sdk.Stop(free_port)
                except Exception as exc:  # noqa: BLE001
                    print("Decoder stop:", exc)

            if stream_open:
                try:
                    sdk.CloseStream(free_port)
                except Exception as exc:  # noqa: BLE001
                    print("CloseStream:", exc)

            try:
                sdk.ReleasePort(free_port)
            except Exception as exc:  # noqa: BLE001
                print("ReleasePort:", exc)

        if login_id:
            try:
                sdk.Logout(login_id)
            except Exception as exc:  # noqa: BLE001
                print("Logout:", exc)

        if initialized:
            try:
                sdk.Cleanup()
            except Exception as exc:  # noqa: BLE001
                print("SDK Cleanup:", exc)

        # ----------------------------------------------------
        # FINAL REPORT
        # ----------------------------------------------------

        elapsed = (
            time.monotonic() - started_at
            if started_at is not None
            else 0.0
        )

        with lock:
            camera_state = dict(state)

        tracking_report = {}

        for track_id, values in tracking.items():
            first = values["first_frame"]
            last = values["last_frame"]

            lifespan = (
                last - first + 1
                if first is not None and last is not None
                else 0
            )

            tracking_report[str(track_id)] = {
                "first_frame": first,
                "last_frame": last,
                "lifespan_frames": lifespan,
                "observed_frames": values["observed"],
                "predicted_frames": values["predicted"],
                "max_consecutive_predicted": (
                    values["max_consecutive_predicted"]
                ),
            }

        performance = {
            "processed_frames": processed_frames,
            "elapsed_seconds": round(elapsed, 3),
            "average_fps": round(
                processed_frames / max(elapsed, 0.001),
                3,
            ),
            "mean_yolo_ms": mean(inference_times),
            "p95_yolo_ms": percentile(inference_times, 95),
            "mean_manager_ms": mean(manager_times),
            "p95_manager_ms": percentile(manager_times, 95),
            "mean_ema_ms_per_frame": mean(ema_times),
            "p95_ema_ms_per_frame": percentile(ema_times, 95),
            "mean_total_frame_ms": mean(frame_times),
            "p95_total_frame_ms": percentile(frame_times, 95),
            "decoded_frames": camera_state["decoded"],
            "received_packets": camera_state["packets"],
            "callback_errors": camera_state["errors"],
            "skipped_decoded_frames": max(
                0,
                camera_state["decoded"] - processed_frames,
            ),
        }

        report = build_report(
            storage,
            tracking_report,
            performance,
            args,
        )

        report["detections"] = {
            "total_yolo_detections": total_detections,
            "mean_person_confidence": mean(person_confidences),
        }

        report_path = output_dir / "three_way_report.json"

        report_path.write_text(
            json.dumps(
                report,
                indent=4,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        # ----------------------------------------------------
        # CONSOLE SUMMARY
        # ----------------------------------------------------

        print("\n" + "=" * 70)
        print("THREE-WAY BENCHMARK RESULTS")
        print("=" * 70)

        print("PROCESSED FRAMES:", processed_frames)
        print("AVERAGE FPS:", performance["average_fps"])
        print("YOLO MEAN:", performance["mean_yolo_ms"])
        print("MANAGER MEAN:", performance["mean_manager_ms"])
        print("EMA MEAN PER FRAME:", performance["mean_ema_ms_per_frame"])

        print("\nMEAN NORMALIZED ACCELERATION")

        for mode in MODES:
            values = [
                report["modes"][mode][name][
                    "mean_normalized_acceleration"
                ]
                for name in KEYPOINT_NAMES
            ]

            valid_values = [
                value for value in values
                if value is not None
            ]

            print(
                f"{mode.upper():10s}:",
                mean(valid_values),
            )

        print("\nREDUCTION COMPARED WITH RAW")

        for mode in ("ema", "dynamic"):
            values = [
                report["comparison_vs_raw"][name][mode][
                    "acceleration_reduction_percent"
                ]
                for name in KEYPOINT_NAMES
            ]

            valid_values = [
                value for value in values
                if value is not None
            ]

            print(
                f"{mode.upper():10s}:",
                mean(valid_values),
                "%",
            )

        print("\nOUTPUT DIRECTORY:", output_dir)
        print("CSV:", csv_path)
        print("JSON:", json_path)
        print("VIDEO:", video_path)

        print("\nBENCHMARK FINISHED")


if __name__ == "__main__":
    main()
