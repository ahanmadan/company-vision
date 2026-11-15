
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



def setup_workstation_rois(frame):
    from pathlib import Path
    config = (
        Path(__file__).resolve().parents[1]
        / "reports/node_5_10_3/multitrack_v2"
        / "workstations.json"
    )

    if config.is_file():
        areas = json.loads(config.read_text(encoding="utf-8"))
        if len(areas) != 3:
            raise RuntimeError("ROI configuration requires 3 areas.")
        return areas

    print("SELECT THREE WORKSTATIONS")
    print("Draw a rectangle for each workstation.")
    print("Press ENTER after each ROI, ESC to finish.")

    rectangles = cv2.selectROIs(
        "SELECT 3 WORKSTATIONS",
        frame,
        showCrosshair=True,
        fromCenter=False,
    )

    cv2.destroyWindow("SELECT 3 WORKSTATIONS")

    if len(rectangles) != 3:
        raise RuntimeError("Exactly three ROI regions required.")

    areas = {}
    for index, rect in enumerate(rectangles, 1):
        x, y, w, h = map(int, rect)
        if w <= 0 or h <= 0:
            raise RuntimeError("Invalid ROI.")
        areas[str(index)] = [x, y, x + w, y + h]

    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        json.dumps(areas, indent=2),
        encoding="utf-8",
    )
    print("ROI configuration saved:", config)
    return areas


def workstation_for_box(box, areas):
    b = np.asarray(box, dtype=float)
    center = (b[:2] + b[2:]) / 2.0
    matches = []

    for key, area in areas.items():
        x1, y1, x2, y2 = area
        if x1 <= center[0] <= x2 and y1 <= center[1] <= y2:
            matches.append(key)

    return matches[0] if len(matches) == 1 else None



class AdaptiveHeadStillness:
    """
    Preserve existing full-pose stillness where dependable.
    Fallback to head-motion evidence for partially occluded tracks.
    Head-only seconds are NOT full-body stillness evidence.
    """

    def __init__(self):
        self.full = WeightedStillness()
        self.head_states = {}
        self.first_head_states = {}
        self.first_track_id = None
        self.max_hidden = 4.0
        self.min_head_conf = 0.45
        self.min_body_conf = 0.42
        self.head_motion_fraction = 0.013
        self.confirm_motion = 2

    def _analyze(self, person):
        pts = np.asarray(
            person.get("raw_keypoints", person.get("keypoints")),
            dtype=np.float64,
        )
        if pts.shape != (9, 3):
            pts = np.asarray(
                person.get("keypoints"), dtype=np.float64
            )

        box = np.asarray(
            person.get("bbox"), dtype=np.float64
        )

        if (
            pts.shape != (9, 3)
            or box.shape != (4,)
            or not np.isfinite(pts).all()
            or not np.isfinite(box).all()
        ):
            return None

        diagonal = max(
            float(np.linalg.norm(box[2:] - box[:2])),
            1.0,
        )

        head_valid = pts[:5, 2] >= self.min_head_conf
        shoulders = pts[5:7, 2] >= self.min_body_conf
        elbows = pts[7:9, 2] >= self.min_body_conf

        if np.count_nonzero(head_valid) < 2:
            return {
                "valid_head": False,
                "full": False,
            }

        weights = np.array(
            [1.5, 1.1, 1.1, 0.7, 0.7],
            dtype=np.float64,
        )
        weights *= np.clip(pts[:5, 2], 0.0, 1.0)
        weights[~head_valid] = 0.0

        center = np.average(
            pts[:5, :2],
            weights=weights,
            axis=0,
        )

        full_body = (
            np.count_nonzero(shoulders) == 2
            and np.count_nonzero(elbows) >= 1
        )

        return {
            "valid_head": True,
            "center": center,
            "diag": diagonal,
            "full": bool(full_body),
            "bbox": box.copy(),
            "head_visible": int(
                np.count_nonzero(head_valid)
            ),
        }

    def update(self, tracks, now):
        now = float(now)
        output = {}
        full_tracks = []
        head_tracks = []
        seen = set()

        for person in tracks:
            tid = int(person["track_id"])
            if tid in seen:
                self.head_states.pop(tid, None)
                output[tid] = {
                    "state": "UNKNOWN",
                    "seconds": 0.0,
                    "reason": "duplicate_id",
                }
                continue

            seen.add(tid)

            if (
                person.get("observed") is not True
                or person.get("behavior_eligible") is not True
            ):
                output[tid] = {
                    "state": "UNKNOWN",
                    "seconds": 0.0,
                    "reason": "occluded_or_ineligible",
                }
                continue

            info = self._analyze(person)

            if info is None or not info["valid_head"]:
                output[tid] = {
                    "state": "UNKNOWN",
                    "seconds": 0.0,
                    "reason": "head_unavailable",
                }
                continue

            if info["full"]:
                full_tracks.append(person)
            else:
                head_tracks.append((person, info))

        # Existing behavior for fully visible people.
        full_output = self.full.update(full_tracks, now)
        output.update(full_output)

        # Select only one clearly visible person.
        # Once selected, NEVER silently switch to another ID.
        if self.first_track_id is None:
            candidates = []

            for person in full_tracks:
                pts = np.asarray(
                    person.get(
                        "raw_keypoints",
                        person.get("keypoints")
                    ),
                    dtype=np.float64,
                )

                if pts.shape != (9, 3):
                    continue

                body_conf = pts[5:9, 2]

                if np.count_nonzero(body_conf >= 0.50) != 4:
                    continue

                candidates.append((
                    float(np.mean(body_conf)),
                    int(person["track_id"]),
                ))

            if candidates:
                candidates.sort(reverse=True)
                self.first_track_id = candidates[0][1]
                print(
                    "FIRST PERSON HEAD TARGET ID:",
                    self.first_track_id,
                )

        for person in full_tracks:
            tid = int(person["track_id"])

            if tid != self.first_track_id:
                continue

            info = self._analyze(person)

            if info is None or not info["valid_head"]:
                self.first_head_states.pop(tid, None)
                continue

            center = info["center"]
            diag = info["diag"]
            state = self.first_head_states.get(tid)

            if state is None:
                self.first_head_states[tid] = {
                    "anchor": center.copy(),
                    "diag": diag,
                    "last": now,
                    "hits": 0,
                }
                continue

            gap = now - state["last"]

            if gap <= 0 or gap > 1.5:
                self.first_head_states[tid] = {
                    "anchor": center.copy(),
                    "diag": diag,
                    "last": now,
                    "hits": 0,
                }
                continue

            displacement = float(
                np.linalg.norm(
                    center - state["anchor"]
                )
            )

            tolerance = max(
                3.5,
                0.006 * state["diag"],
            )

            normalized = (
                max(0.0, displacement - tolerance)
                / max(state["diag"], 1.0)
            )

            # Confirm real head motion across two observations.
            if normalized > 0.012:
                state["hits"] += 1
            else:
                state["hits"] = 0

            state["last"] = now

            if state["hits"] < 2:
                continue

            state["anchor"] = center.copy()
            state["diag"] = diag
            state["hits"] = 0

            # Reset only the selected fully-visible track.
            full_state = self.full.states.get(tid)

            if full_state is not None:
                measurement = self.full._measurement(person)

                if measurement is not None:
                    points, weights, body_diag = measurement

                    full_state.update({
                        "start": now,
                        "last": now,
                        "anchor": points.copy(),
                        "anchor_weights": weights.copy(),
                        "diag": body_diag,
                        "move_hits": 0,
                    })

                    output[tid] = {
                        "state": "OBSERVING",
                        "seconds": 0.0,
                        "reason": "confirmed_first_head_motion",
                        "normalized_motion": round(
                            normalized, 6
                        ),
                        "valid_joints": int(
                            np.count_nonzero(weights > 0.12)
                        ),
                    }

                    print(
                        "FIRST PERSON HEAD MOTION RESET:",
                        tid,
                    )

        for person in full_tracks:
            tid = int(person["track_id"])
            self.head_states.pop(tid, None)

        # Head-centered observation for occluded people.
        for person, info in head_tracks:
            tid = int(person["track_id"])
            center = info["center"]
            diag = info["diag"]
            current_box = info["bbox"]

            state = self.head_states.get(tid)

            if state is None:
                state = {
                    "center": center.copy(),
                    "bbox": current_box.copy(),
                    "diag": diag,
                    "seconds": 0.0,
                    "last": now,
                    "hits": 0,
                }
                self.head_states[tid] = state

                output[tid] = {
                    "state": "HEAD_ONLY",
                    "seconds": 0.0,
                    "reason": "head_baseline",
                }
                continue

            gap = now - state["last"]

            if gap <= 0 or gap > self.max_hidden:
                state.update({
                    "center": center.copy(),
                    "bbox": current_box.copy(),
                    "diag": diag,
                    "seconds": 0.0,
                    "last": now,
                    "hits": 0,
                })
                output[tid] = {
                    "state": "HEAD_ONLY",
                    "seconds": 0.0,
                    "reason": "reacquired_baseline",
                }
                continue

            old_box = state["bbox"]
            old_center = (old_box[:2] + old_box[2:]) / 2
            new_center = (
                current_box[:2] + current_box[2:]
            ) / 2

            box_displacement = float(
                np.linalg.norm(new_center - old_center)
            )

            # Strong position discontinuity: identity uncertain.
            if box_displacement > 0.40 * state["diag"]:
                state.update({
                    "center": center.copy(),
                    "bbox": current_box.copy(),
                    "diag": diag,
                    "seconds": 0.0,
                    "last": now,
                    "hits": 0,
                })
                output[tid] = {
                    "state": "UNKNOWN",
                    "seconds": 0.0,
                    "reason": "possible_identity_change",
                }
                continue

            displacement = float(
                np.linalg.norm(center - state["center"])
            )

            tolerance = max(
                3.5,
                0.006 * state["diag"],
            )

            motion = (
                max(0.0, displacement - tolerance)
                / state["diag"]
            )

            moving = motion > self.head_motion_fraction

            if moving:
                state["hits"] += 1
            else:
                state["hits"] = 0

            confirmed = (
                state["hits"] >= self.confirm_motion
            )

            if confirmed:
                state["center"] = center.copy()
                state["seconds"] = 0.0
                state["hits"] = 0
            elif gap <= 1.5:
                state["seconds"] += gap

            state["last"] = now
            state["bbox"] = current_box.copy()

            output[tid] = {
                "state": "HEAD_ONLY",
                "seconds": round(state["seconds"], 2),
                "reason": (
                    "confirmed_head_motion_reset"
                    if confirmed
                    else "head_motion_observation"
                ),
                "normalized_motion": round(
                    motion, 6
                ),
                "head_visible": info["head_visible"],
            }

        for tid in list(self.head_states):
            if tid not in seen:
                state = self.head_states[tid]
                if now - state["last"] > self.max_hidden:
                    del self.head_states[tid]

        return output


def save_stillness_crops(frame, tracks, states, captured, directory):
    """Save one clean person crop per confirmed 20s stillness episode."""
    from datetime import datetime

    height, width = frame.shape[:2]
    eligible = set()

    for person in tracks:
        if person.get("observed") is not True:
            continue

        tid = int(person["track_id"])
        state = states.get(tid, {})
        seconds = float(state.get("seconds", 0.0))

        if state.get("state") in ("UNKNOWN", "HEAD_ONLY") or seconds < 20.0:
            continue

        eligible.add(tid)

        if tid in captured:
            continue

        box = np.asarray(person["bbox"], dtype=float)

        if box.shape != (4,) or not np.isfinite(box).all():
            continue

        x1, y1, x2, y2 = np.round(box).astype(int)

        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2].copy()

        filename = (
            f"stillness_id{tid}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        )

        path = directory / filename

        if cv2.imwrite(str(path), crop):
            captured.add(tid)
            print(f"STILLNESS 20s CAPTURED: {path}")
        else:
            print(f"WARNING: Image save failed: {path}")

    captured.intersection_update(eligible)


def draw_stillness_only(image, tracks, states):
    """Render only the stillness timer above each observed person."""
    height, width = image.shape[:2]

    for person in tracks:
        if person.get("observed") is not True:
            continue

        tid = int(person["track_id"])
        state = states.get(tid, {})

        if state.get("state") == "UNKNOWN":
            label = "UNKNOWN"
        elif state.get("state") == "HEAD_ONLY":
            seconds = float(state.get("seconds", 0.0))
            label = f"HEAD ONLY: {seconds:.1f}s"
        else:
            seconds = float(state.get("seconds", 0.0))
            label = f"STILL: {seconds:.1f}s"

        box = np.asarray(person["bbox"], dtype=float)

        if box.shape != (4,) or not np.isfinite(box).all():
            continue

        x1, y1 = np.round(box[:2]).astype(int)

        x = max(0, min(width - 170, x1))
        y = max(26, min(height - 4, y1 - 8))

        cv2.rectangle(
            image, (x, y - 24), (x + 165, y + 4),
            (14, 18, 24), -1
        )

        cv2.putText(
            image, label, (x + 5, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2,
            cv2.LINE_AA
        )

    return image


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
        object_model = None
        print("OBJECT DETECTOR: DISABLED FOR SPEED TEST")
        print("GPU PRECISION:", args.gpu_precision.upper())
        print("ANNOTATION BUFFER:", "COPY" if args.annotation_copy else "IN_PLACE")

        print("\nLIVE YOLO POSE STARTED")
        print("Press Q to stop.")

        # ====================================================
        # LIVE PROCESSING LOOP
        # ====================================================

        stillness_tracker = AdaptiveHeadStillness()

        validation_rois = None
        validation_dir = (
            Path(__file__).resolve().parents[1]
            / "reports/node_5_10_3/multitrack_v2"
        )
        validation_dir.mkdir(parents=True, exist_ok=True)
        validation_file = validation_dir / (
            "validation_" + str(uuid4()) + ".jsonl"
        )
        print("VALIDATION LOG:", validation_file)
        print("ROI SELECTION WILL OPEN ON FIRST FRAME")

        captured_ids = set()
        img_dir = Path(__file__).resolve().parents[1] / "img"
        img_dir.mkdir(parents=True, exist_ok=True)
        print("STILLNESS IMAGE DIRECTORY:", img_dir)
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

            if validation_rois is None:
                validation_rois = {}  # ROI selection disabled

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
            detected_objects = []  # DIAGNOSTIC ONLY

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

            validation_record = {
                "frame_id": behavior_processed,
                "timestamp": time.monotonic(),
                "image_shape": list(frame.shape[:2]),
                "boxes": coordinates.tolist(),
                "scores": scores.tolist(),
                "pose9": poses.tolist(),
                "pose17": (
                    wrist_coco.tolist()
                    if wrist_sequence == sequence
                    and wrist_coco is not None
                    else None
                ),
                "tracks": [
                    {
                        "track_id": int(p["track_id"]),
                        "observed": bool(p.get("observed")),
                        "confidence": float(
                            p.get("confidence", 0.0)
                        ),
                        "bbox": [
                            float(v) for v in p["bbox"]
                        ],
                        "workstation": workstation_for_box(
                            p["bbox"], validation_rois
                        ),
                    }
                    for p in tracked_persons
                ],
                "stillness": {
                    str(k): v
                    for k, v in stillness_state.items()
                },
            }

            with validation_file.open(
                "a", encoding="utf-8"
            ) as data_log:
                data_log.write(
                    json.dumps(
                        validation_record,
                        ensure_ascii=False,
                    ) + chr(10)
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

            save_stillness_crops(
                frame, tracked_persons, stillness_state,
                captured_ids, img_dir,
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

            annotated = draw_stillness_only(
                annotated, tracked_persons, stillness_state
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
