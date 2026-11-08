from dataclasses import dataclass, field
from math import hypot

import numpy as np
from scipy.optimize import linear_sum_assignment

# ============================================================
# PROJECT KEYPOINT CONTRACT: COCO INDICES 0-8 ONLY
# ============================================================

KEYPOINT_COUNT = 9

KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
)

KEYPOINT_PRIORITY = np.array(
    [
        0.85,  # nose
        0.45,  # left eye
        0.45,  # right eye
        0.40,  # left ear
        0.40,  # right ear
        1.00,  # left shoulder
        1.00,  # right shoulder
        0.95,  # left elbow
        0.95,  # right elbow
    ],
    dtype=np.float32,
)


# ============================================================
# TRACK
# ============================================================

@dataclass
class Track:
    stable_id: int
    state: np.ndarray
    covariance: np.ndarray
    pose: np.ndarray

    pose_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(
            (KEYPOINT_COUNT, 2),
            dtype=np.float32,
        )
    )

    joint_weights: np.ndarray = field(
        default_factory=lambda: np.zeros(
            KEYPOINT_COUNT,
            dtype=np.float32,
        )
    )

    hits: int = 1
    missed: int = 0
    age: int = 1
    confirmed: bool = False
    last_confidence: float = 0.0


# ============================================================
# GEOMETRY
# ============================================================

def _iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )

    area_a = (
        max(0.0, a[2] - a[0])
        * max(0.0, a[3] - a[1])
    )

    area_b = (
        max(0.0, b[2] - b[0])
        * max(0.0, b[3] - b[1])
    )

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def _to_state(box):
    x1, y1, x2, y2 = box

    return np.array(
        [
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            x2 - x1,
            y2 - y1,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )


def _to_box(state):
    cx, cy, width, height = state[:4]

    width = max(1.0, float(width))
    height = max(1.0, float(height))

    return np.array(
        [
            cx - width / 2,
            cy - height / 2,
            cx + width / 2,
            cy + height / 2,
        ],
        dtype=np.float32,
    )


# ============================================================
# STABLE POSE MANAGER
# ============================================================

class StablePoseManager:
    """
    Stable person association and dynamic 9-keypoint filtering.

    Input:
        boxes: (N, 4)
        poses: (N, 9, 3)
        scores: (N,)

    Output:
        keypoints: (9, 3)
        raw_keypoints: (9, 3)
        keypoint_weights: (9,)

    COCO indices 0-8 only.
    """

    def __init__(
        self,
        *,
        confirm_hits=4,
        max_missed=15,
        min_keypoints=3,
        keypoint_conf=0.25,
        match_distance=0.85,
        ema_alpha=0.35,
        process_noise=1.0,
        measurement_noise=4.0,
    ):
        if confirm_hits < 1 or max_missed < 0:
            raise ValueError("Invalid tracking limits")

        if not 1 <= min_keypoints <= KEYPOINT_COUNT:
            raise ValueError("Invalid min_keypoints")

        if not 0 < keypoint_conf <= 1:
            raise ValueError("Invalid keypoint_conf")

        if not 0 < match_distance <= 2:
            raise ValueError("Invalid match_distance")

        if not 0 < ema_alpha <= 1:
            raise ValueError("Invalid ema_alpha")

        if process_noise <= 0 or measurement_noise <= 0:
            raise ValueError("Noise values must be positive")

        self.confirm_hits = confirm_hits
        self.max_missed = max_missed
        self.min_keypoints = min_keypoints
        self.keypoint_conf = keypoint_conf
        self.match_distance = match_distance
        self.ema_alpha = ema_alpha
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        self.joint_priority = KEYPOINT_PRIORITY.copy()

        self.tracks = {}
        self.next_id = 1
        self.frame_index = 0

        self._h = np.zeros(
            (4, 8),
            dtype=np.float32,
        )

        self._h[:, :4] = np.eye(
            4,
            dtype=np.float32,
        )

        self._r = (
            np.eye(4, dtype=np.float32)
            * measurement_noise
        )

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    @staticmethod
    def _validate(boxes, poses, scores):
        boxes = np.asarray(boxes, dtype=np.float32)
        poses = np.asarray(poses, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)

        if boxes.size == 0:
            boxes = np.empty(
                (0, 4),
                dtype=np.float32,
            )

        if poses.size == 0:
            poses = np.empty(
                (0, KEYPOINT_COUNT, 3),
                dtype=np.float32,
            )

        if scores.size == 0:
            scores = np.empty(
                (0,),
                dtype=np.float32,
            )

        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(
                "boxes must have shape (N, 4)"
            )

        if poses.shape != (
            len(boxes),
            KEYPOINT_COUNT,
            3,
        ):
            raise ValueError(
                "poses must have shape (N, 9, 3); "
                "slice YOLO COCO keypoints with [:, :9, :] "
                "before calling StablePoseManager"
            )

        if scores.shape != (len(boxes),):
            raise ValueError(
                "scores must have shape (N,)"
            )

        if not (
            np.isfinite(boxes).all()
            and np.isfinite(poses).all()
            and np.isfinite(scores).all()
        ):
            raise ValueError("Non-finite input")

        if np.any(boxes[:, 2:] <= boxes[:, :2]):
            raise ValueError("Invalid bounding boxes")

        if np.any(
            (poses[:, :, 2] < 0)
            | (poses[:, :, 2] > 1)
        ):
            raise ValueError(
                "Invalid keypoint confidence"
            )

        if np.any((scores < 0) | (scores > 1)):
            raise ValueError(
                "Invalid detection confidence"
            )

        return boxes, poses, scores

    # ========================================================
    # KALMAN FILTER
    # ========================================================

    @staticmethod
    def _transition():
        transition = np.eye(
            8,
            dtype=np.float32,
        )

        for index in range(4):
            transition[index, index + 4] = 1.0

        return transition

    def _predict(self, track):
        transition = self._transition()

        noise = (
            np.eye(8, dtype=np.float32)
            * self.process_noise
        )

        track.state = transition @ track.state

        track.covariance = (
            transition
            @ track.covariance
            @ transition.T
            + noise
        )

        track.state[2:4] = np.maximum(
            track.state[2:4],
            1.0,
        )

    def _correct(self, track, box):
        measurement = _to_state(box)[:4]

        innovation = (
            measurement
            - self._h @ track.state
        )

        covariance = (
            self._h
            @ track.covariance
            @ self._h.T
            + self._r
        )

        try:
            gain = np.linalg.solve(
                covariance,
                self._h @ track.covariance,
            ).T

        except np.linalg.LinAlgError:
            raise RuntimeError(
                "Kalman covariance is singular"
            ) from None

        track.state = (
            track.state
            + gain @ innovation
        )

        identity = np.eye(
            8,
            dtype=np.float32,
        )

        residual = identity - gain @ self._h

        track.covariance = (
            residual
            @ track.covariance
            @ residual.T
            + gain @ self._r @ gain.T
        )

        track.covariance = (
            track.covariance
            + track.covariance.T
        ) * 0.5

    # ========================================================
    # GEOMETRIC VALIDITY
    # ========================================================

    @staticmethod
    def _geometry_weights(pose, box):
        x1, y1, x2, y2 = box

        width = max(1.0, float(x2 - x1))
        height = max(1.0, float(y2 - y1))

        margin_x = width * 0.15
        margin_y = height * 0.15

        x = pose[:, 0]
        y = pose[:, 1]

        inside = (
            (x >= x1 - margin_x)
            & (x <= x2 + margin_x)
            & (y >= y1 - margin_y)
            & (y <= y2 + margin_y)
        )

        return np.where(
            inside,
            1.0,
            0.25,
        ).astype(np.float32)

    # ========================================================
    # INITIAL DYNAMIC WEIGHTS
    # ========================================================

    def _initial_joint_weights(self, pose, box):
        confidence = pose[:, 2]

        geometry = self._geometry_weights(
            pose,
            box,
        )

        confidence_factor = np.clip(
            (
                confidence - self.keypoint_conf
            ) / (
                1.0 - self.keypoint_conf
            ),
            0.0,
            1.0,
        )

        confidence_factor = (
            0.20 + 0.80 * confidence_factor
        )

        weights = (
            self.joint_priority
            * confidence_factor
            * geometry
        )

        weights = np.where(
            confidence >= self.keypoint_conf,
            weights,
            0.0,
        )

        return np.clip(
            weights,
            0.0,
            1.0,
        ).astype(np.float32)

    # ========================================================
    # DYNAMIC JOINT WEIGHTING
    # ========================================================

    def _dynamic_smoothing(
        self,
        track,
        observation,
        box,
    ):
        result = observation.copy()
        old = track.pose.copy()

        weights = np.zeros(
            KEYPOINT_COUNT,
            dtype=np.float32,
        )

        body_scale = max(
            hypot(
                float(track.state[2]),
                float(track.state[3]),
            ),
            1.0,
        )

        geometry = self._geometry_weights(
            observation,
            box,
        )

        initial_weights = self._initial_joint_weights(
            observation,
            box,
        )

        for index in range(KEYPOINT_COUNT):
            confidence = float(
                observation[index, 2]
            )

            old_confidence = float(
                old[index, 2]
            )

            if confidence < self.keypoint_conf:
                result[index, :2] = old[index, :2]
                result[index, 2] = confidence
                weights[index] = 0.0
                continue

            current = observation[index, :2]
            previous = old[index, :2]

            if old_confidence < self.keypoint_conf:
                result[index, :2] = current
                result[index, 2] = confidence

                track.pose_velocity[index] = 0.0
                weights[index] = initial_weights[index]
                continue

            velocity = track.pose_velocity[index]

            predicted = previous + velocity

            residual = current - predicted

            residual_distance = float(
                np.linalg.norm(residual)
            )

            normalized_residual = (
                residual_distance / body_scale
            )

            # 1. Detection confidence
            confidence_factor = float(
                np.clip(
                    (
                        confidence - self.keypoint_conf
                    ) / (
                        1.0 - self.keypoint_conf
                    ),
                    0.0,
                    1.0,
                )
            )

            confidence_factor = (
                0.20 + 0.80 * confidence_factor
            )

            # 2. Temporal stability
            stability_factor = float(
                np.exp(
                    -normalized_residual / 0.06
                )
            )

            stability_factor = max(
                0.15,
                stability_factor,
            )

            # 3. Geometric validity
            geometry_factor = float(
                geometry[index]
            )

            # 4. Anatomical priority
            priority_factor = float(
                self.joint_priority[index]
            )

            # Final dynamic trust score
            weight = (
                confidence_factor
                * stability_factor
                * geometry_factor
                * priority_factor
            )

            weight = float(
                np.clip(weight, 0.0, 1.0)
            )

            weights[index] = weight

            # High trust: stronger current measurement.
            # Low trust: stronger temporal smoothing.
            alpha = (
                self.ema_alpha
                * (0.20 + 0.80 * weight)
            )

            alpha = float(
                np.clip(alpha, 0.04, 0.85)
            )

            outlier_limit = (
                0.22 + 0.18 * confidence
            )

            if (
                normalized_residual > outlier_limit
                and confidence < 0.70
            ):
                filtered = predicted
            else:
                filtered = (
                    predicted
                    + alpha * residual
                )

            result[index, :2] = filtered
            result[index, 2] = confidence

            displacement = filtered - previous

            track.pose_velocity[index] = (
                0.65 * velocity
                + 0.35 * displacement
            )

        return result, weights

    # ========================================================
    # TRACK ASSOCIATION
    # ========================================================

    def _cost(self, track, box, pose):
        predicted_box = _to_box(track.state)

        predicted_center = (
            predicted_box[:2] + predicted_box[2:]
        ) / 2

        detected_center = (
            box[:2] + box[2:]
        ) / 2

        scale = max(
            30.0,
            hypot(
                predicted_box[2] - predicted_box[0],
                predicted_box[3] - predicted_box[1],
            ),
        )

        distance = float(
            np.linalg.norm(
                predicted_center - detected_center
            ) / scale
        )

        overlap = _iou(
            predicted_box,
            box,
        )

        valid = (
            (track.pose[:, 2] >= self.keypoint_conf)
            & (pose[:, 2] >= self.keypoint_conf)
        )

        if valid.sum() >= 3:
            joint_trust = np.minimum(
                track.joint_weights[valid],
                self._initial_joint_weights(
                    pose,
                    box,
                )[valid],
            )

            joint_trust = np.maximum(
                joint_trust,
                0.05,
            )

            distances = np.linalg.norm(
                track.pose[valid, :2]
                - pose[valid, :2],
                axis=1,
            ) / scale

            pose_distance = float(
                np.average(
                    distances,
                    weights=joint_trust,
                )
            )

        else:
            pose_distance = distance

        cost = (
            0.45 * min(distance, 2.0)
            + 0.30 * (1.0 - overlap)
            + 0.25 * min(pose_distance, 2.0)
        )

        if (
            distance > self.match_distance * 1.8
            and overlap < 0.01
        ):
            return np.inf

        return cost

    # ========================================================
    # CREATE TRACK
    # ========================================================

    def _new_track(
        self,
        box,
        pose,
        confidence,
    ):
        state = _to_state(box)

        covariance = np.diag(
            [25, 25, 25, 25, 100, 100, 100, 100]
        ).astype(np.float32)

        track_id = self.next_id
        self.next_id += 1

        track = Track(
            stable_id=track_id,
            state=state,
            covariance=covariance,
            pose=pose.copy(),
            joint_weights=self._initial_joint_weights(
                pose,
                box,
            ),
            last_confidence=float(confidence),
        )

        track.confirmed = (
            self.confirm_hits == 1
        )

        self.tracks[track_id] = track

    # ========================================================
    # UPDATE
    # ========================================================

    def update(self, boxes, poses, scores):
        boxes, poses, scores = self._validate(
            boxes,
            poses,
            scores,
        )

        self.frame_index += 1

        track_ids = list(self.tracks)

        for track in self.tracks.values():
            self._predict(track)
            track.age += 1

        matches = []

        unmatched_tracks = set(track_ids)

        unmatched_detections = set(
            range(len(boxes))
        )

        if track_ids and len(boxes):
            costs = np.full(
                (len(track_ids), len(boxes)),
                1e6,
                dtype=np.float64,
            )

            for row, track_id in enumerate(track_ids):
                track = self.tracks[track_id]

                for column in range(len(boxes)):
                    cost = self._cost(
                        track,
                        boxes[column],
                        poses[column],
                    )

                    if np.isfinite(cost):
                        costs[row, column] = cost

            rows, columns = linear_sum_assignment(
                costs
            )

            for row, column in zip(
                rows,
                columns,
                strict=True,
            ):
                if costs[row, column] > self.match_distance:
                    continue

                track_id = track_ids[row]

                matches.append(
                    (track_id, int(column))
                )

                unmatched_tracks.discard(track_id)
                unmatched_detections.discard(int(column))

        output = []

        # ----------------------------------------------------
        # MATCHED TRACKS
        # ----------------------------------------------------

        for track_id, detection_index in matches:
            track = self.tracks[track_id]

            box = boxes[detection_index]
            pose = poses[detection_index]

            raw_pose = pose.copy()

            self._correct(
                track,
                box,
            )

            filtered_pose, dynamic_weights = (
                self._dynamic_smoothing(
                    track,
                    pose,
                    box,
                )
            )

            track.pose = filtered_pose
            track.joint_weights = dynamic_weights

            track.hits += 1
            track.missed = 0

            track.last_confidence = float(
                scores[detection_index]
            )

            if track.hits >= self.confirm_hits:
                track.confirmed = True

            if track.confirmed:
                output.append({
                    "track_id": track_id,
                    "bbox": _to_box(track.state),
                    "keypoints": track.pose.copy(),
                    "raw_keypoints": raw_pose,
                    "keypoint_weights": (
                        track.joint_weights.copy()
                    ),
                    "confidence": track.last_confidence,
                    "observed": True,
                    "predicted": False,
                    "missed": 0,
                })

        # ----------------------------------------------------
        # UNMATCHED TRACKS
        # ----------------------------------------------------

        for track_id in list(unmatched_tracks):
            track = self.tracks[track_id]

            track.missed += 1
            track.hits = 0

            if track.missed > self.max_missed:
                del self.tracks[track_id]
                continue

            if track.confirmed:
                output.append({
                    "track_id": track_id,
                    "bbox": _to_box(track.state),
                    "keypoints": track.pose.copy(),
                    "raw_keypoints": None,
                    "keypoint_weights": (
                        track.joint_weights.copy()
                    ),
                    "confidence": track.last_confidence,
                    "observed": False,
                    "predicted": True,
                    "missed": track.missed,
                })

        # ----------------------------------------------------
        # NEW DETECTIONS
        # ----------------------------------------------------

        for detection_index in unmatched_detections:
            pose = poses[detection_index]

            valid = int(
                (
                    pose[:, 2] >= self.keypoint_conf
                ).sum()
            )

            if valid < self.min_keypoints:
                continue

            self._new_track(
                boxes[detection_index],
                pose,
                scores[detection_index],
            )

            track = self.tracks[
                self.next_id - 1
            ]

            if track.confirmed:
                output.append({
                    "track_id": track.stable_id,
                    "bbox": _to_box(track.state),
                    "keypoints": track.pose.copy(),
                    "raw_keypoints": pose.copy(),
                    "keypoint_weights": (
                        track.joint_weights.copy()
                    ),
                    "confidence": track.last_confidence,
                    "observed": True,
                    "predicted": False,
                    "missed": 0,
                })

        return output

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):
        self.tracks.clear()
        self.next_id = 1
        self.frame_index = 0