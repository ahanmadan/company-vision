from dataclasses import dataclass, field
from math import hypot

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Track:
    stable_id: int
    state: np.ndarray
    covariance: np.ndarray
    pose: np.ndarray
    pose_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros((17, 2), dtype=np.float32)
    )
    hits: int = 1
    missed: int = 0
    age: int = 1
    confirmed: bool = False
    last_confidence: float = 0.0


def _iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = max(0.0, x2-x1) * max(0.0, y2-y1)

    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])

    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _to_state(box):
    x1, y1, x2, y2 = box
    return np.array(
        [
            (x1+x2)/2,
            (y1+y2)/2,
            x2-x1,
            y2-y1,
            0, 0, 0, 0,
        ],
        dtype=np.float32,
    )


def _to_box(state):
    cx, cy, width, height = state[:4]
    width = max(1.0, float(width))
    height = max(1.0, float(height))

    return np.array(
        [
            cx-width/2,
            cy-height/2,
            cx+width/2,
            cy+height/2,
        ],
        dtype=np.float32,
    )


class StablePoseManager:
    """
    Stable person association using Hungarian assignment
    and an 8-dimensional constant-velocity Kalman filter.

    Predicted poses are explicitly marked as unobserved.
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

        if not 1 <= min_keypoints <= 17:
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

        self.tracks = {}
        self.next_id = 1
        self.frame_index = 0

        self._h = np.zeros((4, 8), dtype=np.float32)
        self._h[:, :4] = np.eye(4, dtype=np.float32)

        self._r = (
            np.eye(4, dtype=np.float32) * measurement_noise
        )

    @staticmethod
    def _validate(boxes, poses, scores):
        boxes = np.asarray(boxes, dtype=np.float32)
        poses = np.asarray(poses, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)

        if boxes.size == 0:
            boxes = np.empty((0, 4), dtype=np.float32)

        if poses.size == 0:
            poses = np.empty((0, 17, 3), dtype=np.float32)

        if scores.size == 0:
            scores = np.empty((0,), dtype=np.float32)

        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("boxes must have shape (N, 4)")

        if poses.shape != (len(boxes), 17, 3):
            raise ValueError("poses must have shape (N, 17, 3)")

        if scores.shape != (len(boxes),):
            raise ValueError("scores must have shape (N,)")

        if not (
            np.isfinite(boxes).all()
            and np.isfinite(poses).all()
            and np.isfinite(scores).all()
        ):
            raise ValueError("Non-finite input")

        if np.any(boxes[:, 2:] <= boxes[:, :2]):
            raise ValueError("Invalid bounding boxes")

        if np.any((poses[:, :, 2] < 0) | (poses[:, :, 2] > 1)):
            raise ValueError("Invalid keypoint confidence")

        if np.any((scores < 0) | (scores > 1)):
            raise ValueError("Invalid detection confidence")

        return boxes, poses, scores

    def _transition(self):
        f = np.eye(8, dtype=np.float32)

        for i in range(4):
            f[i, i+4] = 1.0

        return f

    def _predict(self, track):
        f = self._transition()
        q = np.eye(8, dtype=np.float32) * self.process_noise

        track.state = f @ track.state
        track.covariance = (
            f @ track.covariance @ f.T + q
        )

        track.state[2:4] = np.maximum(track.state[2:4], 1.0)

    def _correct(self, track, box):
        z = _to_state(box)[:4]

        innovation = z - self._h @ track.state

        s = (
            self._h @ track.covariance @ self._h.T
            + self._r
        )

        try:
            gain = np.linalg.solve(
                s,
                self._h @ track.covariance,
            ).T
        except np.linalg.LinAlgError:
            raise RuntimeError("Kalman covariance is singular") from None

        track.state = track.state + gain @ innovation

        identity = np.eye(8, dtype=np.float32)
        residual = identity - gain @ self._h

        track.covariance = (
            residual @ track.covariance @ residual.T
            + gain @ self._r @ gain.T
        )

        track.covariance = (
            track.covariance + track.covariance.T
        ) * 0.5

    def _cost(self, track, box, pose):
        predicted_box = _to_box(track.state)

        predicted_center = (
            predicted_box[:2] + predicted_box[2:]
        ) / 2

        detected_center = (box[:2] + box[2:]) / 2

        scale = max(
            30.0,
            hypot(
                predicted_box[2]-predicted_box[0],
                predicted_box[3]-predicted_box[1],
            ),
        )

        distance = float(
            np.linalg.norm(predicted_center-detected_center)
            / scale
        )

        overlap = _iou(predicted_box, box)

        valid = (
            (track.pose[:, 2] >= self.keypoint_conf)
            & (pose[:, 2] >= self.keypoint_conf)
        )

        if valid.sum() >= 3:
            pose_distance = float(
                np.linalg.norm(
                    track.pose[valid, :2]-pose[valid, :2],
                    axis=1,
                ).mean() / scale
            )
        else:
            pose_distance = distance

        cost = (
            0.45 * min(distance, 2.0)
            + 0.30 * (1.0-overlap)
            + 0.25 * min(pose_distance, 2.0)
        )

        # Reject implausible spatial associations.
        if distance > self.match_distance * 1.8 and overlap < 0.01:
            return np.inf

        return cost

    def _smooth_pose(self, track, observation):
        """Confidence-aware adaptive filtering for stable COCO keypoints."""

        result = observation.copy()
        old = track.pose.copy()

        width = max(float(track.state[2]), 1.0)
        height = max(float(track.state[3]), 1.0)
        body_scale = max(hypot(width, height), 1.0)

        # Head and shoulder joints need stronger temporal stability.
        sensitive_joints = {0, 1, 2, 3, 4, 5, 6}

        for index in range(17):
            confidence = float(observation[index, 2])
            old_confidence = float(old[index, 2])

            if confidence < self.keypoint_conf:
                result[index, :2] = old[index, :2]
                result[index, 2] = min(
                    confidence,
                    self.keypoint_conf - 0.001,
                )
                continue

            current = observation[index, :2]

            if old_confidence < self.keypoint_conf:
                result[index, :2] = current
                result[index, 2] = confidence
                continue

            distance = float(
                np.linalg.norm(current - old[index, :2])
            )

            normalized_jump = distance / body_scale

            # Higher confidence increases trust in the observation.
            confidence_factor = 0.65 + 0.35 * confidence

            # Large jumps reduce the influence of the new observation.
            alpha = (
                self.ema_alpha
                * confidence_factor
                / (1.0 + 2.5 * normalized_jump)
            )

            # Stronger stabilization for head and shoulders.
            if index in sensitive_joints:
                alpha *= 0.75

            alpha = float(np.clip(alpha, 0.08, 0.65))

            filtered = (
                alpha * current
                + (1.0 - alpha) * old[index, :2]
            )

            result[index, :2] = filtered
            result[index, 2] = confidence

            track.pose_velocity[index] = (
                0.70 * track.pose_velocity[index]
                + 0.30 * (filtered - old[index, :2])
            )

        return result

    def _new_track(self, box, pose, confidence):
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
            last_confidence=float(confidence),
        )

        track.confirmed = self.confirm_hits == 1
        self.tracks[track_id] = track

    def update(self, boxes, poses, scores):
        boxes, poses, scores = self._validate(
            boxes, poses, scores
        )

        self.frame_index += 1

        track_ids = list(self.tracks)

        for track in self.tracks.values():
            self._predict(track)
            track.age += 1

        matches = []
        unmatched_tracks = set(track_ids)
        unmatched_detections = set(range(len(boxes)))

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

            rows, columns = linear_sum_assignment(costs)

            for row, column in zip(rows, columns, strict=True):
                if costs[row, column] > self.match_distance:
                    continue

                track_id = track_ids[row]

                matches.append((track_id, int(column)))
                unmatched_tracks.discard(track_id)
                unmatched_detections.discard(int(column))

        output = []

        for track_id, detection_index in matches:
            track = self.tracks[track_id]

            box = boxes[detection_index]
            pose = poses[detection_index]

            self._correct(track, box)

            track.pose = self._smooth_pose(track, pose)
            track.hits += 1
            track.missed = 0
            track.last_confidence = float(scores[detection_index])

            if track.hits >= self.confirm_hits:
                track.confirmed = True

            if track.confirmed:
                output.append({
                    "track_id": track_id,
                    "bbox": _to_box(track.state),
                    "keypoints": track.pose.copy(),
                    "confidence": track.last_confidence,
                    "observed": True,
                    "predicted": False,
                    "missed": 0,
                })

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
                    "confidence": track.last_confidence,
                    "observed": False,
                    "predicted": True,
                    "missed": track.missed,
                })

        for detection_index in unmatched_detections:
            pose = poses[detection_index]

            valid = int(
                (pose[:, 2] >= self.keypoint_conf).sum()
            )

            if valid < self.min_keypoints:
                continue

            self._new_track(
                boxes[detection_index],
                pose,
                scores[detection_index],
            )

            track = self.tracks[self.next_id-1]

            if track.confirmed:
                output.append({
                    "track_id": track.stable_id,
                    "bbox": _to_box(track.state),
                    "keypoints": track.pose.copy(),
                    "confidence": track.last_confidence,
                    "observed": True,
                    "predicted": False,
                    "missed": 0,
                })

        return output

    def reset(self):
        self.tracks.clear()
        self.next_id = 1
        self.frame_index = 0
