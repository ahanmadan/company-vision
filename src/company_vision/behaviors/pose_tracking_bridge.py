"""Node 5.9.1: adapt YOLO observations and stable tracked poses.

This bridge does not run models, cameras, rules or events.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from math import isfinite
from typing import Any

import numpy as np

from company_vision.behaviors.feature_extractor import (
    FeatureExtractor,
    FrameFeatures,
)
from company_vision.behaviors.input_adapter import (
    DEFAULT_KEYPOINT_NAMES,
    BehaviorInputAdapter,
)
from company_vision.behaviors.schemas import BehaviorFrame


class PoseBridgeError(ValueError):
    """Invalid or ambiguous upstream tracking data."""


class PoseTrackingBridge:
    """Convert current tracking outputs into behavior-ready data."""

    def __init__(self, min_keypoint_confidence: float = 0.25):
        if (
            not isfinite(min_keypoint_confidence)
            or not 0 < min_keypoint_confidence <= 1
        ):
            raise ValueError("Invalid keypoint confidence threshold")

        self.min_keypoint_confidence = min_keypoint_confidence
        self.adapter = BehaviorInputAdapter()
        self.extractor = FeatureExtractor()

    @staticmethod
    def _array(value: Any, shape: tuple[int, ...], name: str):
        try:
            data = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise PoseBridgeError(f"Invalid {name}") from exc

        if data.shape != shape or not np.isfinite(data).all():
            raise PoseBridgeError(
                f"{name} must have shape {shape} and finite values"
            )
        return data

    @staticmethod
    def _confidence(value: Any, name: str) -> float:
        if isinstance(value, bool):
            raise PoseBridgeError(f"Invalid {name}")

        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise PoseBridgeError(f"Invalid {name}") from exc

        if not isfinite(number) or not 0 <= number <= 1:
            raise PoseBridgeError(f"{name} must be in [0,1]")

        return number

    def _point(self, row: Sequence[float], width: int, height: int):
        x, y, confidence = map(float, row)

        valid = (
            confidence >= self.min_keypoint_confidence
            and 0 <= x <= width
            and 0 <= y <= height
        )

        if not valid:
            return {
                "x": None,
                "y": None,
                "confidence": confidence,
                "valid": False,
            }

        return {
            "x": x,
            "y": y,
            "confidence": confidence,
            "valid": True,
        }

    def _points(self, values: Any, width: int, height: int):
        matrix = self._array(values, (9, 3), "tracked keypoints")

        return {
            name: self._point(matrix[index], width, height)
            for index, name in enumerate(DEFAULT_KEYPOINT_NAMES)
        }

    def _supplement(
        self,
        points: dict,
        extras: Mapping[str, Any],
        width: int,
        height: int,
    ) -> None:
        """Use only wrist points explicitly matched to a track."""

        if not isinstance(extras, Mapping):
            raise PoseBridgeError("Supplemental points must be a mapping")

        allowed = {"left_wrist", "right_wrist"}

        if set(extras) - allowed:
            raise PoseBridgeError("Only wrist supplements are supported")

        for name, value in extras.items():
            row = self._array(value, (3,), name)
            points[name] = self._point(row, width, height)

    def _person(
        self,
        item: Mapping[str, Any],
        width: int,
        height: int,
        supplemental: Mapping[str, Any] | None,
    ):
        if not isinstance(item, Mapping):
            raise PoseBridgeError("Tracked person must be a mapping")

        track_id = item.get("track_id")

        if (
            isinstance(track_id, bool)
            or not isinstance(track_id, int | np.integer)
            or track_id < 0
        ):
            raise PoseBridgeError("Valid track_id is required")

        observed = item.get("observed") is True
        predicted = item.get("predicted") is True

        if observed == predicted:
            raise PoseBridgeError(
                "Track must be observed or predicted, not both/neither"
            )

        if observed and item.get("behavior_eligible") is not True:
            eligible = False
        else:
            eligible = observed and not predicted

        bbox = self._array(item.get("bbox"), (4,), "bbox")

        # Tracking filters can produce small out-of-frame box drift.
        # Clip coordinates, but never fabricate an in-frame person.
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, width)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, height)

        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise PoseBridgeError("Tracked bbox is outside valid frame")

        keypoints = self._points(
            item.get("keypoints"), width, height
        )

        raw = item.get("raw_keypoints")

        raw_points = (
            self._points(raw, width, height)
            if raw is not None
            else None
        )

        weights = item.get("keypoint_weights")

        if weights is None:
            weight_map = {}
        else:
            weights = self._array(
                weights, (9,), "keypoint_weights"
            )

            if np.any((weights < 0) | (weights > 1)):
                raise PoseBridgeError("Invalid keypoint weights")

            weight_map = {
                name: float(weights[index])
                for index, name in enumerate(DEFAULT_KEYPOINT_NAMES)
            }

        if supplemental is not None:
            if not observed:
                raise PoseBridgeError(
                    "Predicted tracks cannot receive observed wrist points"
                )
            self._supplement(
                keypoints, supplemental, width, height
            )

        confidence = self._confidence(
            item.get("confidence"), "person confidence"
        )

        missed = item.get("missed", 0)

        if (
            isinstance(missed, bool)
            or not isinstance(missed, int | np.integer)
            or missed < 0
        ):
            raise PoseBridgeError("Invalid missed count")

        return {
            "track_id": int(track_id),
            "bbox": bbox.tolist(),
            "keypoints": keypoints,
            "raw_keypoints": raw_points,
            "keypoint_weights": weight_map,
            "confidence": confidence,
            "observed": observed,
            "predicted": predicted,
            "behavior_eligible": eligible,
            "track_lifecycle": item.get("track_lifecycle"),
            "missed": int(missed),
        }

    @staticmethod
    def _objects_from_yolo(result):
        """Read one already-computed Ultralytics detection result."""

        if result is None:
            return []

        boxes = getattr(result, "boxes", None)

        if boxes is None or len(boxes) == 0:
            return []

        names = getattr(result, "names", {})

        xyxy = boxes.xyxy.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()

        if (
            xyxy.ndim != 2
            or xyxy.shape[1] != 4
            or len(classes) != len(xyxy)
            or len(scores) != len(xyxy)
        ):
            raise PoseBridgeError("Invalid YOLO object result shapes")

        objects = []

        for index, box in enumerate(xyxy):
            class_id = int(classes[index])

            if isinstance(names, Mapping):
                name = names.get(class_id, str(class_id))
            else:
                name = names[class_id]

            objects.append({
                "object_id": index,
                "class_id": class_id,
                "class_name": str(name),
                "confidence": float(scores[index]),
                "bbox": box.tolist(),
            })

        return objects

    def build(
        self,
        *,
        camera_id: str,
        session_id: str,
        frame_id: int,
        timestamp_utc: datetime,
        monotonic_timestamp: float,
        width: int,
        height: int,
        tracked_persons: Sequence[Mapping[str, Any]],
        objects: Sequence[Mapping[str, Any]] | None = None,
        yolo_object_result: Any | None = None,
        object_detection_status: str = "not_run",
        supplemental_wrists: Mapping[int, Mapping[str, Any]]
        | None = None,
        track_lifecycle_events: Sequence[Mapping] = (),
        source_status: str = "ok",
    ) -> BehaviorFrame:
        """Produce canonical BehaviorFrame without running inference."""

        if (
            isinstance(width, bool)
            or isinstance(height, bool)
            or width < 1
            or height < 1
        ):
            raise PoseBridgeError("Invalid image dimensions")

        if objects is not None and yolo_object_result is not None:
            raise PoseBridgeError(
                "Use either objects or yolo_object_result"
            )

        if yolo_object_result is not None:
            objects = self._objects_from_yolo(
                yolo_object_result
            )
            object_detection_status = "available"

        elif objects is not None:
            object_detection_status = "available"

        extras = dict(supplemental_wrists or {})

        tracked_ids = {
            item.get("track_id")
            for item in tracked_persons
        }

        if set(extras) - tracked_ids:
            raise PoseBridgeError(
                "Wrist supplements reference unknown tracks"
            )

        people = [
            self._person(
                item,
                width,
                height,
                extras.get(item.get("track_id")),
            )
            for item in tracked_persons
        ]

        if objects is None:
            objects = ()

        if (
            object_detection_status != "available"
            and objects
        ):
            raise PoseBridgeError(
                "Objects require successful object detection"
            )

        try:
            return self.adapter.from_tracking_output(
                camera_id=camera_id,
                session_id=session_id,
                frame_id=frame_id,
                timestamp_utc=timestamp_utc,
                monotonic_timestamp=monotonic_timestamp,
                width=width,
                height=height,
                persons=people,
                objects=objects,
                object_detection_status=object_detection_status,
                track_lifecycle_events=track_lifecycle_events,
                source_status=source_status,
            )
        except (ValueError, TypeError) as exc:
            raise PoseBridgeError(
                f"BehaviorFrame conversion failed: {exc}"
            ) from exc

    def extract(self, **kwargs) -> FrameFeatures:
        """Produce normalized features ready for plugin contexts."""

        return self.extractor.extract(self.build(**kwargs))