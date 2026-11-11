from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from company_vision.core.schemas import FrameResult

from company_vision.behaviors.schemas import (
    BehaviorFrame,
    BehaviorObject,
    BehaviorPerson,
    TrackLifecycleSignal,
)


DEFAULT_KEYPOINT_NAMES = (
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

_ALLOWED_TRACK_METADATA = {
    "raw_keypoints",
    "keypoint_weights",
    "observed",
    "predicted",
    "behavior_eligible",
    "track_lifecycle",
    "missed",
}


class BehaviorInputError(ValueError):
    """Raised when source observations cannot be mapped safely."""


def _read(value: Any, name: str, default=None):
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _to_list(value):
    if value is None:
        return None

    if hasattr(value, "tolist"):
        return value.tolist()

    return value


class BehaviorInputAdapter:
    """
    Convert canonical FrameResult or neutral tracking records
    into the independent BehaviorFrame contract.

    This adapter does not import or execute YOLO, camera SDK,
    tracker, GPU, or behavior rules.
    """

    def __init__(self, keypoint_names=DEFAULT_KEYPOINT_NAMES):
        names = tuple(keypoint_names)

        if not names:
            raise ValueError("At least one keypoint name is required")

        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("Keypoint names must be non-empty strings")

        if len(names) != len(set(names)):
            raise ValueError("Keypoint names must be unique")

        self.keypoint_names = names

    @staticmethod
    def _bbox(value):
        if value is None:
            raise BehaviorInputError("Missing bbox")

        if all(hasattr(value, key) for key in ("x1", "y1", "x2", "y2")):
            return {
                "x1": float(value.x1),
                "y1": float(value.y1),
                "x2": float(value.x2),
                "y2": float(value.y2),
            }

        if isinstance(value, Mapping):
            if all(key in value for key in ("x1", "y1", "x2", "y2")):
                return {
                    key: float(value[key])
                    for key in ("x1", "y1", "x2", "y2")
                }

            if "xyxy" in value:
                value = value["xyxy"]

        value = _to_list(value)

        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise BehaviorInputError("bbox must contain four coordinates")

        if len(value) != 4:
            raise BehaviorInputError("bbox must contain four coordinates")

        return {
            "x1": float(value[0]),
            "y1": float(value[1]),
            "x2": float(value[2]),
            "y2": float(value[3]),
        }

    @staticmethod
    def _point(value):
        if value is None:
            return {
                "x": None,
                "y": None,
                "confidence": 0.0,
                "valid": False,
            }

        if isinstance(value, Mapping) or hasattr(value, "confidence"):
            x = _read(value, "x")
            y = _read(value, "y")
            confidence = _read(
                value,
                "confidence",
                _read(value, "conf"),
            )
            valid = _read(value, "valid")
        else:
            value = _to_list(value)

            if not isinstance(value, Sequence) or len(value) != 3:
                raise BehaviorInputError(
                    "Keypoint must contain x, y and confidence"
                )

            x, y, confidence = value
            valid = None

        if confidence is None:
            raise BehaviorInputError("Keypoint confidence is missing")

        if valid is None:
            valid = (
                x is not None
                and y is not None
                and float(confidence) > 0
            )

        return {
            "x": None if x is None else float(x),
            "y": None if y is None else float(y),
            "confidence": float(confidence),
            "valid": bool(valid),
        }

    def _keypoints(self, values):
        values = _to_list(values)

        if values is None:
            return {}

        if isinstance(values, Mapping):
            return {
                str(name): self._point(point)
                for name, point in values.items()
            }

        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise BehaviorInputError("Unsupported keypoint collection")

        values = list(values)

        if not values:
            return {}

        if all(_read(point, "name") is not None for point in values):
            result = {}

            for point in values:
                name = str(_read(point, "name"))

                if name in result:
                    raise BehaviorInputError(
                        f"Duplicate keypoint name: {name}"
                    )

                result[name] = self._point(point)

            return result

        if len(values) != len(self.keypoint_names):
            raise BehaviorInputError(
                "Array keypoint count does not match configured names"
            )

        return {
            name: self._point(point)
            for name, point in zip(
                self.keypoint_names,
                values,
                strict=True,
            )
        }

    @staticmethod
    def _weights(values, names):
        values = _to_list(values)

        if values is None:
            return {}

        if isinstance(values, Mapping):
            return {
                str(name): float(weight)
                for name, weight in values.items()
            }

        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise BehaviorInputError("Unsupported keypoint weights")

        if len(values) != len(names):
            raise BehaviorInputError(
                "Keypoint weight count does not match keypoint names"
            )

        return {
            name: float(weight)
            for name, weight in zip(names, values, strict=True)
        }

    def _person(self, item):
        bbox = self._bbox(_read(item, "bbox"))

        keypoints = self._keypoints(
            _read(item, "keypoints")
        )

        raw_values = _read(item, "raw_keypoints")

        raw_keypoints = (
            self._keypoints(raw_values)
            if raw_values is not None
            else None
        )

        weight_names = (
            tuple(keypoints)
            if keypoints
            else self.keypoint_names
        )

        weights = self._weights(
            _read(item, "keypoint_weights"),
            weight_names,
        )

        confidence = _read(
            item,
            "confidence",
            _read(item, "person_confidence"),
        )

        if confidence is None:
            raise BehaviorInputError(
                "Person detection confidence is missing"
            )

        return BehaviorPerson(
            track_id=_read(item, "track_id"),
            bbox=bbox,
            confidence=float(confidence),
            keypoints=keypoints,
            raw_keypoints=raw_keypoints,
            keypoint_weights=weights,
            observed=_read(item, "observed"),
            predicted=_read(item, "predicted"),
            behavior_eligible=bool(
                _read(item, "behavior_eligible", False)
            ),
            track_lifecycle=_read(item, "track_lifecycle"),
            missed=int(_read(item, "missed", 0)),
        )

    def _object(self, item):
        return BehaviorObject(
            object_id=_read(item, "object_id"),
            class_id=int(_read(item, "class_id")),
            class_name=str(_read(item, "class_name")),
            bbox=self._bbox(_read(item, "bbox")),
            confidence=float(_read(item, "confidence")),
        )

    def from_frame_result(
        self,
        frame: FrameResult,
        *,
        session_id: str,
        monotonic_timestamp: float | None = None,
        track_metadata: Mapping[int, Mapping[str, Any]] | None = None,
        track_lifecycle_events: Sequence[
            TrackLifecycleSignal | Mapping
        ] = (),
        object_detection_status: str | None = None,
    ) -> BehaviorFrame:
        if not isinstance(frame, FrameResult):
            raise TypeError("frame must be a FrameResult")

        metadata = dict(track_metadata or {})

        known_ids = {
            person.track_id
            for person in frame.persons
            if person.track_id is not None
        }

        unknown_ids = set(metadata) - known_ids

        if unknown_ids:
            raise BehaviorInputError(
                f"Metadata references unknown tracks: {sorted(unknown_ids)}"
            )

        people = []

        for person in frame.persons:
            payload = person.model_dump()

            extra = metadata.get(person.track_id, {})

            if not isinstance(extra, Mapping):
                raise BehaviorInputError(
                    "Track metadata must be a mapping"
                )

            unknown_fields = set(extra) - _ALLOWED_TRACK_METADATA

            if unknown_fields:
                raise BehaviorInputError(
                    f"Unsupported track metadata: {sorted(unknown_fields)}"
                )

            payload.update(extra)
            people.append(self._person(payload))

        objects = [
            self._object(item)
            for item in frame.objects
        ]

        if object_detection_status is None:
            object_detection_status = (
                "available" if objects else "not_run"
            )

        return BehaviorFrame(
            camera_id=frame.camera_id,
            session_id=session_id,
            frame_id=frame.frame_id,
            timestamp_utc=frame.timestamp,
            monotonic_timestamp=monotonic_timestamp,
            width=frame.width,
            height=frame.height,
            persons=people,
            objects=objects,
            object_detection_status=object_detection_status,
            track_lifecycle_events=track_lifecycle_events,
            source="frame_result",
            source_status=frame.status,
        )

    def from_tracking_output(
        self,
        *,
        camera_id: str,
        session_id: str,
        frame_id: int,
        timestamp_utc: datetime,
        width: int,
        height: int,
        persons: Sequence[Mapping[str, Any]],
        monotonic_timestamp: float | None = None,
        objects: Sequence[Mapping[str, Any]] | None = None,
        object_detection_status: str = "not_run",
        track_lifecycle_events: Sequence[
            TrackLifecycleSignal | Mapping
        ] = (),
        source_status: str = "ok",
    ) -> BehaviorFrame:
        people = [
            self._person(item)
            for item in persons
        ]

        object_items = [
            self._object(item)
            for item in (objects or [])
        ]

        return BehaviorFrame(
            camera_id=camera_id,
            session_id=session_id,
            frame_id=frame_id,
            timestamp_utc=timestamp_utc,
            monotonic_timestamp=monotonic_timestamp,
            width=width,
            height=height,
            persons=people,
            objects=object_items,
            object_detection_status=object_detection_status,
            track_lifecycle_events=track_lifecycle_events,
            source="tracking_output",
            source_status=source_status,
        )
