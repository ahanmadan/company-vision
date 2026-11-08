
from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from ultralytics import YOLO

from company_vision.core.schemas import FramePacket


class PersonTrackingError(RuntimeError):
    """Invalid tracking input, output or execution."""


@dataclass(frozen=True)
class TrackedPerson:
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True)
class TrackingFrameResult:
    packet: FramePacket
    persons: tuple[TrackedPerson, ...]
    inference_ms: float

    @property
    def person_count(self) -> int:
        return len(self.persons)

    @property
    def track_ids(self) -> tuple[int, ...]:
        return tuple(person.track_id for person in self.persons)


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()

    if hasattr(value, "cpu"):
        value = value.cpu()

    return np.asarray(value)


class PersonTracker:
    """
    Track people across consecutive frames using YOLO + ByteTrack.

    One instance belongs to one camera stream.
    Track IDs are temporary and are not employee identities.
    """

    def __init__(
        self,
        model_path: str | Path = "yolo26x.pt",
        *,
        device: str = "cuda:0",
        tracker_config: str = "bytetrack.yaml",
        confidence: float = 0.1,
        image_size: int = 640,
        model_factory: Callable[[str], Any] = YOLO,
    ) -> None:
        if not str(model_path).strip():
            raise ValueError("model_path cannot be empty")

        if not isinstance(device, str) or not device.strip():
            raise ValueError("device cannot be empty")

        if not isinstance(tracker_config, str) or not tracker_config.strip():
            raise ValueError("tracker_config cannot be empty")

        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not isfinite(confidence)
            or not 0 < confidence <= 1
        ):
            raise ValueError("confidence must be in (0, 1]")

        if (
            isinstance(image_size, bool)
            or not isinstance(image_size, int)
            or image_size <= 0
            or image_size % 32 != 0
        ):
            raise ValueError(
                "image_size must be a positive multiple of 32"
            )

        self.device = device
        self.tracker_config = tracker_config
        self.confidence = float(confidence)
        self.image_size = image_size

        self.model = model_factory(str(model_path))

        if getattr(self.model, "task", None) != "detect":
            raise PersonTrackingError(
                "Expected a YOLO detection model"
            )

        self._camera_id: str | None = None
        self._last_frame_id: int | None = None
        self._processed_frames = 0

    @property
    def processed_frames(self) -> int:
        return self._processed_frames

    def track(self, packet: FramePacket) -> TrackingFrameResult:
        if not isinstance(packet, FramePacket):
            raise TypeError("packet must be a FramePacket")

        
        if (
            self._camera_id is not None
            and packet.camera_id != self._camera_id
        ):
            raise PersonTrackingError(
                "One tracker cannot process multiple cameras"
            )

        if (
            self._last_frame_id is not None
            and packet.frame_id <= self._last_frame_id
        ):
            raise PersonTrackingError(
                "Frame IDs must increase strictly"
            )


        image = packet.image

        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or image.shape[0] == 0
            or image.shape[1] == 0
        ):
            raise PersonTrackingError("Invalid BGR frame")

        image = np.ascontiguousarray(image)

        start = perf_counter()

        try:
            results = self.model.track(
                source=image,
                persist=True,
                tracker=self.tracker_config,
                classes=[0],
                device=self.device,
                conf=self.confidence,
                imgsz=self.image_size,
                verbose=False,
                save=False,
            )
        except (RuntimeError, ValueError, OSError) as exc:
            raise PersonTrackingError(
                "Person tracking inference failed"
            ) from exc

        elapsed_ms = (perf_counter() - start) * 1000.0

        if len(results) != 1:
            raise PersonTrackingError(
                "Expected one tracking result"
            )

        persons = self._parse_result(
            results[0],
            width=packet.width,
            height=packet.height,
        )

        self._camera_id = packet.camera_id
        self._last_frame_id = packet.frame_id
        self._processed_frames += 1

        return TrackingFrameResult(
            packet=packet,
            persons=tuple(persons),
            inference_ms=elapsed_ms,
        )

    @staticmethod
    def _parse_result(
        result: Any,
        *,
        width: int,
        height: int,
    ) -> list[TrackedPerson]:
        boxes = getattr(result, "boxes", None)

        if boxes is None:
            raise PersonTrackingError(
                "Missing tracking boxes"
            )

        coordinates = to_numpy(boxes.xyxy)
        scores = to_numpy(boxes.conf)
        classes = to_numpy(boxes.cls)

        if coordinates.ndim != 2 or coordinates.shape[1] != 4:
            raise PersonTrackingError(
                "Invalid tracking boxes shape"
            )

        count = len(coordinates)

        if scores.shape != (count,) or classes.shape != (count,):
            raise PersonTrackingError(
                "Tracking arrays have inconsistent lengths"
            )

        if count == 0:
            return []

        raw_ids = getattr(boxes, "id", None)

        # A tracker can return detections before stable IDs exist.
        # They are not emitted as tracked persons.
        if raw_ids is None:
            return []

        ids = to_numpy(raw_ids)

        if ids.shape != (count,):
            raise PersonTrackingError(
                "Tracking ID count mismatch"
            )

        people: list[TrackedPerson] = []
        seen_ids: set[int] = set()

        for coords, raw_score, raw_class, raw_id in zip(
            coordinates,
            scores,
            classes,
            ids,
            strict=True,
        ):
            score = float(raw_score)
            class_value = float(raw_class)
            id_value = float(raw_id)

            if (
                not isfinite(score)
                or not 0 <= score <= 1
                or not isfinite(class_value)
                or class_value != 0
            ):
                raise PersonTrackingError(
                    "Invalid tracked person class or confidence"
                )

            if (
                not isfinite(id_value)
                or id_value < 0
                or not id_value.is_integer()
            ):
                raise PersonTrackingError("Invalid Track ID")

            track_id = int(id_value)

            if track_id in seen_ids:
                raise PersonTrackingError(
                    "Duplicate Track ID in one frame"
                )

            seen_ids.add(track_id)

            if not np.isfinite(coords).all():
                raise PersonTrackingError(
                    "Non-finite tracking box"
                )

            x1, y1, x2, y2 = (
                float(value) for value in coords
            )

            x1 = min(max(x1, 0.0), float(width))
            x2 = min(max(x2, 0.0), float(width))
            y1 = min(max(y1, 0.0), float(height))
            y2 = min(max(y2, 0.0), float(height))

            if x2 <= x1 or y2 <= y1:
                raise PersonTrackingError(
                    "Degenerate tracking box"
                )

            people.append(
                TrackedPerson(
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    confidence=score,
                )
            )

        return people
