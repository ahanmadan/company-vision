
from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from ultralytics import YOLO

from company_vision.core.schemas import FramePacket

COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


class PoseEstimationError(RuntimeError):
    """Invalid model output or pose inference failure."""


@dataclass(frozen=True)
class PoseKeypoint:
    name: str
    x: float
    y: float
    confidence: float | None


@dataclass(frozen=True)
class PosePerson:
    bbox: tuple[float, float, float, float]
    confidence: float
    keypoints: tuple[PoseKeypoint, ...]


@dataclass(frozen=True)
class PoseFrameResult:
    packet: FramePacket
    people: tuple[PosePerson, ...]
    inference_ms: float

    @property
    def person_count(self) -> int:
        return len(self.people)


def to_numpy(value: Any) -> np.ndarray:
    """Convert an Ultralytics tensor or ndarray to NumPy."""
    if hasattr(value, "detach"):
        value = value.detach()

    if hasattr(value, "cpu"):
        value = value.cpu()

    return np.asarray(value)


class PoseEstimator:
    """Convert YOLO26 Pose output into typed per-frame results."""

    def __init__(
        self,
        model_path: str | Path = "yolo26x-pose.pt",
        *,
        device: str = "cuda:0",
        confidence: float = 0.35,
        image_size: int = 640,
        model_factory: Callable[[str], Any] = YOLO,
    ) -> None:
        if not str(model_path).strip():
            raise ValueError("model_path cannot be empty")

        if not isinstance(device, str) or not device.strip():
            raise ValueError("device cannot be empty")

        if not isinstance(confidence, (int, float)) or isinstance(
            confidence, bool
        ) or not 0 < confidence <= 1:
            raise ValueError("confidence must be in (0, 1]")

        if (
            not isinstance(image_size, int)
            or isinstance(image_size, bool)
            or image_size <= 0
            or image_size % 32
        ):
            raise ValueError("image_size must be a positive multiple of 32")

        self.device = device
        self.confidence = float(confidence)
        self.image_size = image_size
        self.model = model_factory(str(model_path))

        if getattr(self.model, "task", None) != "pose":
            raise PoseEstimationError("Expected a YOLO pose model")

    def predict(self, packet: FramePacket) -> PoseFrameResult:
        if not isinstance(packet, FramePacket):
            raise TypeError("packet must be a FramePacket")

        image = packet.image

        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or not image.shape[0]
            or not image.shape[1]
        ):
            raise PoseEstimationError("Invalid BGR frame")

        image = np.ascontiguousarray(image)

        start = perf_counter()

        try:
            results = self.model.predict(
                source=image,
                device=self.device,
                conf=self.confidence,
                imgsz=self.image_size,
                verbose=False,
                save=False,
            )
        except (RuntimeError, ValueError, OSError) as error:
            raise PoseEstimationError(
                "Pose model inference failed"
            ) from error

        elapsed_ms = (perf_counter() - start) * 1000.0

        if len(results) != 1:
            raise PoseEstimationError(
                "Expected exactly one image result"
            )

        people = self._parse_result(
            results[0],
            width=packet.width,
            height=packet.height,
        )

        return PoseFrameResult(
            packet=packet,
            people=tuple(people),
            inference_ms=elapsed_ms,
        )

    @staticmethod
    def _parse_result(
        result: Any,
        *,
        width: int,
        height: int,
    ) -> list[PosePerson]:
        boxes = getattr(result, "boxes", None)
        keypoints = getattr(result, "keypoints", None)

        if boxes is None:
            raise PoseEstimationError("Missing pose boxes")

        xyxy = to_numpy(boxes.xyxy)
        scores = to_numpy(boxes.conf)

        if xyxy.ndim != 2 or xyxy.shape[1] != 4:
            raise PoseEstimationError("Invalid box dimensions")

        if scores.shape != (len(xyxy),):
            raise PoseEstimationError("Box confidence mismatch")

        if len(xyxy) == 0:
            return []

        if keypoints is None:
            raise PoseEstimationError("Missing pose keypoints")

        points = to_numpy(keypoints.data)

        if points.ndim != 3 or points.shape[0] != len(xyxy):
            raise PoseEstimationError("Keypoint count mismatch")

        if points.shape[1] != 17 or points.shape[2] not in (2, 3):
            raise PoseEstimationError(
                "Expected 17 COCO keypoints"
            )

        people: list[PosePerson] = []

        for index, (coordinates, score) in enumerate(
            zip(xyxy, scores, strict=True)
        ):
            score_value = float(score)

            if not isfinite(score_value) or not 0 <= score_value <= 1:
                raise PoseEstimationError("Invalid person confidence")

            if not np.isfinite(coordinates).all():
                raise PoseEstimationError("Non-finite bounding box")

            x1, y1, x2, y2 = [
                float(value) for value in coordinates
            ]

            x1 = min(max(x1, 0.0), float(width))
            x2 = min(max(x2, 0.0), float(width))
            y1 = min(max(y1, 0.0), float(height))
            y2 = min(max(y2, 0.0), float(height))

            if x2 <= x1 or y2 <= y1:
                raise PoseEstimationError("Degenerate bounding box")

            person_points: list[PoseKeypoint] = []

            for name, raw_point in zip(
                COCO_KEYPOINT_NAMES,
                points[index],
                strict=True,
            ):
                if not np.isfinite(raw_point).all():
                    raise PoseEstimationError(
                        "Non-finite keypoint"
                    )

                confidence = (
                    float(raw_point[2])
                    if len(raw_point) == 3
                    else None
                )

                if confidence is not None and not 0 <= confidence <= 1:
                    raise PoseEstimationError(
                        "Invalid keypoint confidence"
                    )

                person_points.append(
                    PoseKeypoint(
                        name=name,
                        x=float(raw_point[0]),
                        y=float(raw_point[1]),
                        confidence=confidence,
                    )
                )

            people.append(
                PosePerson(
                    bbox=(x1, y1, x2, y2),
                    confidence=score_value,
                    keypoints=tuple(person_points),
                )
            )

        return people
