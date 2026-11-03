
from dataclasses import dataclass
from math import isfinite

import cv2
import numpy as np

from company_vision.core.schemas import FramePacket, FrameResult


class OverlayRendererError(RuntimeError):
    """Raised when rendering a frame fails."""


# COCO 17-keypoint skeleton connections.
SKELETON = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
)


@dataclass(frozen=True)
class OverlayConfig:
    person_color: tuple[int, int, int] = (0, 220, 0)
    object_color: tuple[int, int, int] = (0, 165, 255)
    keypoint_color: tuple[int, int, int] = (255, 255, 0)
    skeleton_color: tuple[int, int, int] = (255, 180, 0)
    text_color: tuple[int, int, int] = (255, 255, 255)
    background_color: tuple[int, int, int] = (30, 30, 30)

    thickness: int = 2
    keypoint_radius: int = 4
    font_scale: float = 0.5
    confidence_threshold: float = 0.25

    show_person_boxes: bool = True
    show_track_ids: bool = True
    show_keypoints: bool = True
    show_skeleton: bool = True
    show_objects: bool = True
    show_confidence: bool = True


def _field(value: object, name: str, default=None):
    if isinstance(value, dict):
        return value.get(name, default)

    return getattr(value, name, default)


def _box_coordinates(box: object) -> tuple[int, int, int, int]:
    names = ("x1", "y1", "x2", "y2")

    values = tuple(_field(box, name) for name in names)

    if any(value is None for value in values):
        raise OverlayRendererError(
            "BoundingBox must contain x1, y1, x2 and y2"
        )

    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
        for value in values
    ):
        raise OverlayRendererError("Invalid bounding box coordinates")

    x1, y1, x2, y2 = values

    if x2 <= x1 or y2 <= y1:
        raise OverlayRendererError("Invalid bounding box dimensions")

    return (
        round(x1),
        round(y1),
        round(x2),
        round(y2),
    )


class OverlayRenderer:
    """
    Render FrameResult observations onto the original BGR frame.

    The renderer never runs inference and never modifies the input frame.
    """

    def __init__(
        self,
        config: OverlayConfig | None = None,
    ) -> None:
        self.config = config or OverlayConfig()

        if self.config.thickness <= 0:
            raise ValueError("thickness must be positive")

        if self.config.keypoint_radius <= 0:
            raise ValueError("keypoint_radius must be positive")

        if (
            not isfinite(self.config.confidence_threshold)
            or not 0 <= self.config.confidence_threshold <= 1
        ):
            raise ValueError(
                "confidence_threshold must be between 0 and 1"
            )

        if (
            not isfinite(self.config.font_scale)
            or self.config.font_scale <= 0
        ):
            raise ValueError("font_scale must be positive")

    @staticmethod
    def _validate(
        packet: FramePacket,
        result: FrameResult,
    ) -> np.ndarray:
        if not isinstance(packet, FramePacket):
            raise TypeError("packet must be a FramePacket")

        if not isinstance(result, FrameResult):
            raise TypeError("result must be a FrameResult")

        if packet.camera_id != result.camera_id:
            raise OverlayRendererError("Camera ID mismatch")

        if packet.frame_id != result.frame_id:
            raise OverlayRendererError("Frame ID mismatch")

        if packet.timestamp != result.timestamp:
            raise OverlayRendererError("Timestamp mismatch")

        image = packet.image

        if not isinstance(image, np.ndarray):
            raise OverlayRendererError("Frame image must be ndarray")

        if (
            image.ndim != 3
            or image.shape[2] != 3
            or image.dtype != np.uint8
        ):
            raise OverlayRendererError(
                "Image must be a uint8 BGR image"
            )

        if image.shape[1] != result.width:
            raise OverlayRendererError("Image width mismatch")

        if image.shape[0] != result.height:
            raise OverlayRendererError("Image height mismatch")

        return image

    @staticmethod
    def _clip_point(
        x: float,
        y: float,
        width: int,
        height: int,
    ) -> tuple[int, int] | None:
        if not isfinite(x) or not isfinite(y):
            return None

        if x < 0 or y < 0 or x >= width or y >= height:
            return None

        return round(x), round(y)

    def _draw_label(
        self,
        image: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
    ) -> None:
        config = self.config

        font = cv2.FONT_HERSHEY_SIMPLEX

        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            font,
            config.font_scale,
            config.thickness,
        )

        image_height, image_width = image.shape[:2]

        x = max(0, min(x, image_width - text_width - 4))
        y = max(text_height + 6, min(y, image_height - baseline - 2))

        top = y - text_height - 6
        bottom = y + baseline

        cv2.rectangle(
            image,
            (x, top),
            (x + text_width + 4, bottom),
            config.background_color,
            cv2.FILLED,
        )

        cv2.putText(
            image,
            text,
            (x + 2, y - 2),
            font,
            config.font_scale,
            config.text_color,
            config.thickness,
            cv2.LINE_AA,
        )

    def _draw_person(
        self,
        image: np.ndarray,
        person: object,
    ) -> None:
        config = self.config
        height, width = image.shape[:2]

        bbox = _box_coordinates(_field(person, "bbox"))

        x1, y1, x2, y2 = bbox

        if config.show_person_boxes:
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                config.person_color,
                config.thickness,
            )

        track_id = _field(person, "track_id")

        label = (
            f"Person ID: {track_id}"
            if track_id is not None
            else "Person: Untracked"
        )

        if config.show_confidence:
            confidence = _field(person, "confidence")

            if confidence is not None:
                label += f" {confidence:.2f}"

        if config.show_track_ids or track_id is None:
            self._draw_label(
                image,
                label,
                x1,
                y1,
                config.person_color,
            )

        keypoints = _field(person, "keypoints", [])

        points: dict[str, tuple[int, int]] = {}

        for keypoint in keypoints:
            name = _field(keypoint, "name")
            x = _field(keypoint, "x")
            y = _field(keypoint, "y")
            confidence = _field(keypoint, "confidence", 0.0)

            if (
                not isinstance(name, str)
                or not isinstance(x, (int, float))
                or not isinstance(y, (int, float))
                or not isinstance(confidence, (int, float))
                or not isfinite(x)
                or not isfinite(y)
                or not isfinite(confidence)
                or not 0 <= confidence <= 1
            ):
                continue

            point = self._clip_point(x, y, width, height)

            if point is not None and confidence >= config.confidence_threshold:
                points[name] = point

        if config.show_skeleton:
            for first, second in SKELETON:
                if first not in points or second not in points:
                    continue

                cv2.line(
                    image,
                    points[first],
                    points[second],
                    config.skeleton_color,
                    config.thickness,
                    cv2.LINE_AA,
                )

        if config.show_keypoints:
            for point in points.values():
                cv2.circle(
                    image,
                    point,
                    config.keypoint_radius,
                    config.keypoint_color,
                    cv2.FILLED,
                    cv2.LINE_AA,
                )

    def _draw_object(
        self,
        image: np.ndarray,
        obj: object,
    ) -> None:
        config = self.config

        x1, y1, x2, y2 = _box_coordinates(
            _field(obj, "bbox")
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            config.object_color,
            config.thickness,
        )

        class_name = _field(obj, "class_name", "object")
        confidence = _field(obj, "confidence")

        label = str(class_name)

        if config.show_confidence and confidence is not None:
            label += f" {confidence:.2f}"

        self._draw_label(
            image,
            label,
            x1,
            y1,
            config.object_color,
        )

    def render(
        self,
        packet: FramePacket,
        result: FrameResult,
    ) -> np.ndarray:
        """
        Return a new annotated BGR image.

        The original packet.image remains unchanged.
        """
        original = self._validate(packet, result)

        output = original.copy()

        if self.config.show_objects:
            for obj in result.objects:
                self._draw_object(output, obj)

        for person in result.persons:
            self._draw_person(output, person)

        return output
