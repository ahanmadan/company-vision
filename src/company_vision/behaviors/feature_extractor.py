from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from math import atan2, degrees, hypot, isfinite

from company_vision.behaviors.schemas import (
    BehaviorFrame,
    BehaviorKeypoint,
    BehaviorObject,
    BehaviorPerson,
)


class FeatureExtractionError(ValueError):
    """Raised when a behavior frame cannot be safely processed."""


@dataclass(frozen=True, slots=True)
class PointFeature:
    name: str
    available: bool
    x_norm: float | None
    y_norm: float | None
    confidence: float
    weight: float | None
    quality: float
    reason: str


@dataclass(frozen=True, slots=True)
class VectorFeature:
    dx: float
    dy: float
    length: float
    angle_deg: float


@dataclass(frozen=True, slots=True)
class ObjectFeatures:
    object_id: str | int | None
    class_id: int
    class_name: str
    confidence: float
    center_x_norm: float
    center_y_norm: float
    width_norm: float
    height_norm: float


@dataclass(frozen=True, slots=True)
class PersonFeatures:
    camera_id: str
    session_id: str
    frame_id: int
    track_id: int | None

    behavior_eligible: bool
    observed: bool | None
    predicted: bool | None

    bbox_center_x_norm: float
    bbox_center_y_norm: float
    bbox_width_norm: float
    bbox_height_norm: float

    keypoints: tuple[PointFeature, ...]

    head_center: tuple[float, float] | None
    shoulder_center: tuple[float, float] | None

    shoulder_width_norm: float | None
    shoulder_line: VectorFeature | None

    head_to_shoulder: VectorFeature | None

    left_arm: VectorFeature | None
    right_arm: VectorFeature | None

    total_keypoints: int
    available_keypoints: int
    unavailable_keypoints: int

    mean_keypoint_quality: float


@dataclass(frozen=True, slots=True)
class FrameFeatures:
    schema_version: str

    camera_id: str
    session_id: str
    frame_id: int

    timestamp_utc: datetime
    monotonic_timestamp: float | None

    width: int
    height: int

    persons: tuple[PersonFeatures, ...]
    objects: tuple[ObjectFeatures, ...]

    object_detection_status: str
    source: str
    source_status: str


class FeatureExtractor:
    """
    Stateless geometry and quality feature extraction.

    Input:
        BehaviorFrame

    Output:
        FrameFeatures

    Responsibilities:
        - Normalize coordinates.
        - Extract body geometry.
        - Calculate geometric vectors.
        - Calculate keypoint quality.
        - Extract generic object geometry.

    Non-responsibilities:
        - Behavior classification.
        - Temporal history.
        - Tracking.
        - Object-person association.
        - Rule evaluation.
        - Event generation.
    """

    _HEAD_POINTS = (
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
    )

    def extract(self, frame: BehaviorFrame) -> FrameFeatures:

        if not isinstance(frame, BehaviorFrame):
            raise TypeError("frame must be a BehaviorFrame")

        persons = tuple(
            self._extract_person(frame, person)
            for person in frame.persons
        )

        objects = tuple(
            self._extract_object(frame, item)
            for item in frame.objects
        )

        return FrameFeatures(
            schema_version=frame.schema_version,
            camera_id=frame.camera_id,
            session_id=frame.session_id,
            frame_id=frame.frame_id,
            timestamp_utc=frame.timestamp_utc,
            monotonic_timestamp=frame.monotonic_timestamp,
            width=frame.width,
            height=frame.height,
            persons=persons,
            objects=objects,
            object_detection_status=frame.object_detection_status,
            source=frame.source,
            source_status=frame.source_status,
        )

    @staticmethod
    def _point_feature(
        name: str,
        point: BehaviorKeypoint,
        weight: float | None,
        has_weight_map: bool,
        bbox_width: float,
        bbox_height: float,
        bbox_x1: float,
        bbox_y1: float,
    ) -> PointFeature:

        confidence = float(point.confidence)

        if has_weight_map:
            effective_weight = (
                float(weight) if weight is not None else 0.0
            )
            quality = effective_weight
        else:
            effective_weight = None
            quality = confidence

        x_norm = None
        y_norm = None

        if point.x is not None and point.y is not None:
            x_norm = (float(point.x) - bbox_x1) / bbox_width
            y_norm = (float(point.y) - bbox_y1) / bbox_height

        reason = "available"
        available = True

        if not point.valid:
            available = False
            reason = "invalid_keypoint"

        elif x_norm is None or y_norm is None:
            available = False
            reason = "missing_coordinates"

        elif quality <= 0:
            available = False
            reason = "zero_quality"

        if not isfinite(quality):
            raise FeatureExtractionError(
                f"Non-finite quality for keypoint {name}"
            )

        return PointFeature(
            name=name,
            available=available,
            x_norm=x_norm,
            y_norm=y_norm,
            confidence=confidence,
            weight=effective_weight,
            quality=quality,
            reason=reason,
        )

    @staticmethod
    def _weighted_center(
        points: Mapping[str, PointFeature],
        names: tuple[str, ...],
    ) -> tuple[float, float] | None:

        selected = [
            points[name]
            for name in names
            if name in points and points[name].available
        ]

        if not selected:
            return None

        total_weight = sum(point.quality for point in selected)

        if total_weight <= 0:
            return None

        x = sum(
            point.x_norm * point.quality
            for point in selected
        ) / total_weight

        y = sum(
            point.y_norm * point.quality
            for point in selected
        ) / total_weight

        return (x, y)

    @staticmethod
    def _vector(
        start: tuple[float, float] | None,
        end: tuple[float, float] | None,
    ) -> VectorFeature | None:

        if start is None or end is None:
            return None

        dx = end[0] - start[0]
        dy = end[1] - start[1]

        return VectorFeature(
            dx=dx,
            dy=dy,
            length=hypot(dx, dy),
            angle_deg=degrees(atan2(dy, dx)),
        )

    def _extract_person(
        self,
        frame: BehaviorFrame,
        person: BehaviorPerson,
    ) -> PersonFeatures:

        box = person.bbox

        bbox_width = float(box.x2 - box.x1)
        bbox_height = float(box.y2 - box.y1)

        if bbox_width <= 0 or bbox_height <= 0:
            raise FeatureExtractionError(
                "Person bounding box must have positive dimensions"
            )

        has_weight_map = bool(person.keypoint_weights)

        points = {}

        for name, point in person.keypoints.items():

            feature = self._point_feature(
                name=name,
                point=point,
                weight=person.keypoint_weights.get(name),
                has_weight_map=has_weight_map,
                bbox_width=bbox_width,
                bbox_height=bbox_height,
                bbox_x1=float(box.x1),
                bbox_y1=float(box.y1),
            )

            points[name] = feature

        head_center = self._weighted_center(
            points,
            self._HEAD_POINTS,
        )

        shoulder_center = self._weighted_center(
            points,
            ("left_shoulder", "right_shoulder"),
        )

        left_shoulder = points.get("left_shoulder")
        right_shoulder = points.get("right_shoulder")

        left_elbow = points.get("left_elbow")
        right_elbow = points.get("right_elbow")

        shoulder_line = self._vector(
            (
                (left_shoulder.x_norm, left_shoulder.y_norm)
                if left_shoulder and left_shoulder.available
                else None
            ),
            (
                (right_shoulder.x_norm, right_shoulder.y_norm)
                if right_shoulder and right_shoulder.available
                else None
            ),
        )

        shoulder_width = (
            shoulder_line.length
            if shoulder_line is not None
            else None
        )

        left_arm = self._vector(
            (
                (left_shoulder.x_norm, left_shoulder.y_norm)
                if left_shoulder and left_shoulder.available
                else None
            ),
            (
                (left_elbow.x_norm, left_elbow.y_norm)
                if left_elbow and left_elbow.available
                else None
            ),
        )

        right_arm = self._vector(
            (
                (right_shoulder.x_norm, right_shoulder.y_norm)
                if right_shoulder and right_shoulder.available
                else None
            ),
            (
                (right_elbow.x_norm, right_elbow.y_norm)
                if right_elbow and right_elbow.available
                else None
            ),
        )

        head_to_shoulder = self._vector(
            shoulder_center,
            head_center,
        )

        available_count = sum(
            int(point.available)
            for point in points.values()
        )

        total_count = len(points)
        unavailable_count = total_count - available_count

        mean_quality = (
            sum(point.quality for point in points.values())
            / total_count
            if total_count
            else 0.0
        )

        return PersonFeatures(
            camera_id=frame.camera_id,
            session_id=frame.session_id,
            frame_id=frame.frame_id,
            track_id=person.track_id,
            behavior_eligible=person.behavior_eligible,
            observed=person.observed,
            predicted=person.predicted,
            bbox_center_x_norm=(
                (float(box.x1) + float(box.x2))
                / 2 / frame.width
            ),
            bbox_center_y_norm=(
                (float(box.y1) + float(box.y2))
                / 2 / frame.height
            ),
            bbox_width_norm=bbox_width / frame.width,
            bbox_height_norm=bbox_height / frame.height,
            keypoints=tuple(points.values()),
            head_center=head_center,
            shoulder_center=shoulder_center,
            shoulder_width_norm=shoulder_width,
            shoulder_line=shoulder_line,
            head_to_shoulder=head_to_shoulder,
            left_arm=left_arm,
            right_arm=right_arm,
            total_keypoints=total_count,
            available_keypoints=available_count,
            unavailable_keypoints=unavailable_count,
            mean_keypoint_quality=mean_quality,
        )

    @staticmethod
    def _extract_object(
        frame: BehaviorFrame,
        item: BehaviorObject,
    ) -> ObjectFeatures:

        box = item.bbox

        box_width = float(box.x2 - box.x1)
        box_height = float(box.y2 - box.y1)

        if box_width <= 0 or box_height <= 0:
            raise FeatureExtractionError(
                "Object bounding box must have positive dimensions"
            )

        return ObjectFeatures(
            object_id=item.object_id,
            class_id=item.class_id,
            class_name=item.class_name,
            confidence=float(item.confidence),
            center_x_norm=(
                (float(box.x1) + float(box.x2))
                / 2 / frame.width
            ),
            center_y_norm=(
                (float(box.y1) + float(box.y2))
                / 2 / frame.height
            ),
            width_norm=box_width / frame.width,
            height_norm=box_height / frame.height,
        )