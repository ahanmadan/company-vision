from datetime import datetime
from math import isfinite
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from company_vision.core.schemas import BoundingBox


class BehaviorStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BehaviorKeypoint(BehaviorStrictModel):
    x: float | None = Field(default=None, ge=0)
    y: float | None = Field(default=None, ge=0)
    confidence: float = Field(ge=0, le=1)
    valid: bool = False

    @field_validator("x", "y", "confidence")
    @classmethod
    def validate_finite(cls, value):
        if value is not None and not isfinite(value):
            raise ValueError("Keypoint values must be finite")
        return value

    @model_validator(mode="after")
    def validate_coordinates(self):
        if (self.x is None) != (self.y is None):
            raise ValueError("Keypoint coordinates must be both present or absent")

        if self.valid and (self.x is None or self.y is None):
            raise ValueError("A valid keypoint requires coordinates")

        return self


class BehaviorPerson(BehaviorStrictModel):
    track_id: int | None = Field(default=None, ge=0)
    bbox: BoundingBox
    confidence: float = Field(ge=0, le=1)

    keypoints: dict[str, BehaviorKeypoint] = Field(default_factory=dict)
    raw_keypoints: dict[str, BehaviorKeypoint] | None = None
    keypoint_weights: dict[str, float] = Field(default_factory=dict)

    observed: bool | None = None
    predicted: bool | None = None
    behavior_eligible: bool = False

    track_lifecycle: str | None = None
    missed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_person(self):
        if self.observed is True and self.predicted is True:
            raise ValueError("Person cannot be observed and predicted")

        if self.behavior_eligible:
            if self.track_id is None:
                raise ValueError("Eligible person requires track_id")

            if self.observed is not True or self.predicted is True:
                raise ValueError(
                    "Eligible person must be explicitly observed"
                )

        unknown_weights = (
            set(self.keypoint_weights) - set(self.keypoints)
        )

        if unknown_weights:
            raise ValueError(
                f"Weights reference unknown keypoints: {sorted(unknown_weights)}"
            )

        return self

    @field_validator("keypoint_weights")
    @classmethod
    def validate_weights(cls, values):
        for name, value in values.items():
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid weight for {name}")
        return values


class BehaviorObject(BehaviorStrictModel):
    object_id: str | int | None = None
    class_id: int = Field(ge=0)
    class_name: str = Field(min_length=1)
    bbox: BoundingBox
    confidence: float = Field(ge=0, le=1)


class TrackLifecycleSignal(BehaviorStrictModel):
    event_type: str = Field(min_length=1)
    track_id: int | None = Field(default=None, ge=0)
    monotonic_timestamp: float | None = Field(default=None, ge=0)
    reason: str | None = None

    @field_validator("monotonic_timestamp")
    @classmethod
    def validate_timestamp(cls, value):
        if value is not None and not isfinite(value):
            raise ValueError("Lifecycle timestamp must be finite")
        return value


class BehaviorFrame(BehaviorStrictModel):
    schema_version: Literal["1.0"] = "1.0"

    camera_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)

    frame_id: int = Field(ge=0)

    timestamp_utc: datetime
    monotonic_timestamp: float | None = Field(default=None, ge=0)

    width: int = Field(gt=0)
    height: int = Field(gt=0)

    persons: list[BehaviorPerson] = Field(default_factory=list)
    objects: list[BehaviorObject] = Field(default_factory=list)

    object_detection_status: Literal[
        "available",
        "not_run",
        "unavailable",
        "error",
    ] = "not_run"

    track_lifecycle_events: list[TrackLifecycleSignal] = Field(
        default_factory=list
    )

    source: Literal["frame_result", "tracking_output"]
    source_status: Literal["ok", "partial", "error"] = "ok"

    @field_validator("timestamp_utc")
    @classmethod
    def validate_utc(cls, value):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp_utc must be timezone-aware")

        if value.utcoffset().total_seconds() != 0:
            raise ValueError("timestamp_utc must be UTC")

        return value

    @field_validator("monotonic_timestamp")
    @classmethod
    def validate_monotonic(cls, value):
        if value is not None and not isfinite(value):
            raise ValueError("Monotonic timestamp must be finite")
        return value

    @model_validator(mode="after")
    def validate_frame(self):
        ids = [
            person.track_id
            for person in self.persons
            if person.track_id is not None
        ]

        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate person track IDs")

        for item in self.persons:
            box = item.bbox

            if box.x2 > self.width or box.y2 > self.height:
                raise ValueError("Person bbox exceeds frame dimensions")

            all_keypoints = list(item.keypoints.values())

            if item.raw_keypoints is not None:
                all_keypoints.extend(item.raw_keypoints.values())

            for point in all_keypoints:
                if not point.valid:
                    continue

                if point.x > self.width or point.y > self.height:
                    raise ValueError("Keypoint exceeds frame dimensions")

        for item in self.objects:
            if (
                item.bbox.x2 > self.width
                or item.bbox.y2 > self.height
            ):
                raise ValueError("Object bbox exceeds frame dimensions")

        if self.objects and self.object_detection_status != "available":
            raise ValueError(
                "Objects cannot be supplied unless detection is available"
            )

        return self
