
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BoundingBox(StrictModel):
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    x2: float = Field(ge=0)
    y2: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_coordinates(self):
        if self.x2 <= self.x1:
            raise ValueError("x2 must be greater than x1")
        if self.y2 <= self.y1:
            raise ValueError("y2 must be greater than y1")
        return self

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


class Keypoint(StrictModel):
    name: str = Field(min_length=1)
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)


class PersonObservation(StrictModel):
    track_id: int | None = Field(default=None, ge=0)
    bbox: BoundingBox
    confidence: float = Field(ge=0, le=1)
    keypoints: list[Keypoint] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_keypoints(self):
        names = [point.name for point in self.keypoints]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate keypoint names")
        return self


class ObjectObservation(StrictModel):
    class_id: int = Field(ge=0)
    class_name: str = Field(min_length=1)
    bbox: BoundingBox
    confidence: float = Field(ge=0, le=1)


def validate_utc_timestamp(value: datetime) -> None:
    if value.tzinfo is None:
        raise ValueError("Timestamp must be timezone-aware")

    if value.utcoffset().total_seconds() != 0:
        raise ValueError("Timestamp must be in UTC")


@dataclass(frozen=True)
class FramePacket:
    camera_id: str
    frame_id: int
    timestamp: datetime
    image: np.ndarray

    def __post_init__(self):
        if not self.camera_id.strip():
            raise ValueError("camera_id cannot be empty")

        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative")

        validate_utc_timestamp(self.timestamp)

        if not isinstance(self.image, np.ndarray):
            raise TypeError("image must be a NumPy array")

        if self.image.dtype != np.uint8:
            raise ValueError("image dtype must be uint8")

        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError(
                "image must have shape (height, width, 3)"
            )

        if self.image.shape[0] == 0 or self.image.shape[1] == 0:
            raise ValueError("image dimensions cannot be zero")

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


class FrameResult(StrictModel):
    camera_id: str = Field(min_length=1)
    frame_id: int = Field(ge=0)
    timestamp: datetime
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    persons: list[PersonObservation] = Field(default_factory=list)
    objects: list[ObjectObservation] = Field(default_factory=list)
    status: Literal["ok", "partial", "error"] = "ok"

    @model_validator(mode="after")
    def validate_frame(self):
        validate_utc_timestamp(self.timestamp)

        track_ids = [
            person.track_id
            for person in self.persons
            if person.track_id is not None
        ]

        if len(track_ids) != len(set(track_ids)):
            raise ValueError("Duplicate person track IDs")

        boxes = (
            [person.bbox for person in self.persons]
            + [obj.bbox for obj in self.objects]
        )

        for box in boxes:
            if box.x2 > self.width or box.y2 > self.height:
                raise ValueError(
                    "BoundingBox exceeds image dimensions"
                )

        for person in self.persons:
            for point in person.keypoints:
                if point.x > self.width or point.y > self.height:
                    raise ValueError(
                        "Keypoint exceeds image dimensions"
                    )

        return self

    @classmethod
    def from_packet(
        cls,
        packet: FramePacket,
        persons: list[PersonObservation] | None = None,
        objects: list[ObjectObservation] | None = None,
        status: Literal["ok", "partial", "error"] = "ok",
    ) -> "FrameResult":
        return cls(
            camera_id=packet.camera_id,
            frame_id=packet.frame_id,
            timestamp=packet.timestamp,
            width=packet.width,
            height=packet.height,
            persons=persons if persons is not None else [],
            objects=objects if objects is not None else [],
            status=status,
        )


def current_utc() -> datetime:
    return datetime.now(UTC)
