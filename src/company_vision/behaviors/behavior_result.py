"""Canonical, database-independent behavior output contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

BehaviorStatus = Literal[
    "no_event",
    "active",
    "confirmed",
    "ended",
    "unknown",
]


def serialize_component(value: Any) -> Any:
    """Convert supported pipeline objects into JSON-compatible data."""

    if isinstance(value, Enum):
        return serialize_component(value.value)

    if isinstance(value, UUID):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")

    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: serialize_component(
                getattr(value, item.name)
            )
            for item in fields(value)
        }

    if isinstance(value, Mapping):
        return {
            str(key): serialize_component(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [serialize_component(item) for item in value]

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    raise TypeError(
        f"Unsupported pipeline component: {type(value).__name__}"
    )


def _serialize_records(
    records: list[Any] | tuple[Any, ...],
    field_name: str,
) -> list[dict[str, Any]]:
    if not isinstance(records, (list, tuple)):
        raise TypeError(f"{field_name} must be a list or tuple.")

    output = []

    for record in records:
        item = serialize_component(record)

        if not isinstance(item, dict):
            raise TypeError(
                f"Each {field_name} entry must serialize to a dictionary."
            )

        output.append(item)

    return output


class BehaviorResult(BaseModel):
    """Versioned final behavior output, independent of storage and UI."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal["1.0"] = "1.0"

    result_id: UUID = Field(default_factory=uuid4)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    camera_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)

    frame_id: int = Field(ge=0)

    track_id: int | str

    status: BehaviorStatus

    events: list[dict[str, Any]] = Field(default_factory=list)

    rule_evaluations: list[dict[str, Any]] = Field(
        default_factory=list
    )

    evidence: dict[str, Any] = Field(default_factory=dict)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("camera_id", "session_id")
    @classmethod
    def validate_identifiers(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Identifier cannot be empty.")

        return value

    @field_validator("created_at")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware.")

        return value

    def to_payload(self) -> dict[str, Any]:
        """Return JSON-compatible representation."""

        return self.model_dump(mode="json")


def create_behavior_result(
    *,
    camera_id: str,
    session_id: str,
    frame_id: int,
    track_id: int | str,
    status: BehaviorStatus,
    events: list[Any] | tuple[Any, ...] = (),
    rule_evaluations: list[Any] | tuple[Any, ...] = (),
    evidence: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    created_at: datetime | None = None,
) -> BehaviorResult:
    """Build a validated result from pipeline components.

    Event lifecycle is preserved as supplied.
    Status is explicit and is never inferred from a rule alone.
    """

    serialized_evidence = serialize_component(evidence or {})
    serialized_metadata = serialize_component(metadata or {})

    if not isinstance(serialized_evidence, dict):
        raise TypeError("Evidence must be a mapping.")

    if not isinstance(serialized_metadata, dict):
        raise TypeError("Metadata must be a mapping.")

    values: dict[str, Any] = {
        "camera_id": camera_id,
        "session_id": session_id,
        "frame_id": frame_id,
        "track_id": track_id,
        "status": status,
        "events": _serialize_records(events, "events"),
        "rule_evaluations": _serialize_records(
            rule_evaluations,
            "rule_evaluations",
        ),
        "evidence": serialized_evidence,
        "metadata": serialized_metadata,
    }

    if created_at is not None:
        values["created_at"] = created_at

    return BehaviorResult.model_validate(values)