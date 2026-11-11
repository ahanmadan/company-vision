from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from uuid import UUID

import pytest
from pydantic import ValidationError

from company_vision.behaviors.behavior_result import (
    BehaviorResult,
    create_behavior_result,
    serialize_component,
)


def make_result(**overrides):
    values = {
        "camera_id": "camera-1",
        "session_id": "session-1",
        "frame_id": 10,
        "track_id": 1,
        "status": "active",
    }
    values.update(overrides)
    return create_behavior_result(**values)


def test_basic_result_creation():
    result = make_result()

    assert isinstance(result, BehaviorResult)
    assert result.schema_version == "1.0"
    assert result.camera_id == "camera-1"
    assert result.status == "active"


def test_result_id_is_uuid():
    result = make_result()

    assert isinstance(result.result_id, UUID)


def test_result_ids_are_unique():
    first = make_result()
    second = make_result()

    assert first.result_id != second.result_id


def test_default_timestamp_is_timezone_aware():
    result = make_result()

    assert result.created_at.tzinfo is not None


@pytest.mark.parametrize(
    "field,value",
    [
        ("camera_id", ""),
        ("session_id", ""),
    ],
)
def test_empty_identifiers_are_rejected(field, value):
    with pytest.raises(ValidationError):
        make_result(**{field: value})


def test_negative_frame_id_is_rejected():
    with pytest.raises(ValidationError):
        make_result(frame_id=-1)


def test_invalid_status_is_rejected():
    with pytest.raises(ValidationError):
        make_result(status="sleeping")


def test_naive_timestamp_is_rejected():
    with pytest.raises(ValidationError):
        make_result(created_at=datetime(2026, 1, 1))


def test_lifecycle_transitions_are_preserved():
    transitions = [
        {"transition": "started"},
        {"transition": "updated"},
        {"transition": "confirmed"},
        {"transition": "ended"},
    ]

    result = make_result(events=transitions)

    assert result.events == transitions
    assert [x["transition"] for x in result.events] == [
        "started",
        "updated",
        "confirmed",
        "ended",
    ]


def test_unknown_status_does_not_modify_event_history():
    history = [{"transition": "confirmed"}]

    result = make_result(
        status="unknown",
        events=history,
    )

    assert result.status == "unknown"
    assert result.events == history


def test_rule_evaluations_are_preserved():
    evaluations = [
        {"rule_id": "phone_usage", "matched": True},
        {"rule_id": "sleep", "matched": False},
    ]

    result = make_result(rule_evaluations=evaluations)

    assert result.rule_evaluations == evaluations


def test_evidence_and_metadata_are_preserved():
    result = make_result(
        evidence={"head_down": True},
        metadata={"source": "tracking_output"},
    )

    assert result.evidence["head_down"] is True
    assert result.metadata["source"] == "tracking_output"


def test_json_payload_is_serializable():
    result = make_result(
        events=[{"transition": "started"}]
    )

    payload = result.to_payload()

    assert payload["schema_version"] == "1.0"
    assert isinstance(payload["result_id"], str)
    assert isinstance(payload["created_at"], str)


def test_frozen_contract_rejects_assignment():
    result = make_result()

    with pytest.raises(ValidationError):
        result.status = "ended"


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        BehaviorResult(
            camera_id="camera-1",
            session_id="session-1",
            frame_id=1,
            track_id=1,
            status="active",
            unexpected_field=True,
        )


def test_dataclass_components_are_serialized():
    @dataclass
    class Event:
        transition: str
        confidence: float

    result = make_result(
        events=[Event("started", 0.95)]
    )

    assert result.events == [
        {"transition": "started", "confidence": 0.95}
    ]


def test_enum_components_are_serialized():
    class Transition(Enum):
        STARTED = "started"

    assert serialize_component(Transition.STARTED) == "started"


def test_datetime_components_are_serialized():
    value = datetime(2026, 11, 11, tzinfo=UTC)

    serialized = serialize_component(value)

    assert serialized == "2026-11-11T00:00:00+00:00"


def test_non_mapping_event_is_rejected():
    with pytest.raises(TypeError):
        make_result(events=["invalid-event"])


def test_independent_results_preserve_camera_and_session():
    first = make_result(
        camera_id="camera-A",
        session_id="session-A",
    )

    second = make_result(
        camera_id="camera-B",
        session_id="session-B",
    )

    assert first.camera_id != second.camera_id
    assert first.session_id != second.session_id