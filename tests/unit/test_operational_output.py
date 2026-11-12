"""Node 5.9.5 operational output tests."""

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from company_vision.behaviors.behavior_execution_pipeline import (
    BehaviorExecutionPipeline,
)
from company_vision.behaviors.behavior_result import BehaviorResult
from company_vision.behaviors.operational_output import (
    OperationalOutputFormatter,
    OperationalOutputPipeline,
)
from company_vision.behaviors.plugin_registry import PluginResult
from company_vision.behaviors.rule_event_pipeline import RuleEventPipeline

BASE = datetime(2026, 11, 12, tzinfo=UTC)


def tracked(
    track_id=1,
    x=40.0,
    observed=True,
    eligible=True,
):
    points = np.array([
        [x + 18, 20 + i * 5, 0.9]
        for i in range(9)
    ], dtype=np.float32)

    return {
        "track_id": track_id,
        "bbox": np.array([x, 10, x + 50, 130]),
        "keypoints": points,
        "raw_keypoints": points.copy() if observed else None,
        "keypoint_weights": np.ones(9) * 0.85,
        "confidence": 0.95,
        "observed": observed,
        "predicted": not observed,
        "behavior_eligible": eligible if observed else False,
        "missed": 0 if observed else 1,
    }


def args(frame_id=1, *, people=None, camera="cam-a", session="s1"):
    if people is None:
        people = [tracked()]

    return {
        "camera_id": camera,
        "session_id": session,
        "frame_id": frame_id,
        "timestamp_utc": BASE + timedelta(seconds=frame_id),
        "monotonic_timestamp": float(frame_id),
        "width": 640,
        "height": 480,
        "tracked_persons": people,
    }


class FakePlugin:
    version = "test"

    def __init__(self, plugin_id, score=0.0, unknown=False):
        self.plugin_id = plugin_id
        self.score = score
        self.unknown = unknown

    def execute(self, context):
        observations = []

        for person in context.frame.persons:
            if person.track_id is None:
                continue

            eligible = (
                person.behavior_eligible
                and person.observed is True
                and person.predicted is not True
            )

            valid = eligible and not self.unknown

            item = {
                "track_id": person.track_id,
                "status": "ok" if valid else "unknown",
                "metrics": {"score": self.score} if valid else {},
                "evidence": {"source": "test"},
            }

            if valid:
                item["confidence"] = 0.95

            observations.append(item)

        return PluginResult(
            plugin_id=self.plugin_id,
            camera_id=context.frame.camera_id,
            session_id=context.frame.session_id,
            frame_id=context.frame.frame_id,
            payload={"observations": observations},
        )


def pipeline(score=0.9, unknown=False):
    plugins = (
        FakePlugin("phone_usage", score, unknown),
        FakePlugin("computer_usage", 0),
        FakePlugin("inactivity", 0),
        FakePlugin("gathering", 0),
    )

    return OperationalOutputPipeline(
        rules=RuleEventPipeline(
            execution=BehaviorExecutionPipeline(plugins=plugins)
        )
    )


def test_canonical_result_is_created():
    output = pipeline().process_tracking(**args())

    assert len(output.results) == 1
    assert isinstance(output.results[0], BehaviorResult)


def test_pending_event_is_not_confirmed():
    output = pipeline().process_tracking(**args())

    assert output.results[0].status == "active"
    assert output.overlays[0].state == "pending"


def test_second_positive_confirms():
    p = pipeline()
    p.process_tracking(**args(1))
    output = p.process_tracking(**args(2))

    assert output.results[0].status == "confirmed"
    assert output.overlays[0].state == "confirmed"


def test_no_event_without_positive_evidence():
    output = pipeline(score=0.0).process_tracking(**args())

    assert output.results[0].status == "no_event"


def test_unknown_evidence_not_negative():
    output = pipeline(unknown=True).process_tracking(**args())

    assert output.results[0].status == "unknown"


def test_real_detector_missing_is_unknown():
    output = OperationalOutputPipeline().process_tracking(**args())

    assert output.results[0].status == "unknown"


def test_predicted_track_hidden():
    output = pipeline().process_tracking(**args(
        people=[tracked(observed=False, eligible=False)]
    ))

    assert output.results[0].status == "unknown"
    assert output.overlays[0].show is False


def test_unconfirmed_observed_track_no_positive():
    output = pipeline().process_tracking(**args(
        people=[tracked(eligible=False)]
    ))

    assert output.results[0].status == "unknown"
    assert output.overlays[0].behavior_labels == ()


def test_bbox_preserves_geometry():
    output = pipeline().process_tracking(**args())

    assert output.overlays[0].bbox_xyxy == (40, 10, 90, 130)


def test_multiple_people_sorted():
    output = pipeline().process_tracking(**args(
        people=[tracked(9, 200), tracked(2, 40), tracked(5, 120)]
    ))

    assert [x.track_id for x in output.overlays] == [2, 5, 9]
    assert len(output.results) == 3


def test_empty_frame():
    output = pipeline().process_tracking(**args(people=[]))

    assert output.results == ()
    assert output.overlays == ()


def test_json_serialization():
    output = pipeline().process_tracking(**args())

    encoded = json.dumps(output.to_payload())
    decoded = json.loads(encoded)

    assert decoded["schema_version"] == "1.0"
    assert len(decoded["results"]) == 1
    assert len(decoded["overlays"]) == 1


def test_four_plugin_evidence_preserved():
    output = pipeline().process_tracking(**args())

    assert set(output.results[0].evidence) == {
        "phone_usage",
        "computer_usage",
        "inactivity",
        "gathering",
    }


def test_rule_evaluations_preserved():
    output = pipeline().process_tracking(**args())

    assert len(output.results[0].rule_evaluations) == 4


def test_transition_records_present():
    output = pipeline().process_tracking(**args())

    assert len(output.transitions) == 1
    assert output.transitions[0]["action"] == "started"


def test_confirmed_behavior_label():
    p = pipeline()
    p.process_tracking(**args(1))
    output = p.process_tracking(**args(2))

    assert "Phone evidence" in output.overlays[0].behavior_labels


def test_session_change_isolated():
    p = pipeline()
    p.process_tracking(**args(1, session="old"))
    output = p.process_tracking(**args(1, session="new"))

    assert output.session_id == "new"
    assert output.results[0].session_id == "new"
    assert any(
        x["event"]["session_id"] == "old"
        for x in output.transitions
    )


def test_clear_resets_frame_sequence():
    p = pipeline()

    p.process_tracking(**args(5))
    p.clear()
    output = p.process_tracking(**args(1))

    assert output.frame_id == 1


def test_formatter_rejects_wrong_type():
    with pytest.raises(TypeError):
        OperationalOutputFormatter().format({})


def test_no_misconduct_label():
    p = pipeline()
    p.process_tracking(**args(1))
    output = p.process_tracking(**args(2))

    text = json.dumps(output.to_payload()).lower()

    assert "misconduct" not in text
    assert "productivity" not in text

def test_mappingproxy_dataclass_serialization():
    """Immutable dataclass evidence is JSON-compatible."""
    from dataclasses import dataclass
    from types import MappingProxyType

    from company_vision.behaviors.behavior_result import (
        serialize_component,
    )

    @dataclass(frozen=True)
    class Record:
        evidence: object

    record = Record(
        evidence=MappingProxyType({
            "source": "test",
            "nested": MappingProxyType({
                "confidence": 0.9,
            }),
        })
    )

    result = serialize_component(record)

    assert result == {
        "evidence": {
            "source": "test",
            "nested": {"confidence": 0.9},
        },
    }


def test_real_event_and_rule_records_are_json_serializable():
    """Serialize RuleEvaluation and EventTransition safely."""
    from company_vision.behaviors.behavior_result import (
        serialize_component,
    )

    p = pipeline()
    batch = p.rules.process_tracking(**args(1))

    assert batch.transitions
    assert batch.evaluations

    encoded = serialize_component({
        "transitions": batch.transitions,
        "evaluations": batch.evaluations,
        "active": batch.active_events,
    })

    json.dumps(encoded)

    assert encoded["transitions"][0]["action"] == "started"
    assert encoded["evaluations"][0]["rule_id"]