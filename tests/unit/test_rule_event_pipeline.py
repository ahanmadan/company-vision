"""Node 5.9.4: rule/event integration contract tests."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from company_vision.behaviors.behavior_execution_pipeline import (
    BehaviorExecutionPipeline,
)
from company_vision.behaviors.event_manager import EventPolicy
from company_vision.behaviors.plugin_registry import PluginResult
from company_vision.behaviors.rule_engine import RuleDefinition
from company_vision.behaviors.rule_event_pipeline import (
    DEFAULT_RULES,
    RuleEventPipeline,
    RuleEventPipelineError,
)

BASE = datetime(2026, 11, 12, tzinfo=UTC)


def tracked(track_id=1, x=40.0, eligible=True):
    points = np.array([
        [x + 18, 20 + i * 5, 0.9]
        for i in range(9)
    ], dtype=np.float32)

    return {
        "track_id": track_id,
        "bbox": np.array([x, 10, x + 50, 130]),
        "keypoints": points,
        "raw_keypoints": points.copy(),
        "keypoint_weights": np.ones(9) * 0.85,
        "confidence": 0.95,
        "observed": True,
        "predicted": False,
        "behavior_eligible": eligible,
        "missed": 0,
    }


def arguments(
    frame_id=1,
    *,
    people=None,
    camera="cam-a",
    session="s1",
    events=(),
):
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
        "track_lifecycle_events": events,
    }


class FakePlugin:
    """Deterministic evidence source for orchestration tests."""

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

            valid = (
                person.behavior_eligible
                and person.observed is True
                and person.predicted is not True
                and not self.unknown
            )

            observation = {
                "track_id": person.track_id,
                "status": "ok" if valid else "unknown",
                "metrics": (
                    {"score": self.score}
                    if valid else {}
                ),
                "evidence": {"source": "synthetic_test"},
            }

            if valid:
                observation["confidence"] = 0.95

            observations.append(observation)

        return PluginResult(
            plugin_id=self.plugin_id,
            camera_id=context.frame.camera_id,
            session_id=context.frame.session_id,
            frame_id=context.frame.frame_id,
            payload={"observations": observations},
        )


def pipeline(score=0.9, unknown=False, policy=None):
    plugins = (
        FakePlugin("phone_usage", score, unknown),
        FakePlugin("computer_usage", 0),
        FakePlugin("inactivity", 0),
        FakePlugin("gathering", 0),
    )
    return RuleEventPipeline(
        execution=BehaviorExecutionPipeline(plugins=plugins),
        policy=policy,
    )


def test_default_four_rules():
    assert len(DEFAULT_RULES) == 4
    assert {r.plugin_id for r in DEFAULT_RULES} == {
        "phone_usage",
        "computer_usage",
        "inactivity",
        "gathering",
    }


def test_real_pipeline_smoke():
    result = RuleEventPipeline().process_tracking(**arguments())
    assert len(result.evaluations) == 4
    assert result.frame_id == 1


def test_first_positive_starts_pending_event():
    p = pipeline()
    result = p.process_tracking(**arguments(1))

    assert len(result.transitions) == 1
    assert result.transitions[0].action == "started"
    assert result.transitions[0].event.status == "pending"


def test_second_positive_confirms_same_event():
    p = pipeline()

    first = p.process_tracking(**arguments(1))
    second = p.process_tracking(**arguments(2))

    assert second.transitions[0].action == "confirmed"
    assert (
        first.transitions[0].event.event_id
        == second.transitions[0].event.event_id
    )


def test_repeated_positive_does_not_duplicate():
    p = pipeline()

    p.process_tracking(**arguments(1))
    p.process_tracking(**arguments(2))
    result = p.process_tracking(**arguments(3))

    assert result.transitions[0].action == "updated"
    assert p.active_event_count == 1


def test_negative_evidence_ends_after_policy():
    p = pipeline()

    p.process_tracking(**arguments(1))
    p.process_tracking(**arguments(2))

    p.execution.registry.get("phone_usage").score = 0.0

    p.process_tracking(**arguments(3))
    result = p.process_tracking(**arguments(4))

    assert result.transitions[0].action == "ended"
    assert p.active_event_count == 0


def test_unknown_does_not_create_event():
    p = pipeline(unknown=True)

    result = p.process_tracking(**arguments())

    assert result.transitions == ()
    assert p.active_event_count == 0


def test_unknown_does_not_close_existing_event():
    p = pipeline()

    p.process_tracking(**arguments(1))
    p.process_tracking(**arguments(2))

    p.execution.registry.get("phone_usage").unknown = True
    result = p.process_tracking(**arguments(3))

    assert p.active_event_count == 1
    assert result.transitions[0].action == "updated"


def test_missing_object_detector_not_fake_phone():
    result = RuleEventPipeline().process_tracking(**arguments())

    phone = next(
        e for e in result.evaluations
        if e.plugin_id == "phone_usage"
    )

    assert phone.outcome == "unknown"


def test_multiple_tracks_have_separate_events():
    p = pipeline()

    result = p.process_tracking(**arguments(
        people=[tracked(1), tracked(2, x=120)],
    ))

    assert p.active_event_count == 2
    assert {e.track_id for e in result.evaluations} == {1, 2}


def test_no_people_does_not_invent_negative():
    p = pipeline()

    result = p.process_tracking(**arguments(people=[]))

    assert result.evaluations
    assert all(
        e.outcome == "unknown"
        for e in result.evaluations
    )
    assert result.transitions == ()


def test_terminal_track_event_closes_record():
    p = pipeline()

    p.process_tracking(**arguments(1))

    result = p.process_tracking(**arguments(
        2,
        people=[],
        events=[{"event_type": "ended", "track_id": 1}],
    ))

    assert any(t.action == "ended" for t in result.transitions)
    assert p.active_event_count == 0


def test_terminal_event_for_present_track_rejected():
    p = pipeline()

    with pytest.raises(RuleEventPipelineError):
        p.process_tracking(**arguments(
            events=[{"event_type": "ended", "track_id": 1}],
        ))

    assert p.active_event_count == 0


def test_camera_session_change_ends_old_events():
    p = pipeline()

    p.process_tracking(**arguments(1, session="old"))
    result = p.process_tracking(**arguments(1, session="new"))

    assert any(
        t.action == "ended"
        and t.event.session_id == "old"
        for t in result.transitions
    )


def test_different_cameras_isolated():
    p = pipeline()

    p.process_tracking(**arguments(camera="a"))
    p.process_tracking(**arguments(camera="b"))

    assert p.active_event_count == 2


def test_duplicate_frame_rejected():
    p = pipeline()

    p.process_tracking(**arguments(1))

    with pytest.raises(ValueError):
        p.process_tracking(**arguments(1))


def test_configurable_threshold():
    custom = (
        RuleDefinition(
            rule_id="phone_test",
            plugin_id="phone_usage",
            metric_key="score",
            operator="gte",
            threshold=0.95,
            min_confidence=0.5,
        ),
    )

    p = RuleEventPipeline(
        execution=BehaviorExecutionPipeline(plugins=(
            FakePlugin("phone_usage", 0.9),
            FakePlugin("computer_usage"),
            FakePlugin("inactivity"),
            FakePlugin("gathering"),
        )),
        rules=custom,
    )

    result = p.process_tracking(**arguments())
    assert result.evaluations[0].outcome == "not_triggered"
    assert result.transitions == ()


def test_custom_policy():
    p = pipeline(policy=EventPolicy(
        confirm_after_triggered_frames=3,
        end_after_not_triggered_frames=2,
    ))

    p.process_tracking(**arguments(1))
    p.process_tracking(**arguments(2))
    result = p.process_tracking(**arguments(3))

    assert result.transitions[0].action == "confirmed"


def test_evaluations_for_track_have_priority_order():
    p = pipeline()
    result = p.process_tracking(**arguments())

    evaluations = result.evaluations_for_track(1)

    assert [e.plugin_id for e in evaluations] == [
        "phone_usage",
        "computer_usage",
        "inactivity",
        "gathering",
    ]


def test_clear_resets_state():
    p = pipeline()

    p.process_tracking(**arguments(5))
    p.clear()

    result = p.process_tracking(**arguments(1))

    assert result.frame_id == 1
    assert p.active_event_count == 1


def test_invalid_rule_plugin_rejected():
    with pytest.raises(ValueError):
        RuleEventPipeline(rules=(
            RuleDefinition(
                rule_id="bad",
                plugin_id="missing_plugin",
                metric_key="score",
                operator="gte",
                threshold=0.5,
            ),
        ))


def test_invalid_process_input_rejected():
    p = pipeline()

    with pytest.raises(TypeError):
        p.process({})


def test_explicit_session_clear():
    p = pipeline()

    p.process_tracking(**arguments(1))

    transitions = p.clear_session(
        "cam-a",
        "s1",
        frame_id=2,
        timestamp_utc=BASE + timedelta(seconds=2),
    )

    assert len(transitions) == 1
    assert transitions[0].action == "ended"
    assert p.active_event_count == 0