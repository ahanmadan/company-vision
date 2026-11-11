from datetime import UTC, datetime, timedelta

import pytest

from company_vision.behaviors.event_manager import (
    BehaviorEvent,
    EventConfigurationError,
    EventContractError,
    EventManager,
    EventPolicy,
)
from company_vision.behaviors.rule_engine import RuleEvaluation

BASE = datetime(2026, 11, 11, 12, 0, tzinfo=UTC)


def evaluation(
    frame_id,
    outcome="triggered",
    *,
    track_id=7,
    rule_id="phone-rule",
    camera_id="camera-1",
    session_id="session-1",
):
    return RuleEvaluation(
        rule_id=rule_id,
        plugin_id="phone_usage",
        camera_id=camera_id,
        session_id=session_id,
        frame_id=frame_id,
        track_id=track_id,
        outcome=outcome,
        metric_key="score",
        observed_value=0.9 if outcome == "triggered" else 0.1,
        threshold=0.7,
        operator="gte",
        confidence=0.95,
        reason="test evidence",
        evidence={"source": "unit_test"},
    )


def timestamp(frame_id):
    return BASE + timedelta(seconds=frame_id)


def process(manager, frame_id, outcome="triggered", **kwargs):
    return manager.process(
        [evaluation(frame_id, outcome, **kwargs)],
        timestamp_utc=timestamp(frame_id),
    )


def test_first_trigger_creates_pending_event():
    manager = EventManager()

    result = process(manager, 1)

    assert len(result) == 1
    assert result[0].action == "started"
    assert result[0].event.status == "pending"
    assert manager.active_event_count == 1


def test_second_trigger_confirms_event():
    manager = EventManager()

    process(manager, 1)
    result = process(manager, 2)

    assert len(result) == 1
    assert result[0].action == "confirmed"
    assert result[0].event.status == "confirmed"
    assert result[0].event.triggered_count == 2


def test_triggered_frames_do_not_create_duplicate_events():
    manager = EventManager()

    first = process(manager, 1)[0].event
    second = process(manager, 2)[0].event
    third = process(manager, 3)[0].event

    assert first.event_id == second.event_id == third.event_id
    assert manager.active_event_count == 1


def test_confirmed_event_is_updated():
    manager = EventManager()

    process(manager, 1)
    process(manager, 2)

    result = process(manager, 3)

    assert result[0].action == "updated"
    assert result[0].event.status == "confirmed"
    assert result[0].event.last_frame_id == 3


def test_event_ends_after_consecutive_negative_results():
    manager = EventManager()

    process(manager, 1)
    process(manager, 2)

    process(manager, 3, "not_triggered")
    result = process(manager, 4, "not_triggered")

    assert result[0].action == "ended"
    assert result[0].event.end_reason == "consecutive_not_triggered"
    assert result[0].event.ended_frame_id == 4
    assert manager.active_event_count == 0


def test_unknown_does_not_end_event():
    manager = EventManager()

    process(manager, 1)
    process(manager, 2)
    process(manager, 3, "not_triggered")

    result = process(manager, 4, "unknown")

    assert result[0].action == "updated"
    assert result[0].event.status == "confirmed"
    assert result[0].event.consecutive_clear_count == 0
    assert manager.active_event_count == 1


def test_not_applicable_does_not_create_event():
    manager = EventManager()

    result = process(manager, 1, "not_applicable")

    assert result == ()
    assert manager.active_event_count == 0


def test_unknown_does_not_create_event():
    manager = EventManager()

    assert process(manager, 1, "unknown") == ()
    assert manager.active_event_count == 0


def test_unknown_resets_consecutive_clear_counter():
    manager = EventManager()

    process(manager, 1)
    process(manager, 2)
    process(manager, 3, "not_triggered")
    process(manager, 4, "unknown")

    result = process(manager, 5, "not_triggered")

    assert result[0].event.consecutive_clear_count == 1
    assert manager.active_event_count == 1


def test_different_tracks_have_independent_events():
    manager = EventManager()

    manager.process(
        [
            evaluation(1, track_id=7),
            evaluation(1, track_id=8),
        ],
        timestamp_utc=timestamp(1),
    )

    assert manager.active_event_count == 2


def test_different_rules_have_independent_events():
    manager = EventManager()

    manager.process(
        [
            evaluation(1, rule_id="phone-rule"),
            evaluation(1, rule_id="inactivity-rule"),
        ],
        timestamp_utc=timestamp(1),
    )

    assert manager.active_event_count == 2


def test_camera_and_session_are_isolated():
    manager = EventManager()

    manager.process(
        [evaluation(1, camera_id="camera-A")],
        timestamp_utc=timestamp(1),
    )

    manager.process(
        [evaluation(1, camera_id="camera-B")],
        timestamp_utc=timestamp(1),
    )

    assert manager.active_event_count == 2
def test_duplicate_evaluation_is_rejected():
    manager = EventManager()
    item = evaluation(1)

    with pytest.raises(EventContractError):
        manager.process(
            [item, item],
            timestamp_utc=timestamp(1),
        )

    assert manager.active_event_count == 0


def test_mixed_frame_batch_is_rejected():
    manager = EventManager()

    with pytest.raises(EventContractError):
        manager.process(
            [evaluation(1), evaluation(2, track_id=8)],
            timestamp_utc=timestamp(2),
        )

    assert manager.active_event_count == 0


def test_out_of_order_frame_is_rejected():
    manager = EventManager()

    process(manager, 5)

    with pytest.raises(EventContractError):
        process(manager, 4)

    assert manager.active_event_count == 1


def test_non_utc_timestamp_is_rejected():
    manager = EventManager()

    local_time = datetime(
        2026, 11, 11, 12, 0,
        tzinfo=timedelta(hours=3) if False else UTC,
    )

    with pytest.raises(EventContractError):
        manager.process(
            [evaluation(1)],
            timestamp_utc=local_time.replace(
                tzinfo=__import__("datetime").timezone(
                    timedelta(hours=3)
                )
            ),
        )


def test_track_loss_ends_only_matching_track_events():
    manager = EventManager()

    manager.process(
        [
            evaluation(1, track_id=7),
            evaluation(1, track_id=8),
        ],
        timestamp_utc=timestamp(1),
    )

    ended = manager.end_track(
        "camera-1",
        "session-1",
        7,
        frame_id=2,
        timestamp_utc=timestamp(2),
    )

    assert len(ended) == 1
    assert ended[0].event.track_id == 7
    assert manager.active_event_count == 1


def test_session_end_ends_only_matching_session():
    manager = EventManager()

    manager.process(
        [evaluation(1, session_id="session-1")],
        timestamp_utc=timestamp(1),
    )

    manager.process(
        [evaluation(1, session_id="session-2")],
        timestamp_utc=timestamp(1),
    )

    ended = manager.end_session(
        "camera-1",
        "session-1",
        frame_id=2,
        timestamp_utc=timestamp(2),
    )

    assert len(ended) == 1
    assert ended[0].event.session_id == "session-1"
    assert manager.active_event_count == 1
def test_event_snapshots_are_immutable():
    manager = EventManager()

    event = process(manager, 1)[0].event

    assert isinstance(event, BehaviorEvent)

    with pytest.raises((AttributeError, TypeError)):
        event.status = "confirmed"

    with pytest.raises(TypeError):
        event.evidence["changed"] = True


def test_active_event_filtering():
    manager = EventManager()

    manager.process(
        [evaluation(1, camera_id="camera-A")],
        timestamp_utc=timestamp(1),
    )

    manager.process(
        [evaluation(1, camera_id="camera-B")],
        timestamp_utc=timestamp(1),
    )

    result = manager.get_active_events(camera_id="camera-A")

    assert len(result) == 1
    assert result[0].camera_id == "camera-A"
def test_clear_removes_all_state():
    manager = EventManager()

    process(manager, 1)
    manager.clear()

    assert manager.active_event_count == 0
    assert manager.get_active_events() == ()


def test_invalid_policy_is_rejected():
    with pytest.raises(EventConfigurationError):
        EventPolicy(confirm_after_triggered_frames=0)

    with pytest.raises(EventConfigurationError):
        EventPolicy(end_after_not_triggered_frames=-1)


def test_group_event_with_no_track_id():
    manager = EventManager()

    result = process(manager, 1, track_id=None)

    assert result[0].event.track_id is None
    assert manager.active_event_count == 1
