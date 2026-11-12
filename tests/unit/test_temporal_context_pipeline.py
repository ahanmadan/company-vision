"""Node 5.9.2: temporal context integration tests."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from company_vision.behaviors.plugin_registry import BehaviorPluginContext
from company_vision.behaviors.temporal_context_pipeline import (
    BehaviorTemporalPipeline,
    TemporalPipelineError,
)

BASE = datetime(2026, 11, 12, tzinfo=UTC)


def person(track=1, observed=True, eligible=True, x=20.0):
    points = np.array(
        [[x + 15, 15 + i * 5, 0.9] for i in range(9)],
        dtype=np.float32,
    )

    return {
        "track_id": track,
        "bbox": np.array([x, 10, x + 40, 100], dtype=np.float32),
        "keypoints": points,
        "raw_keypoints": points.copy() if observed else None,
        "keypoint_weights": np.ones(9, dtype=np.float32) * 0.8,
        "confidence": 0.95,
        "observed": observed,
        "predicted": not observed,
        "behavior_eligible": eligible if observed else False,
        "missed": 0 if observed else 1,
    }


def args(
    frame_id=1,
    *,
    camera="cam-a",
    session="session-a",
    people=None,
    seconds=None,
    events=(),
):
    if people is None:
        people = [person()]

    if seconds is None:
        seconds = float(frame_id)

    return {
        "camera_id": camera,
        "session_id": session,
        "frame_id": frame_id,
        "timestamp_utc": BASE + timedelta(seconds=seconds),
        "monotonic_timestamp": seconds,
        "width": 640,
        "height": 480,
        "tracked_persons": people,
        "track_lifecycle_events": events,
    }


def test_first_frame_context():
    pipeline = BehaviorTemporalPipeline()
    result = pipeline.process_tracking(**args())

    assert isinstance(result, BehaviorPluginContext)
    assert result.frame.persons[0].track_id == 1
    assert len(result.histories) == 1
    assert result.histories[0].observation_count == 1


def test_history_accumulates():
    pipeline = BehaviorTemporalPipeline()

    pipeline.process_tracking(**args(1))
    result = pipeline.process_tracking(**args(2))

    assert result.histories[0].observation_count == 2


def test_history_bounded():
    pipeline = BehaviorTemporalPipeline(max_history_per_track=2)

    for frame_id in range(1, 5):
        pipeline.process_tracking(**args(frame_id))

    history = pipeline.get_history("cam-a", "session-a", 1)

    assert history.observation_count == 2
    assert [x.frame_id for x in history.observations] == [3, 4]


def test_multiple_people_independent():
    pipeline = BehaviorTemporalPipeline()

    pipeline.process_tracking(**args(
        1, people=[person(1), person(2, x=100)],
    ))

    result = pipeline.process_tracking(**args(
        2, people=[person(1), person(2, x=100)],
    ))

    assert [h.track_id for h in result.histories] == [1, 2]
    assert all(h.observation_count == 2 for h in result.histories)


def test_camera_isolation():
    pipeline = BehaviorTemporalPipeline()

    pipeline.process_tracking(**args(camera="cam-a"))
    pipeline.process_tracking(**args(camera="cam-b"))

    assert pipeline.active_track_count == 2


def test_new_session_resets_camera_history():
    pipeline = BehaviorTemporalPipeline()

    pipeline.process_tracking(**args(10, session="old"))
    result = pipeline.process_tracking(**args(1, session="new"))

    assert result.histories[0].observation_count == 1
    assert pipeline.get_history("cam-a", "old", 1) is None


def test_session_reset_preserves_other_camera():
    pipeline = BehaviorTemporalPipeline()

    pipeline.process_tracking(**args(camera="cam-a", session="old"))
    pipeline.process_tracking(**args(camera="cam-b"))
    pipeline.process_tracking(**args(camera="cam-a", session="new"))

    assert pipeline.get_history("cam-b", "session-a", 1) is not None


def test_terminal_event_ends_track():
    pipeline = BehaviorTemporalPipeline()

    pipeline.process_tracking(**args(1))

    result = pipeline.process_tracking(**args(
        2,
        people=[],
        events=[{"event_type": "ended", "track_id": 1}],
    ))

    assert result.histories == ()
    assert pipeline.get_history("cam-a", "session-a", 1) is None


def test_implicit_absence_does_not_end_track():
    pipeline = BehaviorTemporalPipeline()

    pipeline.process_tracking(**args(1))
    pipeline.process_tracking(**args(2, people=[]))

    assert pipeline.get_history("cam-a", "session-a", 1) is not None


def test_absent_track_not_exposed_in_current_context():
    pipeline = BehaviorTemporalPipeline()

    pipeline.process_tracking(**args(1))
    result = pipeline.process_tracking(**args(2, people=[]))

    assert result.histories == ()


def test_predicted_track_breaks_history():
    pipeline = BehaviorTemporalPipeline()

    pipeline.process_tracking(**args(1))

    result = pipeline.process_tracking(**args(
        2, people=[person(observed=False, eligible=False)],
    ))

    assert result.histories == ()
    assert pipeline.get_history("cam-a", "session-a", 1) is None

    resumed = pipeline.process_tracking(**args(3))

    assert resumed.histories[0].observation_count == 1


def test_unconfirmed_track_breaks_history():
    pipeline = BehaviorTemporalPipeline()

    pipeline.process_tracking(**args(1))

    result = pipeline.process_tracking(**args(
        2, people=[person(eligible=False)],
    ))

    assert result.histories == ()


def test_duplicate_frame_rejected_without_mutation():
    pipeline = BehaviorTemporalPipeline()

    pipeline.process_tracking(**args(1))

    with pytest.raises(TemporalPipelineError):
        pipeline.process_tracking(**args(1))

    history = pipeline.get_history("cam-a", "session-a", 1)

    assert history.observation_count == 1


def test_out_of_order_frame_rejected():
    pipeline = BehaviorTemporalPipeline()

    pipeline.process_tracking(**args(3))

    with pytest.raises(TemporalPipelineError):
        pipeline.process_tracking(**args(2))


def test_time_regression_rejected():
    pipeline = BehaviorTemporalPipeline()

    pipeline.process_tracking(**args(1, seconds=10))

    with pytest.raises(TemporalPipelineError):
        pipeline.process_tracking(**args(2, seconds=9))


def test_failed_event_does_not_mutate_histories():
    pipeline = BehaviorTemporalPipeline()

    pipeline.process_tracking(**args(1))

    with pytest.raises(TemporalPipelineError):
        pipeline.process_tracking(**args(
            2,
            events=[{"event_type": "ended", "track_id": 1}],
        ))

    assert pipeline.get_history(
        "cam-a", "session-a", 1
    ).observation_count == 1


def test_terminal_event_without_track_rejected():
    pipeline = BehaviorTemporalPipeline()

    pipeline.process_tracking(**args(1))

    with pytest.raises(TemporalPipelineError):
        pipeline.process_tracking(**args(
            2,
            people=[],
            events=[{"event_type": "lost"}],
        ))


def test_duplicate_terminal_events_rejected():
    pipeline = BehaviorTemporalPipeline()

    with pytest.raises(TemporalPipelineError):
        pipeline.process_tracking(**args(
            1,
            people=[],
            events=[
                {"event_type": "ended", "track_id": 1},
                {"event_type": "lost", "track_id": 1},
            ],
        ))


def test_full_context_contains_latest_frame():
    pipeline = BehaviorTemporalPipeline()

    result = pipeline.process_tracking(**args(1))

    assert result.histories[0].latest.frame_id == 1
    assert result.frame.frame_id == 1


def test_no_fixed_person_limit():
    pipeline = BehaviorTemporalPipeline()

    people = [
        person(track=i, x=10 + i * 45)
        for i in range(1, 11)
    ]

    result = pipeline.process_tracking(**args(
        1,
        people=people,

    ))

    assert len(result.histories) == 10


def test_clear_resets_timeline():
    pipeline = BehaviorTemporalPipeline()

    pipeline.process_tracking(**args(5))
    pipeline.clear()

    result = pipeline.process_tracking(**args(1))

    assert result.histories[0].observation_count == 1


def test_invalid_history_size():
    with pytest.raises(ValueError):
        BehaviorTemporalPipeline(max_history_per_track=0)


def test_unknown_terminal_type_does_not_delete_history():
    pipeline = BehaviorTemporalPipeline()

    pipeline.process_tracking(**args(1))
    pipeline.process_tracking(**args(
        2,
        people=[],
        events=[{"event_type": "observed", "track_id": 1}],
    ))

    assert pipeline.get_history("cam-a", "session-a", 1) is not None