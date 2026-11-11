from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from company_vision.behaviors.feature_extractor import (
    FeatureExtractor,
)
from company_vision.behaviors.schemas import (
    BehaviorFrame,
    BehaviorKeypoint,
    BehaviorPerson,
)
from company_vision.behaviors.temporal_state_manager import (
    TemporalStateError,
    TemporalStateManager,
)
from company_vision.core.schemas import BoundingBox

BASE_TIME = datetime(2026, 11, 11, tzinfo=UTC)


def make_person(
    track_id=1,
    observed=True,
    predicted=False,
    eligible=True,
    offset=0,
):

    names = {
        "nose": (50, 20),
        "left_eye": (45, 22),
        "right_eye": (55, 22),
        "left_ear": (40, 25),
        "right_ear": (60, 25),
        "left_shoulder": (40, 60),
        "right_shoulder": (60, 60),
        "left_elbow": (25, 80),
        "right_elbow": (75, 80),
    }

    points = {
        name: BehaviorKeypoint(
            x=x + offset,
            y=y,
            confidence=0.9,
            valid=True,
        )
        for name, (x, y) in names.items()
    }

    return BehaviorPerson(
        track_id=track_id,
        bbox=BoundingBox(
            x1=20 + offset,
            y1=10,
            x2=100 + offset,
            y2=110,
        ),
        confidence=0.95,
        keypoints=points,
        keypoint_weights={
            name: 0.8 for name in names
        },
        observed=observed,
        predicted=predicted,
        behavior_eligible=eligible,
    )


def make_features(
    frame_id,
    track_ids=(1,),
    camera_id="camera_01",
    session_id="session_01",
    seconds=None,
    monotonic_seconds=None,
    observed=True,
    predicted=False,
    eligible=True,
):

    if seconds is None:
        seconds = frame_id

    persons = [
        make_person(
            track_id=track_id,
            observed=observed,
            predicted=predicted,
            eligible=eligible,
            offset=index * 110,
        )
        for index, track_id in enumerate(track_ids)
    ]

    frame = BehaviorFrame(
        camera_id=camera_id,
        session_id=session_id,
        frame_id=frame_id,
        timestamp_utc=BASE_TIME + timedelta(seconds=seconds),
        monotonic_timestamp=100.0 + (seconds if monotonic_seconds is None else monotonic_seconds),
        width=max(2000, len(persons) * 110 + 200),
        height=600,
        persons=persons,
        source="tracking_output",
        source_status="ok",
    )

    return FeatureExtractor().extract(frame)


def test_first_observation_creates_history():

    manager = TemporalStateManager()

    result = manager.update(make_features(1))

    assert len(result) == 1
    assert result[0].track_id == 1
    assert result[0].observation_count == 1
    assert manager.active_track_count == 1


def test_multiple_observations_append_chronologically():

    manager = TemporalStateManager()

    manager.update(make_features(1))
    manager.update(make_features(2))
    manager.update(make_features(3))

    history = manager.get_history(
        "camera_01", "session_01", 1
    )

    assert history is not None
    assert [x.frame_id for x in history.observations] == [1, 2, 3]


def test_tracks_are_independent():

    manager = TemporalStateManager()

    manager.update(make_features(1, track_ids=(1, 2, 3)))

    manager.update(make_features(2, track_ids=(1, 2)))

    first = manager.get_history("camera_01", "session_01", 1)
    second = manager.get_history("camera_01", "session_01", 2)
    third = manager.get_history("camera_01", "session_01", 3)

    assert first.observation_count == 2
    assert second.observation_count == 2
    assert third.observation_count == 1


def test_camera_histories_are_isolated():

    manager = TemporalStateManager()

    manager.update(
        make_features(1, camera_id="camera_A")
    )

    manager.update(
        make_features(1, camera_id="camera_B")
    )

    assert manager.active_track_count == 2

    assert manager.get_history(
        "camera_A", "session_01", 1
    ) is not None

    assert manager.get_history(
        "camera_B", "session_01", 1
    ) is not None


def test_session_change_clears_old_camera_history():

    manager = TemporalStateManager()

    manager.update(make_features(1, track_ids=(1, 2)))

    manager.update(
        make_features(
            1,
            track_ids=(1,),
            session_id="session_02",
        )
    )

    assert manager.get_history(
        "camera_01", "session_01", 1
    ) is None

    assert manager.get_history(
        "camera_01", "session_01", 2
    ) is None

    assert manager.get_history(
        "camera_01", "session_02", 1
    ) is not None


def test_session_change_does_not_clear_other_cameras():

    manager = TemporalStateManager()

    manager.update(
        make_features(1, camera_id="camera_A")
    )

    manager.update(
        make_features(1, camera_id="camera_B")
    )

    manager.update(
        make_features(
            1,
            camera_id="camera_A",
            session_id="session_02",
        )
    )

    assert manager.get_history(
        "camera_B", "session_01", 1
    ) is not None


def test_explicit_track_loss_removes_history():

    manager = TemporalStateManager()

    manager.update(make_features(1, track_ids=(1, 2)))


    manager.update(
        make_features(2, track_ids=(2,))
    )

    assert manager.end_track(
        "camera_01", "session_01", 1
    ) is True

    assert manager.get_history(
        "camera_01", "session_01", 1
    ) is None

    assert manager.get_history(
        "camera_01", "session_01", 2
    ) is not None


def test_ended_track_ids_are_processed_atomically():

    manager = TemporalStateManager()

    manager.update(make_features(1, track_ids=(1, 2)))

    result = manager.update(
        make_features(2, track_ids=(2,)),
        ended_track_ids=(1,),
    )

    assert len(result) == 1
    assert manager.get_history(
        "camera_01", "session_01", 1
    ) is None

    assert manager.get_history(
        "camera_01", "session_01", 2
    ).observation_count == 2


def test_unknown_ended_track_is_safe():

    manager = TemporalStateManager()

    manager.update(
        make_features(1),
        ended_track_ids=(99,),
    )

    assert manager.active_track_count == 1


def test_history_is_bounded():

    manager = TemporalStateManager(max_history_per_track=3)

    for frame_id in range(1, 8):
        manager.update(make_features(frame_id))

    history = manager.get_history(
        "camera_01", "session_01", 1
    )

    assert history.observation_count == 3

    assert [
        item.frame_id for item in history.observations
    ] == [5, 6, 7]


def test_empty_frame_does_not_implicitly_delete_track():

    manager = TemporalStateManager()

    manager.update(make_features(1))

    manager.update(make_features(2, track_ids=()))

    assert manager.get_history(
        "camera_01", "session_01", 1
    ).observation_count == 1


def test_untracked_person_is_not_stored():

    manager = TemporalStateManager()

    manager.update(make_features(1, track_ids=(None,), eligible=False))

    assert manager.active_track_count == 0


def test_predicted_observation_preserves_ineligibility():

    manager = TemporalStateManager()

    manager.update(
        make_features(
            1,
            observed=False,
            predicted=True,
            eligible=False,
        )
    )

    history = manager.get_history(
        "camera_01", "session_01", 1
    )

    person = history.latest.person

    assert person.predicted is True
    assert person.behavior_eligible is False


def test_variable_person_count():

    manager = TemporalStateManager()

    ids = tuple(range(1, 21))

    manager.update(make_features(1, track_ids=ids))

    assert manager.active_track_count == 20


def test_duplicate_track_ids_are_rejected():

    manager = TemporalStateManager()

    frame = make_features(1)

    invalid = replace(
        frame,
        persons=(frame.persons[0], frame.persons[0]),
    )

    with pytest.raises(TemporalStateError):
        manager.update(invalid)

    assert manager.active_track_count == 0


def test_out_of_order_frame_is_rejected():

    manager = TemporalStateManager()

    manager.update(make_features(5))

    with pytest.raises(TemporalStateError):
        manager.update(make_features(4))

    history = manager.get_history(
        "camera_01", "session_01", 1
    )

    assert history.observation_count == 1


def test_utc_timestamp_regression_is_rejected():

    manager = TemporalStateManager()

    manager.update(make_features(2, seconds=20))

    with pytest.raises(TemporalStateError):
        manager.update(make_features(3, seconds=10))


def test_monotonic_timestamp_regression_is_rejected():

    manager = TemporalStateManager()

    manager.update(make_features(2, seconds=20))

    with pytest.raises(TemporalStateError):
        manager.update(
            make_features(
                3,
                seconds=30,
                monotonic_seconds=10,
            )
        )


def test_failed_update_does_not_partially_modify_state():

    manager = TemporalStateManager()

    manager.update(
        make_features(1, track_ids=(1, 2))
    )

    with pytest.raises(TemporalStateError):
        manager.update(
            make_features(2, track_ids=(1, 2)),
            ended_track_ids=(99, 99),
        )

    assert manager.get_history(
        "camera_01", "session_01", 1
    ).observation_count == 1

    assert manager.get_history(
        "camera_01", "session_01", 2
    ).observation_count == 1


def test_wrong_input_type_is_rejected():

    manager = TemporalStateManager()

    with pytest.raises(TypeError):
        manager.update({})


def test_invalid_history_limit_is_rejected():

    with pytest.raises(ValueError):
        TemporalStateManager(max_history_per_track=0)

    with pytest.raises(ValueError):
        TemporalStateManager(max_history_per_track=-1)


def test_active_histories_are_deterministically_sorted():

    manager = TemporalStateManager()

    manager.update(
        make_features(1, track_ids=(9, 2, 5))
    )

    histories = manager.get_active_histories(
        "camera_01", "session_01"
    )

    assert [item.track_id for item in histories] == [2, 5, 9]


def test_clear_session_removes_only_target_session():

    manager = TemporalStateManager()

    manager.update(make_features(1))

    manager.update(
        make_features(
            1,
            camera_id="camera_02",
            session_id="session_02",
        )
    )

    removed = manager.clear_session(
        "camera_01", "session_01"
    )

    assert removed == 1

    assert manager.get_history(
        "camera_02", "session_02", 1
    ) is not None


def test_clear_removes_all_state():

    manager = TemporalStateManager()

    manager.update(make_features(1, track_ids=(1, 2)))

    manager.clear()

    assert manager.active_track_count == 0


def test_history_snapshots_are_immutable():

    manager = TemporalStateManager()

    manager.update(make_features(1))

    history = manager.get_history(
        "camera_01", "session_01", 1
    )

    with pytest.raises((AttributeError, TypeError)):
        history.track_id = 99

    assert isinstance(history.observations, tuple)


def test_session_change_resets_frame_id_sequence():

    manager = TemporalStateManager()

    manager.update(make_features(100))

    result = manager.update(
        make_features(
            1,
            session_id="session_02",
        )
    )

    assert result[0].observation_count == 1
    assert result[0].latest.frame_id == 1