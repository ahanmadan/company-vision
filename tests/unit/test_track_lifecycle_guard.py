from company_vision.vision.track_lifecycle_guard import (
    TrackLifecycleGuard,
)


def track(track_id, observed=True):
    return {
        "track_id": track_id,
        "observed": observed,
        "predicted": not observed,
        "keypoints": None,
    }


def test_observed_track_requires_confirmation():
    guard = TrackLifecycleGuard(
        min_confirmed_observations=2
    )

    assert guard.update([track(1)], 0.0) == []
    result = guard.update([track(1)], 0.04)

    assert len(result) == 1
    assert result[0]["behavior_eligible"] is True


def test_prediction_is_not_published():
    guard = TrackLifecycleGuard(
        min_confirmed_observations=1
    )

    assert len(guard.update([track(1)], 0.0)) == 1
    assert guard.update([track(1, False)], 0.04) == []
    assert guard.retained_tracks == 1


def test_short_occlusion_retains_identity():
    guard = TrackLifecycleGuard(
        max_hidden_seconds=0.75,
        min_confirmed_observations=1,
    )

    guard.update([track(7)], 0.0)
    assert guard.update([track(7, False)], 0.3) == []

    result = guard.update([track(7)], 0.5)

    assert len(result) == 1
    assert result[0]["track_id"] == 7


def test_expired_track_removed():
    guard = TrackLifecycleGuard(
        max_hidden_seconds=0.75,
        min_confirmed_observations=1,
    )

    guard.update([track(2)], 0.0)
    assert guard.update([], 0.8) == []
    assert guard.retained_tracks == 0


def test_unknown_prediction_is_ignored():
    guard = TrackLifecycleGuard()

    assert guard.update([track(99, False)], 0.0) == []
    assert guard.retained_tracks == 0


def test_duplicate_track_rejected():
    import pytest

    guard = TrackLifecycleGuard()

    with pytest.raises(ValueError):
        guard.update(
            [track(1), track(1)],
            0.0,
        )
