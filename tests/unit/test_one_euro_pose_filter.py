import numpy as np

from company_vision.vision.one_euro_pose_filter import OneEuroPoseFilter
from company_vision.vision.stable_pose_manager import StablePoseManager


def make_pose(x=100.0, confidence=0.95):
    pose = np.zeros((9, 3), dtype=np.float32)
    pose[:, 0] = x
    pose[:, 1] = 150.0
    pose[:, 2] = confidence
    return pose


def test_constant_pose_remains_constant():
    f = OneEuroPoseFilter()
    pose = make_pose()
    for _ in range(30):
        result = f.update(pose)
    np.testing.assert_allclose(result, pose, atol=1e-4)


def test_jitter_is_reduced():
    rng = np.random.default_rng(42)
    f = OneEuroPoseFilter()
    measurements = 100 + rng.normal(0, 3, 120)
    filtered = []

    for x in measurements:
        filtered.append(float(f.update(make_pose(x))[0, 0]))

    assert np.std(filtered[20:]) < np.std(measurements[20:])


def test_fast_motion_is_followed():
    f = OneEuroPoseFilter()
    f.update(make_pose(100))
    for x in range(105, 205, 5):
        result = f.update(make_pose(x))

    assert float(result[0, 0]) > 170


def test_low_confidence_does_not_move_position():
    f = OneEuroPoseFilter()
    f.update(make_pose(100))
    result = f.update(make_pose(500, confidence=0.1))

    assert result[0, 0] == 100
    assert result[0, 2] == 0.1


def test_manager_supports_both_filter_modes():
    for mode in ("one_euro", "dynamic"):
        manager = StablePoseManager(
            confirm_hits=1,
            pose_filter_mode=mode,
        )
        box = np.array([[50, 50, 250, 450]], dtype=np.float32)
        pose = make_pose()[None, :, :]
        score = np.array([0.95], dtype=np.float32)

        result = manager.update(box, pose, score)

        assert len(result) == 1
        assert result[0]["keypoints"].shape == (9, 3)
        assert result[0]["track_id"] == 1

def test_variable_dt_changes_filter_response():
    pose_start = make_pose(100)
    pose_next = make_pose(120)

    fast = OneEuroPoseFilter(fps=25)
    slow = OneEuroPoseFilter(fps=25)

    fast.update(pose_start)
    slow.update(pose_start)

    fast_result = fast.update(pose_next, dt=0.01)
    slow_result = slow.update(pose_next, dt=0.10)

    assert slow_result[0, 0] > fast_result[0, 0]


def test_invalid_dt_rejected():
    import pytest

    f = OneEuroPoseFilter()

    for dt in (0, -0.01, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            f.update(make_pose(), dt=dt)


def test_manager_forwards_dt_to_filter():
    manager = StablePoseManager(
        confirm_hits=1,
        pose_filter_mode="one_euro",
    )

    box = np.array([[50, 50, 250, 450]], dtype=np.float32)
    score = np.array([0.95], dtype=np.float32)

    manager.update(box, make_pose(100)[None], score, dt=0.04)

    result = manager.update(
        box,
        make_pose(120)[None],
        score,
        dt=0.10,
    )

    assert len(result) == 1
    assert result[0]["keypoints"][0, 0] > 100


def test_default_manager_uses_legacy_dynamic_filter():
    manager = StablePoseManager(confirm_hits=1)

    assert manager.pose_filter_mode == "dynamic"


def test_manager_rejects_invalid_dt():
    import pytest

    manager = StablePoseManager(confirm_hits=1)

    box = np.array([[50, 50, 250, 450]], dtype=np.float32)
    score = np.array([0.95], dtype=np.float32)

    with pytest.raises(ValueError):
        manager.update(
            box,
            make_pose()[None],
            score,
            dt=0,
        )
