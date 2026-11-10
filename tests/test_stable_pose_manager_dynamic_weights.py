import numpy as np
import pytest

from company_vision.vision.stable_pose_manager import (
    KEYPOINT_COUNT,
    KEYPOINT_NAMES,
    KEYPOINT_PRIORITY,
    StablePoseManager,
    Track,
)

# ============================================================
# TEST FIXTURES
# ============================================================

def make_box():
    return np.array(
        [[100, 50, 300, 450]],
        dtype=np.float32,
    )


def make_pose(confidence=0.95):
    pose = np.array(
        [
            [200, 80, confidence],
            [190, 75, confidence],
            [210, 75, confidence],
            [180, 85, confidence],
            [220, 85, confidence],
            [160, 150, confidence],
            [240, 150, confidence],
            [140, 230, confidence],
            [260, 230, confidence],
        ],
        dtype=np.float32,
    )

    assert pose.shape == (9, 3)

    return pose


def make_manager(**kwargs):
    config = {
        "confirm_hits": 1,
        "max_missed": 5,
        "min_keypoints": 3,
        "keypoint_conf": 0.25,
        "ema_alpha": 0.40,
    }

    config.update(kwargs)

    return StablePoseManager(**config)


# ============================================================
# CONTRACT TESTS
# ============================================================

def test_project_contract_has_exactly_nine_keypoints():
    assert KEYPOINT_COUNT == 9
    assert len(KEYPOINT_NAMES) == 9
    assert KEYPOINT_PRIORITY.shape == (9,)


def test_track_velocity_has_exactly_nine_points():
    track = Track(
        stable_id=1,
        state=np.zeros(8, dtype=np.float32),
        covariance=np.eye(8, dtype=np.float32),
        pose=np.zeros((9, 3), dtype=np.float32),
    )

    assert track.pose.shape == (9, 3)
    assert track.pose_velocity.shape == (9, 2)
    assert track.joint_weights.shape == (9,)


# ============================================================
# INPUT DIMENSION TESTS
# ============================================================

def test_accepts_nine_keypoint_input():
    manager = make_manager()

    output = manager.update(
        make_box(),
        make_pose()[None, :, :],
        np.array([0.95], dtype=np.float32),
    )

    assert len(output) == 1
    assert output[0]["keypoints"].shape == (9, 3)
    assert output[0]["raw_keypoints"].shape == (9, 3)
    assert output[0]["keypoint_weights"].shape == (9,)


def test_rejects_seventeen_keypoint_input():
    manager = make_manager()

    boxes = make_box()

    poses = np.zeros(
        (1, 17, 3),
        dtype=np.float32,
    )

    scores = np.array(
        [0.95],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="9, 3",
    ):
        manager.update(
            boxes,
            poses,
            scores,
        )


def test_rejects_eight_keypoint_input():
    manager = make_manager()

    poses = np.zeros(
        (1, 8, 3),
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        manager.update(
            make_box(),
            poses,
            np.array([0.95], dtype=np.float32),
        )


def test_accepts_empty_nine_keypoint_input():
    manager = make_manager()

    output = manager.update(
        np.empty((0, 4), dtype=np.float32),
        np.empty((0, 9, 3), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
    )

    assert output == []
    assert manager.frame_index == 1


# ============================================================
# OUTPUT DIMENSION TESTS
# ============================================================

def test_all_observed_output_arrays_have_nine_points():
    manager = make_manager()

    output = manager.update(
        make_box(),
        make_pose()[None, :, :],
        np.array([0.95], dtype=np.float32),
    )

    person = output[0]

    assert person["keypoints"].shape == (9, 3)
    assert person["raw_keypoints"].shape == (9, 3)
    assert person["keypoint_weights"].shape == (9,)

    assert person["observed"] is True
    assert person["predicted"] is False


def test_weights_are_finite_and_between_zero_and_one():
    manager = make_manager()

    output = manager.update(
        make_box(),
        make_pose()[None, :, :],
        np.array([0.95], dtype=np.float32),
    )

    weights = output[0]["keypoint_weights"]

    assert np.isfinite(weights).all()
    assert np.all(weights >= 0.0)
    assert np.all(weights <= 1.0)


# ============================================================
# DYNAMIC WEIGHT TESTS
# ============================================================

def test_shoulder_has_higher_priority_than_eye():
    manager = make_manager()

    output = manager.update(
        make_box(),
        make_pose()[None, :, :],
        np.array([0.95], dtype=np.float32),
    )

    weights = output[0]["keypoint_weights"]

    assert weights[5] > weights[1]
    assert weights[6] > weights[2]


def test_invalid_keypoint_has_zero_weight():
    manager = make_manager()

    pose = make_pose()

    pose[7, 2] = 0.10

    output = manager.update(
        make_box(),
        pose[None, :, :],
        np.array([0.95], dtype=np.float32),
    )

    weights = output[0]["keypoint_weights"]

    assert weights.shape == (9,)
    assert weights[7] == 0.0


def test_weight_changes_when_confidence_changes():
    manager = make_manager()

    first_pose = make_pose(0.95)

    first = manager.update(
        make_box(),
        first_pose[None, :, :],
        np.array([0.95], dtype=np.float32),
    )

    initial_weight = float(
        first[0]["keypoint_weights"][5]
    )

    second_pose = make_pose(0.30)

    second = manager.update(
        make_box(),
        second_pose[None, :, :],
        np.array([0.95], dtype=np.float32),
    )

    updated_weight = float(
        second[0]["keypoint_weights"][5]
    )

    assert updated_weight < initial_weight


def test_weighting_uses_only_nine_points():
    manager = make_manager()

    pose = make_pose()

    output = manager.update(
        make_box(),
        pose[None, :, :],
        np.array([0.95], dtype=np.float32),
    )

    weights = output[0]["keypoint_weights"]

    assert len(weights) == 9
    assert weights.shape == (9,)


# ============================================================
# TRACKING TESTS
# ============================================================

def test_tracking_preserves_nine_keypoint_dimensions():
    manager = make_manager()

    boxes = make_box()
    scores = np.array([0.95], dtype=np.float32)

    first = manager.update(
        boxes,
        make_pose()[None, :, :],
        scores,
    )

    second_pose = make_pose()

    second_pose[:, 0] += 2.0
    second_pose[:, 1] += 1.0

    second = manager.update(
        boxes,
        second_pose[None, :, :],
        scores,
    )

    assert len(first) == 1
    assert len(second) == 1

    assert first[0]["track_id"] == second[0]["track_id"]

    assert second[0]["keypoints"].shape == (9, 3)
    assert second[0]["raw_keypoints"].shape == (9, 3)
    assert second[0]["keypoint_weights"].shape == (9,)


def test_predicted_track_preserves_nine_point_dimensions():
    manager = make_manager(max_missed=5)

    manager.update(
        make_box(),
        make_pose()[None, :, :],
        np.array([0.95], dtype=np.float32),
    )

    output = manager.update(
        np.empty((0, 4), dtype=np.float32),
        np.empty((0, 9, 3), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
    )

    assert len(output) == 1

    person = output[0]

    assert person["predicted"] is True
    assert person["raw_keypoints"] is None
    assert person["keypoints"].shape == (9, 3)
    assert person["keypoint_weights"].shape == (9,)


# ============================================================
# VALIDATION TESTS
# ============================================================

def test_rejects_invalid_confidence():
    manager = make_manager()

    pose = make_pose()
    pose[0, 2] = 1.5

    with pytest.raises(ValueError):
        manager.update(
            make_box(),
            pose[None, :, :],
            np.array([0.95], dtype=np.float32),
        )


def test_rejects_non_finite_pose():
    manager = make_manager()

    pose = make_pose()
    pose[0, 0] = np.nan

    with pytest.raises(ValueError):
        manager.update(
            make_box(),
            pose[None, :, :],
            np.array([0.95], dtype=np.float32),
        )


def test_reset_clears_tracks_and_frame_index():
    manager = make_manager()

    manager.update(
        make_box(),
        make_pose()[None, :, :],
        np.array([0.95], dtype=np.float32),
    )

    assert len(manager.tracks) == 1
    assert manager.frame_index == 1

    manager.reset()

    assert len(manager.tracks) == 0
    assert manager.frame_index == 0
    assert manager.next_id == 1