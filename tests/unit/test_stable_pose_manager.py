import numpy as np
import pytest

from company_vision.vision.stable_pose_manager import (
    StablePoseManager,
)


def sample_detection(x=100):
    box = np.array(
        [[x, 100, x+180, 500]],
        dtype=np.float32,
    )

    pose = np.zeros((1, 17, 3), dtype=np.float32)

    for i in range(17):
        pose[0, i] = [x+50+i, 150+i*10, 0.95]

    scores = np.array([0.95], dtype=np.float32)

    return box, pose, scores


def test_new_track_requires_confirmation():
    manager = StablePoseManager(confirm_hits=3)

    boxes, poses, scores = sample_detection()

    assert manager.update(boxes, poses, scores) == []
    assert manager.update(boxes, poses, scores) == []

    result = manager.update(boxes, poses, scores)

    assert len(result) == 1
    assert result[0]["track_id"] == 1
    assert result[0]["observed"] is True


def test_stable_id_survives_small_movement():
    manager = StablePoseManager(confirm_hits=1)

    boxes, poses, scores = sample_detection()

    first = manager.update(boxes, poses, scores)

    boxes2, poses2, scores2 = sample_detection(105)

    second = manager.update(boxes2, poses2, scores2)

    assert first[0]["track_id"] == second[0]["track_id"]


def test_track_survives_short_occlusion():
    manager = StablePoseManager(
        confirm_hits=1,
        max_missed=3,
    )

    boxes, poses, scores = sample_detection()

    first = manager.update(boxes, poses, scores)

    missing = manager.update(
        np.empty((0, 4)),
        np.empty((0, 17, 3)),
        np.empty((0,)),
    )

    assert len(missing) == 1
    assert missing[0]["track_id"] == first[0]["track_id"]
    assert missing[0]["predicted"] is True
    assert missing[0]["observed"] is False


def test_expired_track_is_removed():
    manager = StablePoseManager(
        confirm_hits=1,
        max_missed=2,
    )

    boxes, poses, scores = sample_detection()
    manager.update(boxes, poses, scores)

    for _ in range(3):
        manager.update(
            np.empty((0, 4)),
            np.empty((0, 17, 3)),
            np.empty((0,)),
        )

    assert len(manager.tracks) == 0


def test_invalid_box_rejected():
    manager = StablePoseManager()

    with pytest.raises(ValueError):
        manager.update(
            np.array([[100, 100, 50, 50]]),
            np.zeros((1, 17, 3)),
            np.array([0.9]),
        )


def test_invalid_pose_shape_rejected():
    manager = StablePoseManager()

    with pytest.raises(ValueError):
        manager.update(
            np.array([[10, 10, 100, 100]]),
            np.zeros((1, 16, 3)),
            np.array([0.9]),
        )


def test_multiple_people_keep_separate_ids():
    manager = StablePoseManager(confirm_hits=1)

    boxes = np.array([
        [100, 100, 280, 500],
        [400, 100, 580, 500],
        [700, 100, 880, 500],
    ], dtype=np.float32)

    poses = np.zeros((3, 17, 3), dtype=np.float32)

    for person in range(3):
        for i in range(17):
            poses[person, i] = [
                boxes[person, 0]+50+i,
                150+i*10,
                0.95,
            ]

    result = manager.update(
        boxes,
        poses,
        np.array([0.95, 0.94, 0.93]),
    )

    assert len(result) == 3
    assert len({item["track_id"] for item in result}) == 3


def test_low_keypoint_detection_is_rejected():
    manager = StablePoseManager(
        confirm_hits=1,
        min_keypoints=3,
    )

    boxes, poses, scores = sample_detection()
    poses[0, :, 2] = 0.0
    poses[0, :2, 2] = 0.95

    assert manager.update(boxes, poses, scores) == []


def test_reset_clears_tracking_state():
    manager = StablePoseManager(confirm_hits=1)

    boxes, poses, scores = sample_detection()
    manager.update(boxes, poses, scores)

    manager.reset()

    assert len(manager.tracks) == 0
    assert manager.frame_index == 0


def test_missing_keypoint_does_not_become_observed():
    manager = StablePoseManager(confirm_hits=1)

    boxes, poses, scores = sample_detection()
    manager.update(boxes, poses, scores)

    poses[0, 0, 2] = 0.0

    result = manager.update(boxes, poses, scores)

    assert result[0]["keypoints"][0, 2] < 0.25

def test_confirmed_track_recovers_same_id_after_occlusion():
    manager = StablePoseManager(
        confirm_hits=1,
        max_missed=5,
    )

    boxes, poses, scores = sample_detection(100)
    first = manager.update(boxes, poses, scores)
    original_id = first[0]["track_id"]

    empty = manager.update(
        np.empty((0, 4), dtype=np.float32),
        np.empty((0, 17, 3), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
    )

    assert empty[0]["predicted"] is True

    boxes, poses, scores = sample_detection(102)
    recovered = manager.update(boxes, poses, scores)

    assert len(recovered) == 1
    assert recovered[0]["track_id"] == original_id
    assert recovered[0]["observed"] is True


def test_multiple_track_ids_survive_detection_order_change():
    manager = StablePoseManager(confirm_hits=1)

    boxes = np.array([
        [100, 100, 280, 500],
        [500, 100, 680, 500],
    ], dtype=np.float32)

    poses = np.zeros((2, 17, 3), dtype=np.float32)

    for person in range(2):
        for joint in range(17):
            poses[person, joint] = [
                boxes[person, 0] + 50 + joint,
                150 + joint * 10,
                0.95,
            ]

    first = manager.update(
        boxes,
        poses,
        np.array([0.95, 0.95], dtype=np.float32),
    )

    original_ids = {
        round(float(item["bbox"][0])): item["track_id"]
        for item in first
    }

    second = manager.update(
        boxes[::-1].copy(),
        poses[::-1].copy(),
        np.array([0.95, 0.95], dtype=np.float32),
    )

    recovered_ids = {
        round(float(item["bbox"][0])): item["track_id"]
        for item in second
    }

    assert recovered_ids == original_ids

