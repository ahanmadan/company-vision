import numpy as np
import pytest

from company_vision.benchmarking.side_profile_audit import (
    analyze_pose_result,
    analyze_pose_result_with_confidence,
    calculate_box_iou,
    calculate_overlap_pairs,
)


def make_data():
    boxes = np.array([[100, 100, 300, 400]], dtype=np.float32)
    box_conf = np.array([0.85], dtype=np.float32)
    keypoints = np.ones((1, 17, 2), dtype=np.float32) * 150
    keypoint_conf = np.ones((1, 17), dtype=np.float32) * 0.9

    return boxes, box_conf, keypoints, keypoint_conf


def test_01_empty_detection():
    result = analyze_pose_result(
        np.empty((0, 4)),
        np.empty((0, 17, 2)),
        1920,
        1080,
    )
    assert result["person_candidates"] == 0
    assert result["persons"] == []


def test_02_single_person():
    boxes, _, keypoints, _ = make_data()

    result = analyze_pose_result(
        boxes, keypoints, 1920, 1080
    )

    assert result["person_candidates"] == 1
    assert len(result["persons"][0]["keypoints"]) == 17


def test_03_confidence_report():
    boxes, box_conf, keypoints, keypoint_conf = make_data()

    result = analyze_pose_result_with_confidence(
        boxes, box_conf, keypoints, keypoint_conf, 1920, 1080
    )

    assert result["accepted_persons"] == 1
    assert result["rejected_persons"] == 0


def test_04_low_confidence_person():
    boxes, box_conf, keypoints, keypoint_conf = make_data()
    box_conf[0] = 0.1

    result = analyze_pose_result_with_confidence(
        boxes, box_conf, keypoints, keypoint_conf, 1920, 1080
    )

    assert result["rejected_persons"] == 1


def test_05_low_confidence_keypoints():
    boxes, box_conf, keypoints, keypoint_conf = make_data()
    keypoint_conf[0, :5] = 0.1

    result = analyze_pose_result_with_confidence(
        boxes, box_conf, keypoints, keypoint_conf, 1920, 1080
    )

    assert result["persons"][0]["visible_keypoints"] == 12
    assert result["persons"][0]["low_confidence_keypoints"] == 5


def test_06_iou_identical_boxes():
    box = [10, 10, 100, 100]
    assert calculate_box_iou(box, box) == pytest.approx(1.0)


def test_07_iou_non_overlapping_boxes():
    a = [10, 10, 50, 50]
    b = [100, 100, 150, 150]

    assert calculate_box_iou(a, b) == 0


def test_08_partial_overlap():
    a = [0, 0, 100, 100]
    b = [50, 50, 150, 150]

    assert calculate_box_iou(a, b) == pytest.approx(1 / 7)


def test_09_overlap_pair_detection():
    persons = [
        {"detection_index": 0, "bbox": [0, 0, 100, 100]},
        {"detection_index": 1, "bbox": [50, 50, 150, 150]},
        {"detection_index": 2, "bbox": [300, 300, 400, 400]},
    ]

    pairs = calculate_overlap_pairs(persons)

    assert len(pairs) == 1
    assert pairs[0]["person_a"] == 0
    assert pairs[0]["person_b"] == 1


def test_10_invalid_image_dimensions():
    with pytest.raises(ValueError):
        analyze_pose_result(
            np.empty((0, 4)),
            np.empty((0, 17, 2)),
            0,
            1080,
        )


def test_11_invalid_box_shape():
    with pytest.raises(ValueError):
        analyze_pose_result(
            np.zeros((1, 5)),
            np.zeros((1, 17, 2)),
            1920,
            1080,
        )


def test_12_invalid_keypoint_shape():
    with pytest.raises(ValueError):
        analyze_pose_result(
            np.zeros((1, 4)),
            np.zeros((1, 16, 2)),
            1920,
            1080,
        )


def test_13_invalid_confidence():
    boxes, box_conf, keypoints, keypoint_conf = make_data()
    box_conf[0] = 1.5

    with pytest.raises(ValueError):
        analyze_pose_result_with_confidence(
            boxes, box_conf, keypoints, keypoint_conf, 1920, 1080
        )


def test_14_box_outside_image():
    with pytest.raises(ValueError):
        analyze_pose_result(
            np.array([[0, 0, 2000, 100]], dtype=np.float32),
            np.zeros((1, 17, 2)),
            1920,
            1080,
        )
