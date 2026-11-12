"""Four-fix helper tests, without camera or GPU."""

from types import SimpleNamespace

import numpy as np

from company_vision.behaviors.live_behavior_evidence import (
    diagnostic_rows,
    iou,
    match_wrists,
    result_objects,
)


def test_iou():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_iou_disjoint():
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_missing_wrist_source():
    assert match_wrists([], [], None, (480, 640, 3)) == {}


def test_matching_wrist_track():
    tracks = [{
        "track_id": 7,
        "bbox": [10, 10, 110, 210],
        "observed": True,
        "behavior_eligible": True,
    }]
    boxes = np.array([[10, 10, 110, 210]], dtype=np.float32)
    coco = np.zeros((1, 17, 3), dtype=np.float32)
    coco[0, 9] = [40, 80, 0.9]
    coco[0, 10] = [70, 90, 0.8]

    result = match_wrists(tracks, boxes, coco, (480, 640, 3))

    assert result[7]["left_wrist"] == [40.0, 80.0, 0.9] or (
        np.allclose(result[7]["left_wrist"], [40, 80, 0.9])
    )
    assert "right_wrist" in result[7]


def test_predicted_track_receives_no_wrists():
    tracks = [{
        "track_id": 7,
        "bbox": [10, 10, 110, 210],
        "observed": False,
        "behavior_eligible": False,
    }]
    boxes = np.array([[10, 10, 110, 210]], dtype=np.float32)
    coco = np.zeros((1, 17, 3), dtype=np.float32)

    assert match_wrists(tracks, boxes, coco, (480, 640, 3)) == {}


def test_empty_object_result():
    result = SimpleNamespace(boxes=None, names={})
    assert result_objects(result) == []


def test_empty_diagnostics():
    result = SimpleNamespace(results=[])
    assert diagnostic_rows(result) == {}