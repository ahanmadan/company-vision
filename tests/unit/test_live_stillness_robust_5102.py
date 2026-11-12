"""Behavior stillness jitter/temporal acceptance tests."""
import numpy as np

from company_vision.behaviors.live_behavior_enhancements import WeightedStillness


def person(shift=0.0, bbox_shift=0.0, track_id=7, valid=True):
    pose = np.array([[120 + i * 4 + shift, 130 + i * 5, .95]
                     for i in range(9)], dtype=float)
    if not valid:
        pose[:, 2] = 0.0
    return {"track_id": track_id, "bbox": [90+bbox_shift,90,190+bbox_shift,280],
            "keypoints": pose, "keypoint_weights": np.ones(9),
            "observed": True, "behavior_eligible": True}


def test_continuous_stillness_20_seconds():
    monitor = WeightedStillness()
    for i in range(241):
        t=i/12
        result=monitor.update([person(shift=.7*((i%3)-1))],t)[7]
    assert result["seconds"] >= 19.9
    assert result["state"] == "STILLNESS OBSERVED"


def test_continuous_stillness_60_seconds():
    monitor=WeightedStillness()
    for i in range(721):
        result=monitor.update([person(shift=.5*((i%3)-1))],i/12)[7]
    assert result["seconds"] >= 59.9
    assert result["state"] == "PROLONGED STILLNESS"


def test_isolated_large_bbox_jump_does_not_reset():
    monitor=WeightedStillness()
    for i in range(241):
        result=monitor.update([person(bbox_shift=22 if i==220 else 0)],i/12)[7]
    assert result["seconds"] >= 19.9


def test_single_pose_outlier_does_not_reset():
    monitor=WeightedStillness()
    for i in range(241):
        result=monitor.update([person(shift=30 if i==220 else 0)],i/12)[7]
    assert result["seconds"] >= 19.9


def test_real_sustained_displacement_resets():
    monitor=WeightedStillness()
    for i in range(120):
        monitor.update([person()],i/12)
    for i in range(120,130):
        result=monitor.update([person(shift=45)],i/12)[7]
    assert result["seconds"] < 1


def test_low_quality_is_unknown_not_stillness():
    monitor=WeightedStillness()
    result=monitor.update([person(valid=False)],0.0)[7]
    assert result["state"] == "UNKNOWN"


def test_missing_track_drops_clock():
    monitor=WeightedStillness()
    monitor.update([person()],0.0)
    monitor.update([],0.1)
    assert monitor.states == {}