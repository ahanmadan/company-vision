"""Node 5.9.1: PoseTrackingBridge contract tests."""

from datetime import UTC, datetime

import numpy as np
import pytest

from company_vision.behaviors.plugin_registry import BehaviorPluginContext
from company_vision.behaviors.pose_tracking_bridge import (
    PoseBridgeError,
    PoseTrackingBridge,
)


def tracked(
    track_id=1,
    observed=True,
    eligible=True,
    x=50.0,
):
    bbox = np.array([x, 10, x + 40, 100], dtype=np.float32)

    points = np.array([
        [x + 20, 20 + index * 6, 0.9]
        for index in range(9)
    ], dtype=np.float32)

    return {
        "track_id": track_id,
        "bbox": bbox,
        "keypoints": points,
        "raw_keypoints": points.copy() if observed else None,
        "keypoint_weights": np.ones(9, dtype=np.float32) * 0.85,
        "confidence": 0.95,
        "observed": observed,
        "predicted": not observed,
        "behavior_eligible": eligible if observed else False,
        "track_lifecycle": "observed" if observed else "predicted",
        "missed": 0 if observed else 1,
    }


def options(**changes):
    result = {
        "camera_id": "office",
        "session_id": "session-1",
        "frame_id": 10,
        "timestamp_utc": datetime.now(UTC),
        "monotonic_timestamp": 10.0,
        "width": 640,
        "height": 480,
        "tracked_persons": [tracked()],
    }
    result.update(changes)
    return result


def test_nine_keypoints_preserved():
    result = PoseTrackingBridge().build(**options())
    person = result.persons[0]

    assert len(person.keypoints) == 9
    assert len(person.raw_keypoints) == 9
    assert person.track_id == 1
    assert person.keypoint_weights["nose"] == pytest.approx(0.85)


def test_multi_person_dynamic_count():
    people = [tracked(i, x=30.0 * i) for i in range(1, 7)]
    result = PoseTrackingBridge().build(
        **options(tracked_persons=people)
    )
    assert len(result.persons) == 6


def test_predicted_track_ineligible():
    result = PoseTrackingBridge().build(
        **options(tracked_persons=[
            tracked(observed=False, eligible=False)
        ])
    )
    assert result.persons[0].predicted is True
    assert result.persons[0].behavior_eligible is False


def test_unconfirmed_observed_track_remains_ineligible():
    result = PoseTrackingBridge().build(
        **options(tracked_persons=[
            tracked(eligible=False)
        ])
    )
    assert result.persons[0].behavior_eligible is False


def test_missing_object_detection_not_faked():
    result = PoseTrackingBridge().build(**options())
    assert result.object_detection_status == "not_run"
    assert result.objects == []


def test_empty_successful_object_detection():
    result = PoseTrackingBridge().build(
        **options(objects=[])
    )
    assert result.object_detection_status == "available"
    assert result.objects == []


def test_real_object_record_preserved():
    obj = {
        "object_id": 7,
        "class_id": 67,
        "class_name": "cell phone",
        "confidence": 0.94,
        "bbox": [60, 55, 75, 85],
    }

    result = PoseTrackingBridge().build(
        **options(objects=[obj])
    )
    assert result.objects[0].class_id == 67
    assert result.objects[0].confidence == pytest.approx(0.94)


def test_valid_matched_wrist_supplement():
    result = PoseTrackingBridge().build(**options(
        supplemental_wrists={
            1: {"right_wrist": [82, 88, 0.96]}
        },
    ))
    assert "right_wrist" in result.persons[0].keypoints
    assert len(result.persons[0].raw_keypoints) == 9


def test_unknown_track_supplement_rejected():
    with pytest.raises(PoseBridgeError):
        PoseTrackingBridge().build(**options(
            supplemental_wrists={
                999: {"right_wrist": [80, 80, 0.9]}
            },
        ))


def test_predicted_track_cannot_gain_wrist():
    with pytest.raises(PoseBridgeError):
        PoseTrackingBridge().build(**options(
            tracked_persons=[tracked(observed=False, eligible=False)],
            supplemental_wrists={
                1: {"right_wrist": [80, 80, 0.9]}
            },
        ))


def test_low_confidence_landmark_is_invalid():
    person = tracked()
    person["keypoints"][0, 2] = 0.05

    result = PoseTrackingBridge().build(
        **options(tracked_persons=[person])
    )

    assert result.persons[0].keypoints["nose"].valid is False


def test_outside_landmark_invalid_not_clamped():
    person = tracked()
    person["keypoints"][0, 0] = 9999

    result = PoseTrackingBridge().build(
        **options(tracked_persons=[person])
    )

    assert result.persons[0].keypoints["nose"].valid is False


def test_duplicate_tracks_rejected():
    with pytest.raises(PoseBridgeError):
        PoseTrackingBridge().build(**options(
            tracked_persons=[tracked(1), tracked(1)]
        ))


def test_unusable_bbox_rejected():
    person = tracked()
    person["bbox"] = np.array([-100, 10, -50, 100])

    with pytest.raises(PoseBridgeError):
        PoseTrackingBridge().build(
            **options(tracked_persons=[person])
        )


def test_features_ready_for_plugin_context():
    features = PoseTrackingBridge().extract(**options())
    context = BehaviorPluginContext(frame=features)

    assert context.frame.persons[0].track_id == 1
    assert context.frame.persons[0].behavior_eligible is True
    assert context.frame.persons[0].available_keypoints == 9


def test_empty_person_frame():
    result = PoseTrackingBridge().build(
        **options(tracked_persons=[])
    )
    assert result.persons == []


def test_invalid_timestamp_rejected():
    with pytest.raises(PoseBridgeError):
        PoseTrackingBridge().build(**options(
            timestamp_utc=datetime(2026, 11, 12)
        ))


def test_deterministic_conversion():
    bridge = PoseTrackingBridge()
    data = options()

    a = bridge.build(**data)
    b = bridge.build(**data)

    assert a.model_dump() == b.model_dump()


def test_object_source_collision():
    with pytest.raises(PoseBridgeError):
        PoseTrackingBridge().build(**options(
            objects=[],
            yolo_object_result=object(),
        ))


def test_invalid_keypoint_shape():
    person = tracked()
    person["keypoints"] = np.zeros((17, 3))

    with pytest.raises(PoseBridgeError):
        PoseTrackingBridge().build(
            **options(tracked_persons=[person])
        )


def test_unknown_wrist_name_rejected():
    with pytest.raises(PoseBridgeError):
        PoseTrackingBridge().build(**options(
            supplemental_wrists={
                1: {"left_knee": [80, 80, 0.9]}
            },
        ))