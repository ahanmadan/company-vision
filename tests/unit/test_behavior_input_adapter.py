from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from company_vision.core.schemas import (
    BoundingBox,
    FrameResult,
    Keypoint,
    ObjectObservation,
    PersonObservation,
)

from company_vision.behaviors.input_adapter import (
    BehaviorInputAdapter,
    BehaviorInputError,
)


def make_frame(person_count=1):
    persons = []

    for index in range(person_count):
        offset = index * 100

        points = [
            Keypoint(
                name=name,
                x=20 + offset,
                y=20,
                confidence=0.9,
            )
            for name in (
                "nose",
                "left_eye",
                "right_eye",
                "left_ear",
                "right_ear",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
            )
        ]

        persons.append(
            PersonObservation(
                track_id=index + 1,
                bbox=BoundingBox(
                    x1=10 + offset,
                    y1=10,
                    x2=80 + offset,
                    y2=90,
                ),
                confidence=0.95,
                keypoints=points,
            )
        )

    return FrameResult(
        camera_id="office_camera_01",
        frame_id=100,
        timestamp=datetime.now(UTC),
        width=person_count * 100 + 100,
        height=200,
        persons=persons,
        status="ok",
    )


def make_tracker_person(track_id=1):
    points = [
        [20.0 + index, 30.0 + index, 0.9]
        for index in range(9)
    ]

    return {
        "track_id": track_id,
        "bbox": [10, 10, 80, 90],
        "keypoints": points,
        "raw_keypoints": points,
        "keypoint_weights": [0.8] * 9,
        "confidence": 0.95,
        "observed": True,
        "predicted": False,
        "behavior_eligible": True,
        "track_lifecycle": "observed",
        "missed": 0,
    }


def test_frame_result_conversion_preserves_identity():
    adapter = BehaviorInputAdapter()

    result = adapter.from_frame_result(
        make_frame(),
        session_id="session_001",
        monotonic_timestamp=100.5,
        track_metadata={
            1: {
                "observed": True,
                "predicted": False,
                "behavior_eligible": True,
                "track_lifecycle": "observed",
                "keypoint_weights": [0.8] * 9,
            }
        },
    )

    assert result.camera_id == "office_camera_01"
    assert result.session_id == "session_001"
    assert result.persons[0].track_id == 1
    assert result.persons[0].behavior_eligible is True
    assert len(result.persons[0].keypoints) == 9
    assert len(result.persons[0].keypoint_weights) == 9


def test_variable_person_count_without_fixed_limit():
    adapter = BehaviorInputAdapter()

    result = adapter.from_frame_result(
        make_frame(7),
        session_id="session_002",
    )

    assert len(result.persons) == 7
    assert [p.track_id for p in result.persons] == list(range(1, 8))


def test_tracking_output_preserves_raw_and_filtered_points():
    adapter = BehaviorInputAdapter()

    result = adapter.from_tracking_output(
        camera_id="camera_01",
        session_id="session_003",
        frame_id=200,
        timestamp_utc=datetime.now(UTC),
        monotonic_timestamp=200.5,
        width=100,
        height=100,
        persons=[make_tracker_person()],
    )

    person = result.persons[0]

    assert person.track_id == 1
    assert person.observed is True
    assert person.predicted is False
    assert person.behavior_eligible is True
    assert len(person.raw_keypoints) == 9
    assert len(person.keypoint_weights) == 9


def test_predicted_person_is_preserved_but_not_eligible():
    adapter = BehaviorInputAdapter()
    person = make_tracker_person()

    person["observed"] = False
    person["predicted"] = True
    person["behavior_eligible"] = False

    result = adapter.from_tracking_output(
        camera_id="camera_01",
        session_id="session_004",
        frame_id=201,
        timestamp_utc=datetime.now(UTC),
        width=100,
        height=100,
        persons=[person],
    )

    assert result.persons[0].predicted is True
    assert result.persons[0].behavior_eligible is False


def test_empty_frame_is_valid():
    adapter = BehaviorInputAdapter()

    result = adapter.from_tracking_output(
        camera_id="camera_01",
        session_id="session_005",
        frame_id=202,
        timestamp_utc=datetime.now(UTC),
        width=100,
        height=100,
        persons=[],
    )

    assert result.persons == []
    assert result.object_detection_status == "not_run"


def test_duplicate_track_ids_are_rejected():
    adapter = BehaviorInputAdapter()

    with pytest.raises(ValidationError):
        adapter.from_tracking_output(
            camera_id="camera_01",
            session_id="session_006",
            frame_id=203,
            timestamp_utc=datetime.now(UTC),
            width=100,
            height=100,
            persons=[
                make_tracker_person(1),
                make_tracker_person(1),
            ],
        )


def test_out_of_frame_bbox_is_rejected():
    adapter = BehaviorInputAdapter()
    person = make_tracker_person()

    person["bbox"] = [10, 10, 150, 90]

    with pytest.raises(ValidationError):
        adapter.from_tracking_output(
            camera_id="camera_01",
            session_id="session_007",
            frame_id=204,
            timestamp_utc=datetime.now(UTC),
            width=100,
            height=100,
            persons=[person],
        )


def test_naive_timestamp_is_rejected():
    adapter = BehaviorInputAdapter()

    with pytest.raises(ValidationError):
        adapter.from_tracking_output(
            camera_id="camera_01",
            session_id="session_008",
            frame_id=205,
            timestamp_utc=datetime(2026, 11, 11, 10, 0, 0),
            width=100,
            height=100,
            persons=[],
        )


def test_unknown_track_metadata_is_rejected():
    adapter = BehaviorInputAdapter()

    with pytest.raises(BehaviorInputError):
        adapter.from_frame_result(
            make_frame(),
            session_id="session_009",
            track_metadata={
                1: {
                    "unknown_field": True,
                }
            },
        )


def test_metadata_for_unknown_track_is_rejected():
    adapter = BehaviorInputAdapter()

    with pytest.raises(BehaviorInputError):
        adapter.from_frame_result(
            make_frame(),
            session_id="session_010",
            track_metadata={
                999: {
                    "observed": True,
                }
            },
        )


def test_object_observations_are_preserved():
    adapter = BehaviorInputAdapter()

    frame = make_frame()

    frame.objects = [
        ObjectObservation(
            class_id=67,
            class_name="cell phone",
            bbox=BoundingBox(
                x1=30,
                y1=30,
                x2=50,
                y2=60,
            ),
            confidence=0.9,
        )
    ]

    result = adapter.from_frame_result(
        frame,
        session_id="session_011",
    )

    assert len(result.objects) == 1
    assert result.object_detection_status == "available"
    assert result.objects[0].class_name == "cell phone"


def test_eligible_person_requires_observed_track():
    adapter = BehaviorInputAdapter()
    person = make_tracker_person()

    person["observed"] = False
    person["predicted"] = True
    person["behavior_eligible"] = True

    with pytest.raises(ValidationError):
        adapter.from_tracking_output(
            camera_id="camera_01",
            session_id="session_012",
            frame_id=206,
            timestamp_utc=datetime.now(UTC),
            width=100,
            height=100,
            persons=[person],
        )
