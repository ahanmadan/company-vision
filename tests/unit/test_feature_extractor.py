from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from company_vision.behaviors.feature_extractor import (
    FeatureExtractor,
)
from company_vision.behaviors.schemas import (
    BehaviorFrame,
    BehaviorKeypoint,
    BehaviorObject,
    BehaviorPerson,
)
from company_vision.core.schemas import BoundingBox


def make_person(track_id=1, offset=0, weights=None):

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

    keypoints = {
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
        keypoints=keypoints,
        keypoint_weights=(
            weights if weights is not None
            else {name: 0.8 for name in names}
        ),
        observed=True,
        predicted=False,
        behavior_eligible=True,
    )


def make_frame(persons=None, objects=None):

    persons = persons or []
    objects = objects or []

    frame_width = max(
        1000,
        max(
            (int(person.bbox.x2) + 10 for person in persons),
            default=1000,
        ),
    )

    return BehaviorFrame(
        camera_id="camera_01",
        session_id="session_01",
        frame_id=100,
        timestamp_utc=datetime.now(UTC),
        monotonic_timestamp=100.5,
        width=frame_width,
        height=600,
        persons=persons,
        objects=objects,
        object_detection_status=(
            "available" if objects else "unknown"
        ),
        source="tracking_output",
        source_status="ok",
    )


def get_point(person, name):
    return next(
        point for point in person.keypoints
        if point.name == name
    )


def test_basic_frame_extraction():

    result = FeatureExtractor().extract(
        make_frame([make_person()])
    )

    assert result.camera_id == "camera_01"
    assert result.session_id == "session_01"
    assert result.frame_id == 100
    assert len(result.persons) == 1


def test_identity_preservation():

    result = FeatureExtractor().extract(
        make_frame([make_person(track_id=77)])
    )

    assert result.persons[0].track_id == 77
    assert result.persons[0].behavior_eligible is True


def test_variable_person_count():

    persons = [
        make_person(track_id=index + 1, offset=index * 110)
        for index in range(12)
    ]

    result = FeatureExtractor().extract(
        make_frame(persons)
    )

    assert len(result.persons) == 12


def test_empty_frame():

    result = FeatureExtractor().extract(make_frame())

    assert result.persons == ()
    assert result.objects == ()


def test_bbox_normalization():

    result = FeatureExtractor().extract(
        make_frame([make_person()])
    )

    person = result.persons[0]

    assert person.bbox_width_norm == pytest.approx(0.08)
    assert person.bbox_height_norm == pytest.approx(100 / 600)


def test_keypoint_normalization():

    result = FeatureExtractor().extract(
        make_frame([make_person()])
    )

    nose = get_point(result.persons[0], "nose")

    assert nose.available is True
    assert nose.x_norm == pytest.approx(30 / 80)
    assert nose.y_norm == pytest.approx(10 / 100)


def test_zero_weight_is_not_usable():

    weights = {
        name: 0.8
        for name in (
            "nose", "left_eye", "right_eye",
            "left_ear", "right_ear",
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
        )
    }

    weights["left_ear"] = 0.0

    result = FeatureExtractor().extract(
        make_frame([make_person(weights=weights)])
    )

    ear = get_point(result.persons[0], "left_ear")

    assert ear.available is False
    assert ear.reason == "zero_quality"
    assert ear.weight == 0.0


def test_invalid_keypoint_is_not_usable():

    person = make_person()
    person.keypoints["nose"] = BehaviorKeypoint(
        x=50,
        y=20,
        confidence=0.9,
        valid=False,
    )

    result = FeatureExtractor().extract(make_frame([person]))

    nose = get_point(result.persons[0], "nose")

    assert nose.available is False
    assert nose.reason == "invalid_keypoint"


def test_missing_coordinates_are_not_usable():

    person = make_person()
    person.keypoints["nose"] = BehaviorKeypoint(
        x=None,
        y=None,
        confidence=0.9,
        valid=False,
    )

    result = FeatureExtractor().extract(make_frame([person]))

    nose = get_point(result.persons[0], "nose")

    assert nose.available is False


def test_empty_weight_map_uses_confidence():

    person = make_person(weights={})

    result = FeatureExtractor().extract(make_frame([person]))

    nose = get_point(result.persons[0], "nose")

    assert nose.available is True
    assert nose.weight is None
    assert nose.quality == pytest.approx(0.9)


def test_shoulder_geometry():

    result = FeatureExtractor().extract(
        make_frame([make_person()])
    )

    person = result.persons[0]

    assert person.shoulder_center is not None
    assert person.shoulder_width_norm == pytest.approx(0.25)
    assert person.shoulder_line is not None


def test_head_geometry():

    result = FeatureExtractor().extract(
        make_frame([make_person()])
    )

    person = result.persons[0]

    assert person.head_center is not None
    assert person.head_to_shoulder is not None


def test_arm_vectors():

    result = FeatureExtractor().extract(
        make_frame([make_person()])
    )

    person = result.persons[0]

    assert person.left_arm is not None
    assert person.right_arm is not None
    assert person.left_arm.length > 0
    assert person.right_arm.length > 0


def test_missing_shoulder_disables_dependent_features():

    person = make_person()

    person.keypoints["left_shoulder"] = BehaviorKeypoint(
        x=None,
        y=None,
        confidence=0.0,
        valid=False,
    )

    result = FeatureExtractor().extract(make_frame([person]))

    extracted = result.persons[0]

    assert extracted.shoulder_line is None
    assert extracted.left_arm is None
    assert extracted.right_arm is not None


def test_object_geometry():

    obj = BehaviorObject(
        object_id="phone_01",
        class_id=67,
        class_name="cell phone",
        bbox=BoundingBox(
            x1=100,
            y1=100,
            x2=150,
            y2=200,
        ),
        confidence=0.95,
    )

    result = FeatureExtractor().extract(
        make_frame(objects=[obj])
    )

    assert len(result.objects) == 1
    assert result.objects[0].class_name == "cell phone"
    assert result.objects[0].width_norm == pytest.approx(0.05)


def test_input_is_not_modified():

    frame = make_frame([make_person()])

    before = frame.model_dump(mode="python")

    FeatureExtractor().extract(frame)

    after = frame.model_dump(mode="python")

    assert before == after


def test_output_is_immutable():

    result = FeatureExtractor().extract(
        make_frame([make_person()])
    )

    with pytest.raises((AttributeError, TypeError)):
        result.persons[0].track_id = 999


def test_predicted_person_remains_ineligible():

    person = make_person()

    person.observed = False
    person.predicted = True
    person.behavior_eligible = False

    result = FeatureExtractor().extract(make_frame([person]))

    assert result.persons[0].predicted is True
    assert result.persons[0].behavior_eligible is False


def test_arbitrary_keypoint_names_are_preserved():

    person = make_person()

    person.keypoints["custom_joint"] = BehaviorKeypoint(
        x=55,
        y=50,
        confidence=0.8,
        valid=True,
    )

    person.keypoint_weights["custom_joint"] = 0.7

    result = FeatureExtractor().extract(make_frame([person]))

    custom = get_point(result.persons[0], "custom_joint")

    assert custom.available is True
    assert result.persons[0].total_keypoints == 10


def test_extractor_is_stateless():

    extractor = FeatureExtractor()

    first = extractor.extract(
        make_frame([make_person(track_id=1)])
    )

    second = extractor.extract(
        make_frame([make_person(track_id=2)])
    )

    assert first.persons[0].track_id == 1
    assert second.persons[0].track_id == 2


def test_wrong_input_type_is_rejected():

    with pytest.raises(TypeError):
        FeatureExtractor().extract({})


def test_zero_bbox_dimensions_are_rejected():

    with pytest.raises(ValidationError):
        BoundingBox(
            x1=20,
            y1=10,
            x2=20,
            y2=110,
        )

    with pytest.raises(ValidationError):
        BoundingBox(
            x1=20,
            y1=10,
            x2=100,
            y2=10,
        )

def test_quality_summary():

    result = FeatureExtractor().extract(
        make_frame([make_person()])
    )

    person = result.persons[0]

    assert person.total_keypoints == 9
    assert person.available_keypoints == 9
    assert person.unavailable_keypoints == 0
    assert person.mean_keypoint_quality == pytest.approx(0.8)