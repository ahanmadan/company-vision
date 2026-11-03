
from types import SimpleNamespace

import numpy as np
import pytest

from company_vision.core.schemas import FramePacket, current_utc
from company_vision.vision.pose_estimator import (
    COCO_KEYPOINT_NAMES,
    PoseEstimationError,
    PoseEstimator,
    PoseFrameResult,
)


def packet(image=None):
    if image is None:
        image = np.zeros((240, 320, 3), dtype=np.uint8)

    return FramePacket(
        camera_id="test_camera",
        frame_id=7,
        timestamp=current_utc(),
        image=image,
    )


def make_result(persons=1, point_dimensions=3):
    boxes = np.tile(
        np.array([[10, 20, 100, 200]], dtype=np.float32),
        (persons, 1),
    )
    scores = np.full(persons, 0.91, dtype=np.float32)
    points = np.zeros(
        (persons, 17, point_dimensions),
        dtype=np.float32,
    )

    points[:, :, 0] = 30
    points[:, :, 1] = 40

    if point_dimensions == 3:
        points[:, :, 2] = 0.85

    return SimpleNamespace(
        boxes=SimpleNamespace(xyxy=boxes, conf=scores),
        keypoints=SimpleNamespace(data=points),
    )


class FakePoseModel:
    task = "pose"

    def __init__(self, result):
        self.result = result
        self.last_kwargs = None

    def predict(self, **kwargs):
        self.last_kwargs = kwargs
        return [self.result]


def estimator(result=None, **kwargs):
    fake = FakePoseModel(
        result if result is not None else make_result()
    )

    model = PoseEstimator(
        model_factory=lambda path: fake,
        **kwargs,
    )
    return model, fake


def test_01_valid_pose_model():
    model, _ = estimator()
    assert model.device == "cuda:0"


def test_02_single_person():
    model, _ = estimator()
    output = model.predict(packet())

    assert isinstance(output, PoseFrameResult)
    assert output.person_count == 1
    assert output.packet.frame_id == 7


def test_03_seventeen_keypoints():
    model, _ = estimator()
    output = model.predict(packet())

    points = output.people[0].keypoints

    assert len(points) == 17
    assert tuple(point.name for point in points) == (
        COCO_KEYPOINT_NAMES
    )
    assert points[9].name == "left_wrist"
    assert points[10].name == "right_wrist"


def test_04_confidence_and_bbox():
    model, _ = estimator()
    person = model.predict(packet()).people[0]

    assert person.confidence == pytest.approx(0.91)
    assert person.bbox == (10.0, 20.0, 100.0, 200.0)
    assert person.keypoints[0].confidence == pytest.approx(
        0.85
    )


def test_05_no_person():
    model, _ = estimator(make_result(persons=0))
    output = model.predict(packet())

    assert output.person_count == 0
    assert output.people == ()


def test_06_two_people():
    model, _ = estimator(make_result(persons=2))
    assert model.predict(packet()).person_count == 2


def test_07_model_arguments():
    model, fake = estimator(
        confidence=0.4,
        image_size=640,
        device="cpu",
    )

    model.predict(packet())

    assert fake.last_kwargs["conf"] == 0.4
    assert fake.last_kwargs["device"] == "cpu"
    assert fake.last_kwargs["imgsz"] == 640
    assert fake.last_kwargs["save"] is False


def test_08_invalid_confidence():
    with pytest.raises(ValueError):
        estimator(confidence=0)


def test_09_invalid_image_size():
    with pytest.raises(ValueError):
        estimator(image_size=641)


def test_10_wrong_model_task():
    fake = FakePoseModel(make_result())
    fake.task = "detect"

    with pytest.raises(PoseEstimationError):
        PoseEstimator(model_factory=lambda path: fake)


def test_11_missing_keypoints():
    result = make_result()
    result.keypoints = None

    model, _ = estimator(result)

    with pytest.raises(PoseEstimationError):
        model.predict(packet())


def test_12_invalid_keypoint_shape():
    result = make_result()
    result.keypoints.data = np.zeros(
        (1, 16, 3),
        dtype=np.float32,
    )

    model, _ = estimator(result)

    with pytest.raises(PoseEstimationError):
        model.predict(packet())


def test_13_frame_identity_preserved():
    source = packet()
    model, _ = estimator()
    output = model.predict(source)

    assert output.packet is source
    assert output.packet.camera_id == "test_camera"


def test_14_inference_time_recorded():
    model, _ = estimator()
    output = model.predict(packet())

    assert output.inference_ms >= 0


def test_15_missing_model_path():
    with pytest.raises(ValueError):
        estimator(model_path="")


def test_16_two_coordinate_keypoints():
    model, _ = estimator(
        make_result(point_dimensions=2)
    )
    person = model.predict(packet()).people[0]

    assert len(person.keypoints) == 17
    assert person.keypoints[0].confidence is None
