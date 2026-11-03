
from types import SimpleNamespace

import numpy as np
import pytest

from company_vision.core.schemas import FramePacket, current_utc
from company_vision.vision.object_detector import (
    COCO_PERSON_CLASS,
    COCO_PHONE_CLASS,
    DetectionFrameResult,
    ObjectDetectionError,
    ObjectDetector,
)

NAMES = {
    0: "person",
    56: "chair",
    63: "laptop",
    67: "cell phone",
}


def make_packet():
    return FramePacket(
        camera_id="test_cam",
        frame_id=12,
        timestamp=current_utc(),
        image=np.zeros((240, 320, 3), dtype=np.uint8),
    )


def make_result(
    class_ids=(0,),
    scores=None,
    boxes=None,
):
    count = len(class_ids)

    if boxes is None:
        boxes = np.tile(
            np.array([[10, 20, 120, 200]], dtype=np.float32),
            (count, 1),
        )

    if scores is None:
        scores = np.full(count, 0.9, dtype=np.float32)

    return SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=np.asarray(boxes, dtype=np.float32),
            conf=np.asarray(scores, dtype=np.float32),
            cls=np.asarray(class_ids, dtype=np.float32),
        ),
    )


class FakeDetectionModel:
    task = "detect"
    names = NAMES

    def __init__(self, result):
        self.result = result
        self.last_kwargs = None

    def predict(self, **kwargs):
        self.last_kwargs = kwargs
        return [self.result]


def make_detector(result=None, **kwargs):
    if result is None:
        result = make_result()

    fake = FakeDetectionModel(result)

    detector = ObjectDetector(
        model_factory=lambda path: fake,
        **kwargs,
    )

    return detector, fake


def test_01_valid_detector():
    detector, _ = make_detector()
    assert detector.device == "cuda:0"


def test_02_single_person():
    detector, _ = make_detector()
    output = detector.predict(make_packet())

    assert isinstance(output, DetectionFrameResult)
    assert output.object_count == 1
    assert output.objects[0].class_id == COCO_PERSON_CLASS
    assert output.objects[0].class_name == "person"


def test_03_person_and_phone():
    detector, _ = make_detector(
        make_result(class_ids=(0, 67))
    )
    output = detector.predict(make_packet())

    assert output.object_count == 2
    assert len(output.by_class(COCO_PHONE_CLASS)) == 1
    assert len(output.by_class(COCO_PERSON_CLASS)) == 1


def test_04_no_objects():
    detector, _ = make_detector(
        make_result(class_ids=())
    )
    output = detector.predict(make_packet())

    assert output.object_count == 0
    assert output.objects == ()


def test_05_confidence_and_bbox():
    detector, _ = make_detector()
    obj = detector.predict(make_packet()).objects[0]

    assert obj.confidence == pytest.approx(0.9)
    assert obj.bbox == (10.0, 20.0, 120.0, 200.0)


def test_06_class_filter():
    detector, fake = make_detector(
        make_result(class_ids=(0, 67)),
        class_ids={67},
    )

    output = detector.predict(make_packet())

    assert output.object_count == 1
    assert output.objects[0].class_id == 67
    assert fake.last_kwargs["classes"] == [67]


def test_07_bbox_clipping():
    detector, _ = make_detector(
        make_result(boxes=[[-10, -20, 500, 700]])
    )
    obj = detector.predict(make_packet()).objects[0]

    assert obj.bbox == (0.0, 0.0, 320.0, 240.0)


def test_08_invalid_confidence():
    with pytest.raises(ValueError):
        make_detector(confidence=0)


def test_09_invalid_image_size():
    with pytest.raises(ValueError):
        make_detector(image_size=641)


def test_10_wrong_model_task():
    fake = FakeDetectionModel(make_result())
    fake.task = "pose"

    with pytest.raises(ObjectDetectionError):
        ObjectDetector(model_factory=lambda path: fake)


def test_11_unknown_class():
    detector, _ = make_detector(
        make_result(class_ids=(999,))
    )

    with pytest.raises(ObjectDetectionError):
        detector.predict(make_packet())


def test_12_invalid_box_shape():
    result = make_result()
    result.boxes.xyxy = np.zeros((1, 5), dtype=np.float32)

    detector, _ = make_detector(result)

    with pytest.raises(ObjectDetectionError):
        detector.predict(make_packet())


def test_13_nonfinite_box():
    result = make_result()
    result.boxes.xyxy[0, 0] = float("nan")

    detector, _ = make_detector(result)

    with pytest.raises(ObjectDetectionError):
        detector.predict(make_packet())


def test_14_frame_identity_preserved():
    source = make_packet()
    detector, _ = make_detector()
    output = detector.predict(source)

    assert output.packet is source
    assert output.packet.frame_id == 12
    assert output.packet.camera_id == "test_cam"


def test_15_predict_parameters():
    detector, fake = make_detector(
        device="cpu",
        confidence=0.4,
        image_size=640,
    )

    detector.predict(make_packet())

    assert fake.last_kwargs["device"] == "cpu"
    assert fake.last_kwargs["conf"] == 0.4
    assert fake.last_kwargs["imgsz"] == 640
    assert fake.last_kwargs["save"] is False
    assert fake.last_kwargs["verbose"] is False


def test_16_inference_duration():
    detector, _ = make_detector()
    output = detector.predict(make_packet())

    assert output.inference_ms >= 0
