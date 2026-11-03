
from types import SimpleNamespace

import numpy as np
import pytest

from company_vision.core.schemas import FramePacket, current_utc
from company_vision.vision.person_tracker import (
    PersonTracker,
    PersonTrackingError,
    TrackingFrameResult,
)


def packet(frame_id=0, camera_id="cam_01"):
    return FramePacket(
        camera_id=camera_id,
        frame_id=frame_id,
        timestamp=current_utc(),
        image=np.zeros((240, 320, 3), dtype=np.uint8),
    )


def result(ids=(1,), classes=None):
    count = len(ids)

    if classes is None:
        classes = [0] * count

    coords = np.tile(
        np.array([[10, 20, 100, 200]], dtype=np.float32),
        (count, 1),
    )

    return SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=coords,
            conf=np.full(count, 0.9, dtype=np.float32),
            cls=np.asarray(classes, dtype=np.float32),
            id=np.asarray(ids, dtype=np.float32),
        )
    )


class FakeTrackingModel:
    task = "detect"

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def track(self, **kwargs):
        self.calls.append(kwargs)

        if not self.results:
            return [result(ids=())]

        return [self.results.pop(0)]


def make_tracker(results=None, **kwargs):
    if results is None:
        results = [result()]

    fake = FakeTrackingModel(results)

    tracker = PersonTracker(
        model_factory=lambda path: fake,
        **kwargs,
    )

    return tracker, fake


def test_01_tracker_initialization():
    tracker, _ = make_tracker()
    assert tracker.device == "cuda:0"


def test_02_single_person():
    tracker, _ = make_tracker()
    output = tracker.track(packet())

    assert isinstance(output, TrackingFrameResult)
    assert output.person_count == 1
    assert output.track_ids == (1,)


def test_03_two_people():
    tracker, _ = make_tracker([result(ids=(1, 2))])
    output = tracker.track(packet())

    assert output.person_count == 2
    assert output.track_ids == (1, 2)


def test_04_no_people():
    tracker, _ = make_tracker([result(ids=())])
    output = tracker.track(packet())

    assert output.person_count == 0


def test_05_track_id_stable_across_frames():
    tracker, _ = make_tracker([
        result(ids=(7,)),
        result(ids=(7,)),
        result(ids=(7,)),
    ])

    ids = [
        tracker.track(packet(i)).track_ids
        for i in range(3)
    ]

    assert ids == [(7,), (7,), (7,)]


def test_06_track_arguments():
    tracker, fake = make_tracker(
        device="cpu",
        confidence=0.2,
    )

    tracker.track(packet())

    kwargs = fake.calls[0]

    assert kwargs["persist"] is True
    assert kwargs["tracker"] == "bytetrack.yaml"
    assert kwargs["classes"] == [0]
    assert kwargs["device"] == "cpu"
    assert kwargs["conf"] == 0.2
    assert kwargs["save"] is False


def test_07_camera_change_rejected():
    tracker, _ = make_tracker([result()])

    tracker.track(packet(0, "cam_a"))

    with pytest.raises(PersonTrackingError):
        tracker.track(packet(1, "cam_b"))


def test_08_duplicate_frame_id_rejected():
    tracker, _ = make_tracker([result()])

    tracker.track(packet(0))

    with pytest.raises(PersonTrackingError):
        tracker.track(packet(0))


def test_09_out_of_order_frame_rejected():
    tracker, _ = make_tracker([result()])

    tracker.track(packet(5))

    with pytest.raises(PersonTrackingError):
        tracker.track(packet(3))


def test_10_duplicate_track_id_rejected():
    tracker, _ = make_tracker([
        result(ids=(4, 4))
    ])

    with pytest.raises(PersonTrackingError):
        tracker.track(packet())


def test_11_nonperson_class_rejected():
    tracker, _ = make_tracker([
        result(ids=(1,), classes=(67,))
    ])

    with pytest.raises(PersonTrackingError):
        tracker.track(packet())


def test_12_invalid_confidence():
    with pytest.raises(ValueError):
        make_tracker(confidence=0)


def test_13_invalid_image_size():
    with pytest.raises(ValueError):
        make_tracker(image_size=641)


def test_14_missing_tracking_ids():
    empty_ids = result(ids=(1,))
    empty_ids.boxes.id = None

    tracker, _ = make_tracker([empty_ids])
    output = tracker.track(packet())

    assert output.person_count == 0


def test_15_frame_identity_preserved():
    source = packet(10)
    tracker, _ = make_tracker()

    output = tracker.track(source)

    assert output.packet is source
    assert tracker.processed_frames == 1


def test_16_wrong_model_task():
    fake = FakeTrackingModel([result()])
    fake.task = "pose"

    with pytest.raises(PersonTrackingError):
        PersonTracker(model_factory=lambda path: fake)
