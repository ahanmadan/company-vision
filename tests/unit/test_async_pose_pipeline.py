import time
from threading import Event

import numpy as np
import pytest

from company_vision.vision.async_pose_pipeline import (
    AsyncPosePipeline,
)


class FakeTensor:
    def __init__(self, data):
        self.data = np.asarray(data)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class FakeBoxes:
    def __init__(self):
        self.xyxy = FakeTensor([[10, 20, 100, 200]])
        self.conf = FakeTensor([0.95])

    def __len__(self):
        return 1


class FakeKeypoints:
    def __init__(self):
        self.data = FakeTensor(
            np.ones((1, 17, 3), dtype=np.float32)
        )


class FakeResult:
    def __init__(self):
        self.boxes = FakeBoxes()
        self.keypoints = FakeKeypoints()


class FakeModel:
    def __init__(self):
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [FakeResult()]


class SequencedBlockingModel:
    def __init__(self):
        self.first_entered = Event()
        self.second_entered = Event()
        self.release_first = Event()
        self.release_second = Event()
        self.calls = 0

    def predict(self, **kwargs):
        self.calls += 1

        if self.calls == 1:
            self.first_entered.set()
            if not self.release_first.wait(3):
                raise TimeoutError("First inference timed out")

        elif self.calls == 2:
            self.second_entered.set()
            if not self.release_second.wait(3):
                raise TimeoutError("Second inference timed out")

        return [FakeResult()]


class FailingModel:
    def predict(self, **kwargs):
        raise ValueError("Simulated inference failure")


def make_frame():
    return np.zeros((64, 64, 3), dtype=np.uint8)


def wait_for_result(pipeline, timeout=3):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        pipeline.check_error()

        result = pipeline.poll()
        if result is not None:
            return result

        time.sleep(0.005)

    raise AssertionError("No result received")


def test_async_inference():
    model = FakeModel()
    pipeline = AsyncPosePipeline(model)
    pipeline.start()

    try:
        captured = time.monotonic()
        pipeline.submit(1, make_frame(), captured)

        result = wait_for_result(pipeline)

        assert result.sequence == 1
        assert result.coordinates.shape == (1, 4)
        assert result.poses.shape == (1, 9, 3)
        assert result.scores.shape == (1,)
        assert result.inference_ms >= 0
        assert model.calls[0]["quantize"] == 16
        assert model.calls[0]["device"] == "cuda:0"

    finally:
        pipeline.stop()


def test_latest_frame_replacement():
    model = SequencedBlockingModel()
    pipeline = AsyncPosePipeline(model)
    pipeline.start()

    try:
        pipeline.submit(1, make_frame(), time.monotonic())

        assert model.first_entered.wait(timeout=2)

        for sequence in range(2, 11):
            pipeline.submit(
                sequence,
                make_frame(),
                time.monotonic(),
            )

        assert pipeline.stats()["dropped_input"] >= 8

        model.release_first.set()

        assert model.second_entered.wait(timeout=2)

        first = pipeline.poll()

        assert first is not None
        assert first.sequence == 1

        model.release_second.set()

        second = wait_for_result(pipeline)

        assert second.sequence == 10

    finally:
        model.release_first.set()
        model.release_second.set()
        pipeline.stop()


def test_invalid_frame():
    pipeline = AsyncPosePipeline(FakeModel())
    pipeline.start()

    try:
        with pytest.raises(ValueError):
            pipeline.submit(
                1,
                np.zeros((64, 64), dtype=np.uint8),
                time.monotonic(),
            )
    finally:
        pipeline.stop()


def test_worker_error_propagation():
    pipeline = AsyncPosePipeline(FailingModel())
    pipeline.start()

    pipeline.submit(1, make_frame(), time.monotonic())

    try:
        deadline = time.monotonic() + 3
        detected = False

        while time.monotonic() < deadline:
            try:
                pipeline.check_error()
            except RuntimeError as exc:
                assert isinstance(exc.__cause__, ValueError)
                detected = True
                break

            time.sleep(0.005)

        assert detected

    finally:
        with pytest.raises(RuntimeError):
            pipeline.stop()


def test_double_start_rejected():
    pipeline = AsyncPosePipeline(FakeModel())
    pipeline.start()

    try:
        with pytest.raises(RuntimeError):
            pipeline.start()
    finally:
        pipeline.stop()


def test_invalid_precision():
    with pytest.raises(ValueError):
        AsyncPosePipeline(
            FakeModel(),
            gpu_precision="invalid",
        )

