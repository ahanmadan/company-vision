
from datetime import UTC

import cv2
import numpy as np
import pytest

from company_vision.camera.webcam_source import (
    WebcamSource,
    WebcamSourceError,
)
from company_vision.core.schemas import FramePacket


class FakeCapture:
    def __init__(self, opened=True, frames=None):
        self.opened = opened
        self.frames = list(frames or [])
        self.released = False
        self.settings = {}

    def isOpened(self):
        return self.opened and not self.released

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def get(self, prop):
        values = {
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 30.0,
        }
        return values.get(prop, 0)

    def set(self, prop, value):
        self.settings[prop] = value
        return True

    def release(self):
        self.released = True
        self.opened = False


@pytest.fixture
def camera_factory(monkeypatch):
    created = []

    def install(*, opened=True, frames=None):
        def factory(index, backend):
            capture = FakeCapture(
                opened=opened,
                frames=frames,
            )
            created.append((index, backend, capture))
            return capture

        monkeypatch.setattr(
            cv2,
            "VideoCapture",
            factory,
        )
        return created

    return install


def test_01_open_webcam(camera_factory):
    camera_factory()
    with WebcamSource() as source:
        assert source.is_open


def test_02_camera_index_passed(camera_factory):
    created = camera_factory()

    with WebcamSource(camera_index=2):
        pass

    assert created[0][0] == 2


def test_03_invalid_index():
    with pytest.raises(ValueError):
        WebcamSource(camera_index=-1)


def test_04_blank_camera_id():
    with pytest.raises(ValueError):
        WebcamSource(camera_id="   ")


def test_05_open_failure(camera_factory):
    created = camera_factory(opened=False)

    with pytest.raises(WebcamSourceError), WebcamSource():
        pass

    assert created[0][2].released


def test_06_frame_packet(camera_factory):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    camera_factory(frames=[frame])

    with WebcamSource(camera_id="cam_01") as source:
        packet = source.read()

    assert isinstance(packet, FramePacket)
    assert packet.camera_id == "cam_01"
    assert packet.image.shape == (480, 640, 3)


def test_07_sequential_frame_ids(camera_factory):
    frames = [
        np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(3)
    ]
    camera_factory(frames=frames)

    with WebcamSource() as source:
        ids = [source.read().frame_id for _ in range(3)]
        assert source.frames_read == 3

    assert ids == [0, 1, 2]


def test_08_utc_timestamp(camera_factory):
    camera_factory(
        frames=[np.zeros((480, 640, 3), dtype=np.uint8)]
    )

    with WebcamSource() as source:
        packet = source.read()

    assert packet.timestamp.utcoffset().total_seconds() == 0
    assert packet.timestamp.tzinfo == UTC

def test_09_read_failure(camera_factory):
    camera_factory(frames=[])

    with WebcamSource() as source, pytest.raises(WebcamSourceError):
        source.read()

    assert source.frames_read == 0


def test_10_read_before_open():
    source = WebcamSource()

    with pytest.raises(WebcamSourceError):
        source.read()


def test_11_close_releases_camera(camera_factory):
    created = camera_factory()
    source = WebcamSource()

    source.open()
    source.close()

    assert not source.is_open
    assert created[0][2].released


def test_12_reopen_resets_frame_id(camera_factory):
    created = camera_factory(
        frames=[np.zeros((480, 640, 3), dtype=np.uint8)]
    )

    source = WebcamSource()

    with source:
        assert source.read().frame_id == 0

    with source:
        assert source.read().frame_id == 0

    assert len(created) == 2


def test_13_requested_resolution(camera_factory):
    created = camera_factory()

    with WebcamSource(width=800, height=600):
        pass

    capture = created[0][2]
    assert capture.settings[cv2.CAP_PROP_FRAME_WIDTH] == 800
    assert capture.settings[cv2.CAP_PROP_FRAME_HEIGHT] == 600


def test_14_double_open_rejected(camera_factory):
    camera_factory()

    with WebcamSource() as source, pytest.raises(WebcamSourceError):
        source.open()


def test_15_invalid_frame_rejected(camera_factory):
    gray = np.zeros((480, 640), dtype=np.uint8)
    camera_factory(frames=[gray])

    with WebcamSource() as source, pytest.raises(WebcamSourceError):
        source.read()

    assert source.frames_read == 0


def test_16_close_is_idempotent(camera_factory):
    camera_factory()
    source = WebcamSource()

    source.open()
    source.close()
    source.close()

    assert not source.is_open
