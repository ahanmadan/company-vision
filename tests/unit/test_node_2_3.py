
from datetime import UTC

import cv2
import numpy as np
import pytest

from company_vision.camera.rtsp_source import (
    RTSPSource,
    RTSPSourceError,
)
from company_vision.core.schemas import FramePacket

TEST_URL = "rtsp://127.0.0.1:8554/test"


class FakeRTSPCapture:
    def __init__(
        self,
        opened=True,
        frames=None,
        metadata_error=False,
    ):
        self.opened = opened
        self.frames = list(frames or [])
        self.metadata_error = metadata_error
        self.released = False

    def isOpened(self):
        return self.opened and not self.released

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def get(self, prop):
        if self.metadata_error:
            raise RuntimeError("Metadata failure")

        values = {
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 25.0,
        }
        return values.get(prop, 0)

    def release(self):
        self.released = True
        self.opened = False


@pytest.fixture
def rtsp_factory(monkeypatch):
    created = []

    def install(
        *,
        opened=True,
        frames=None,
        metadata_error=False,
    ):
        def factory(url, backend, params):
            capture = FakeRTSPCapture(
                opened=opened,
                frames=frames,
                metadata_error=metadata_error,
            )
            created.append(
                {
                    "url": url,
                    "backend": backend,
                    "params": params,
                    "capture": capture,
                }
            )
            return capture

        monkeypatch.setattr(
            cv2,
            "VideoCapture",
            factory,
        )
        return created

    return install


def sample_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_01_valid_rtsp_url():
    source = RTSPSource(TEST_URL)
    assert source.camera_id == "office_rtsp_01"


def test_02_invalid_scheme():
    with pytest.raises(ValueError):
        RTSPSource("http://127.0.0.1/video")


def test_03_empty_url():
    with pytest.raises(ValueError):
        RTSPSource("")


def test_04_missing_host():
    with pytest.raises(ValueError):
        RTSPSource("rtsp:///stream")


def test_05_invalid_camera_id():
    with pytest.raises(ValueError):
        RTSPSource(TEST_URL, camera_id="   ")


def test_06_invalid_timeout():
    with pytest.raises(ValueError):
        RTSPSource(TEST_URL, open_timeout_ms=0)


def test_07_successful_open(rtsp_factory):
    rtsp_factory()

    with RTSPSource(TEST_URL) as source:
        assert source.is_open
        assert source.width == 640
        assert source.height == 480


def test_08_ffmpeg_backend_and_timeouts(rtsp_factory):
    created = rtsp_factory()

    with RTSPSource(
        TEST_URL,
        open_timeout_ms=3000,
        read_timeout_ms=4000,
    ):
        pass

    assert created[0]["backend"] == cv2.CAP_FFMPEG
    assert created[0]["params"] == [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
        3000,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
        4000,
    ]


def test_09_connection_failure(rtsp_factory):
    created = rtsp_factory(opened=False)

    with pytest.raises(RTSPSourceError), RTSPSource(TEST_URL):
        pass

    assert created[0]["capture"].released


def test_10_frame_packet(rtsp_factory):
    rtsp_factory(frames=[sample_frame()])

    with RTSPSource(TEST_URL, camera_id="office_01") as source:
        packet = source.read()

    assert isinstance(packet, FramePacket)
    assert packet.camera_id == "office_01"
    assert packet.frame_id == 0
    assert packet.image.shape == (480, 640, 3)


def test_11_sequential_ids(rtsp_factory):
    rtsp_factory(
        frames=[sample_frame() for _ in range(3)]
    )

    with RTSPSource(TEST_URL) as source:
        ids = [source.read().frame_id for _ in range(3)]
        assert source.frames_read == 3

    assert ids == [0, 1, 2]


def test_12_utc_timestamp(rtsp_factory):
    rtsp_factory(frames=[sample_frame()])

    with RTSPSource(TEST_URL) as source:
        packet = source.read()

    assert packet.timestamp.utcoffset().total_seconds() == 0
    assert packet.timestamp.tzinfo == UTC


def test_13_read_failure(rtsp_factory):
    rtsp_factory(frames=[])

    with RTSPSource(TEST_URL) as source, pytest.raises(RTSPSourceError):
        source.read()

    assert source.frames_read == 0


def test_14_invalid_frame(rtsp_factory):
    gray = np.zeros((480, 640), dtype=np.uint8)
    rtsp_factory(frames=[gray])

    with RTSPSource(TEST_URL) as source, pytest.raises(RTSPSourceError):
        source.read()

    assert source.frames_read == 0


def test_15_close_and_reopen(rtsp_factory):
    rtsp_factory(frames=[sample_frame()])
    source = RTSPSource(TEST_URL)

    with source:
        assert source.read().frame_id == 0

    assert not source.is_open

    with source:
        assert source.read().frame_id == 0


def test_16_metadata_failure_releases_capture(rtsp_factory):
    created = rtsp_factory(metadata_error=True)

    with pytest.raises(RTSPSourceError), RTSPSource(TEST_URL):
        pass

    assert created[0]["capture"].released


def test_17_read_before_open():
    source = RTSPSource(TEST_URL)

    with pytest.raises(RTSPSourceError):
        source.read()


def test_18_double_open(rtsp_factory):
    rtsp_factory()

    with RTSPSource(TEST_URL) as source, pytest.raises(RTSPSourceError):
        source.open()


def test_19_invalid_backend():
    with pytest.raises(ValueError):
        RTSPSource(TEST_URL, backend=cv2.CAP_ANY)


def test_20_close_idempotent(rtsp_factory):
    rtsp_factory()
    source = RTSPSource(TEST_URL)

    source.open()
    source.close()
    source.close()

    assert not source.is_open
