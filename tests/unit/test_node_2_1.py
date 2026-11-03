
from datetime import UTC, timedelta

import cv2
import numpy as np
import pytest

from company_vision.camera.video_source import (
    VideoFileSource,
    VideoSourceError,
)
from company_vision.core.schemas import (
    FramePacket,
)


@pytest.fixture
def sample_video(tmp_path):
    path = tmp_path / "test_video.avi"

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (160, 120),
    )

    if not writer.isOpened():
        pytest.fail(
            "MJPG VideoWriter is unavailable"
        )

    try:
        for frame_id in range(10):
            frame = np.zeros(
                (120, 160, 3),
                dtype=np.uint8,
            )

            frame[:, :, 1] = frame_id * 20
            writer.write(frame)
    finally:
        writer.release()

    return path


def test_01_open_valid_video(sample_video):
    with VideoFileSource(sample_video) as source:
        assert source.is_open
        assert source.width == 160
        assert source.height == 120


def test_02_missing_file_rejected(tmp_path):
    path = tmp_path / "missing.avi"

    with pytest.raises(FileNotFoundError), VideoFileSource(path):
        pass


def test_03_invalid_file_rejected(tmp_path):
    path = tmp_path / "invalid.avi"
    path.write_text("not a video")

    with pytest.raises(VideoSourceError), VideoFileSource(path):
        pass


def test_04_frame_packet_type(sample_video):
    with VideoFileSource(sample_video) as source:
        packet = source.read()

        assert isinstance(packet, FramePacket)
        assert packet.image.dtype == np.uint8
        assert packet.image.shape == (
            120, 160, 3
        )


def test_05_frame_ids_are_sequential(sample_video):
    with VideoFileSource(sample_video) as source:
        ids = []

        while True:
            packet = source.read()

            if packet is None:
                break

            ids.append(packet.frame_id)

        assert ids == list(range(10))
        assert source.frames_read == 10


def test_06_end_of_file(sample_video):
    with VideoFileSource(sample_video) as source:
        for _ in range(10):
            assert source.read() is not None

        assert source.read() is None
        assert source.is_eof
        assert source.read() is None


def test_07_camera_id_preserved(sample_video):
    with VideoFileSource(
        sample_video,
        camera_id="office_01",
    ) as source:
        packet = source.read()

        assert packet.camera_id == "office_01"


def test_08_utc_timestamp(sample_video):
    with VideoFileSource(sample_video) as source:
        packet = source.read()

        assert packet.timestamp.tzinfo is not None
        assert packet.timestamp.utcoffset() == (
            timedelta(0)
        )
        assert packet.timestamp.tzinfo == UTC


def test_09_close_releases_source(sample_video):
    source = VideoFileSource(sample_video)

    source.open()
    assert source.is_open

    source.close()
    assert not source.is_open

    with pytest.raises(VideoSourceError):
        source.read()


def test_10_read_before_open_rejected(sample_video):
    source = VideoFileSource(sample_video)

    with pytest.raises(VideoSourceError):
        source.read()


def test_11_double_open_rejected(sample_video):
    with VideoFileSource(sample_video) as source, pytest.raises(VideoSourceError):
        source.open()


def test_12_reopen_resets_frame_id(sample_video):
    source = VideoFileSource(sample_video)

    with source:
        first = source.read()
        assert first.frame_id == 0

    with source:
        first_again = source.read()
        assert first_again.frame_id == 0


def test_13_invalid_camera_id_rejected(sample_video):
    with pytest.raises(ValueError):
        VideoFileSource(
            sample_video,
            camera_id="   ",
        )
