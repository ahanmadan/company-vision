
from collections import deque

import numpy as np
import pytest

from company_vision.camera.frame_ingestor import (
    EmptyPolicy,
    FrameIngestError,
    FrameIngestor,
    IngestStatus,
)
from company_vision.core.schemas import FramePacket, current_utc


def make_packet(
    frame_id=0,
    camera_id="test_cam",
    image=None,
):
    if image is None:
        image = np.zeros((32, 48, 3), dtype=np.uint8)

    return FramePacket(
        camera_id=camera_id,
        frame_id=frame_id,
        timestamp=current_utc(),
        image=image,
    )


def make_reader(values):
    items = deque(values)

    def read():
        if items:
            return items.popleft()
        return None

    return read


def test_01_file_frame_received():
    ingestor = FrameIngestor(
        make_reader([make_packet()]),
        empty_policy=EmptyPolicy.FILE,
    )

    result = ingestor.read_once()

    assert result.status is IngestStatus.FRAME
    assert result.packet is not None
    assert result.packet.frame_id == 0


def test_02_file_eof():
    ingestor = FrameIngestor(
        make_reader([make_packet(), None]),
        empty_policy=EmptyPolicy.FILE,
    )

    assert ingestor.read_once().status is IngestStatus.FRAME
    assert ingestor.read_once().status is IngestStatus.EOF
    assert ingestor.is_eof


def test_03_eof_is_persistent():
    calls = 0

    def reader():
        nonlocal calls
        calls += 1

    ingestor = FrameIngestor(
        reader,
        empty_policy=EmptyPolicy.FILE,
    )

    assert ingestor.read_once().status is IngestStatus.EOF
    assert ingestor.read_once().status is IngestStatus.EOF
    assert calls == 1


def test_04_live_empty_is_not_eof():
    ingestor = FrameIngestor(
        make_reader([None, make_packet()]),
        empty_policy=EmptyPolicy.LIVE,
    )

    assert ingestor.read_once().status is IngestStatus.EMPTY
    assert not ingestor.is_eof
    assert ingestor.read_once().status is IngestStatus.FRAME


def test_05_frame_id_gaps_are_allowed():
    ingestor = FrameIngestor(
        make_reader([
            make_packet(0),
            make_packet(4),
            make_packet(9),
        ]),
        empty_policy=EmptyPolicy.LIVE,
    )

    ids = []

    for _ in range(3):
        result = ingestor.read_once()
        assert result.packet is not None
        ids.append(result.packet.frame_id)

    assert ids == [0, 4, 9]


def test_06_duplicate_frame_rejected():
    ingestor = FrameIngestor(
        make_reader([make_packet(2), make_packet(2)]),
        empty_policy=EmptyPolicy.LIVE,
    )

    ingestor.read_once()

    with pytest.raises(FrameIngestError):
        ingestor.read_once()


def test_07_out_of_order_frame_rejected():
    ingestor = FrameIngestor(
        make_reader([make_packet(5), make_packet(3)]),
        empty_policy=EmptyPolicy.LIVE,
    )

    ingestor.read_once()

    with pytest.raises(FrameIngestError):
        ingestor.read_once()


def test_08_wrong_camera_rejected():
    ingestor = FrameIngestor(
        make_reader([make_packet(camera_id="wrong")]),
        empty_policy=EmptyPolicy.FILE,
        expected_camera_id="expected",
    )

    with pytest.raises(FrameIngestError):
        ingestor.read_once()


def test_09_camera_change_rejected():
    ingestor = FrameIngestor(
        make_reader([
            make_packet(0, "cam_a"),
            make_packet(1, "cam_b"),
        ]),
        empty_policy=EmptyPolicy.LIVE,
    )

    ingestor.read_once()

    with pytest.raises(FrameIngestError):
        ingestor.read_once()


def test_10_noncontiguous_image_normalized():
    original = np.zeros((32, 96, 3), dtype=np.uint8)
    noncontiguous = original[:, ::2, :]

    assert not noncontiguous.flags.c_contiguous

    ingestor = FrameIngestor(
        make_reader([make_packet(image=noncontiguous)]),
        empty_policy=EmptyPolicy.FILE,
    )

    result = ingestor.read_once()

    assert result.packet is not None
    assert result.packet.image.flags.c_contiguous
    assert result.packet.image.shape == (32, 48, 3)


def test_11_stats():
    ingestor = FrameIngestor(
        make_reader([
            None,
            make_packet(0),
            None,
            make_packet(1),
        ]),
        empty_policy=EmptyPolicy.LIVE,
    )

    for _ in range(4):
        ingestor.read_once()

    stats = ingestor.stats()

    assert stats.polls == 4
    assert stats.frames == 2
    assert stats.empty_polls == 2
    assert stats.last_frame_id == 1
    assert stats.camera_id == "test_cam"


def test_12_invalid_reader():
    with pytest.raises(TypeError):
        FrameIngestor(
            None,
            empty_policy=EmptyPolicy.FILE,
        )


def test_13_invalid_policy():
    with pytest.raises(TypeError):
        FrameIngestor(
            make_reader([]),
            empty_policy="file",
        )


def test_14_reader_error_propagates():
    def failing_reader():
        raise RuntimeError("Simulated source failure")

    ingestor = FrameIngestor(
        failing_reader,
        empty_policy=EmptyPolicy.LIVE,
    )

    with pytest.raises(RuntimeError):
        ingestor.read_once()
