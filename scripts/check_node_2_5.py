
import time
from pathlib import Path

import numpy as np

from company_vision.camera.buffered_source import (
    BufferedReconnectingSource,
)
from company_vision.camera.frame_ingestor import (
    EmptyPolicy,
    FrameIngestor,
    IngestStatus,
)
from company_vision.camera.video_source import VideoFileSource
from company_vision.core.schemas import FramePacket, current_utc

VIDEO_PATH = Path("data/samples/node_2_1_sample.avi")


class SyntheticLiveSource:
    def __init__(self) -> None:
        self.opened = False
        self.counter = 0

    def open(self) -> None:
        self.opened = True

    def read(self) -> FramePacket:
        if not self.opened:
            raise RuntimeError("Source not open")

        time.sleep(0.01)

        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[:, :, 1] = self.counter % 256

        packet = FramePacket(
            camera_id="synthetic_cam",
            frame_id=self.counter,
            timestamp=current_utc(),
            image=frame,
        )

        self.counter += 1
        return packet

    def close(self) -> None:
        self.opened = False


def check_file_source() -> None:
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(
            "Run scripts/create_sample_video.py first"
        )

    with VideoFileSource(
        VIDEO_PATH,
        camera_id="sample_cam",
    ) as source:
        ingestor = FrameIngestor(
            source.read,
            empty_policy=EmptyPolicy.FILE,
            expected_camera_id="sample_cam",
        )

        ids = []

        while True:
            result = ingestor.read_once()

            if result.status is IngestStatus.EOF:
                break

            assert result.status is IngestStatus.FRAME
            assert result.packet is not None

            ids.append(result.packet.frame_id)

        assert ids == list(range(24))
        assert ingestor.stats().frames == 24
        assert ingestor.is_eof

    print("FILE INGESTION: PASS")


def check_buffered_source() -> None:
    with BufferedReconnectingSource(
        source_factory=SyntheticLiveSource,
        buffer_size=4,
        initial_backoff=0.01,
        max_backoff=0.05,
    ) as source:
        ingestor = FrameIngestor(
            lambda: source.read(timeout=0.1),
            empty_policy=EmptyPolicy.LIVE,
            expected_camera_id="synthetic_cam",
        )

        received = 0
        deadline = time.monotonic() + 5.0

        while received < 10 and time.monotonic() < deadline:
            result = ingestor.read_once()

            if result.status is IngestStatus.EMPTY:
                continue

            assert result.status is IngestStatus.FRAME
            assert result.packet is not None
            assert result.packet.image.shape == (120, 160, 3)

            received += 1

        assert received == 10
        assert ingestor.stats().frames == 10

    print("BUFFERED LIVE INGESTION: PASS")


def main() -> None:
    check_file_source()
    check_buffered_source()
    print("NODE 2.5 SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
