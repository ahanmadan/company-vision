
import time

import numpy as np

from company_vision.camera.buffered_source import (
    BufferedReconnectingSource,
)
from company_vision.core.schemas import (
    FramePacket,
    current_utc,
)


class SimulatedLiveSource:
    def __init__(
        self,
        *,
        fail_open: bool = False,
        max_frames: int = 10,
    ) -> None:
        self.fail_open = fail_open
        self.max_frames = max_frames
        self.opened = False
        self.frame_id = 0

    def open(self) -> None:
        if self.fail_open:
            raise RuntimeError("Simulated connection failure")

        self.opened = True

    def read(self) -> FramePacket:
        if not self.opened:
            raise RuntimeError("Source is not open")

        if self.frame_id >= self.max_frames:
            raise RuntimeError("Simulated stream disconnection")

        time.sleep(0.01)

        image = np.zeros((120, 160, 3), dtype=np.uint8)
        image[:, :, 1] = self.frame_id * 10

        packet = FramePacket(
            camera_id="simulated_rtsp_01",
            frame_id=self.frame_id,
            timestamp=current_utc(),
            image=image,
        )

        self.frame_id += 1
        return packet

    def close(self) -> None:
        self.opened = False


def main() -> None:
    attempts = 0

    def source_factory() -> SimulatedLiveSource:
        nonlocal attempts
        attempts += 1

        return SimulatedLiveSource(
            fail_open=(attempts == 1),
            max_frames=10,
        )

    packets = []
    deadline = time.monotonic() + 8.0

    with BufferedReconnectingSource(
        source_factory=source_factory,
        buffer_size=4,
        initial_backoff=0.01,
        max_backoff=0.05,
    ) as buffered:
        while time.monotonic() < deadline:
            packet = buffered.read(timeout=0.1)

            if packet is not None:
                packets.append(packet)

            stats = buffered.stats()

            if (
                len(packets) >= 15
                and stats.successful_connections >= 2
                and stats.connection_failures >= 2
            ):
                break

        stats = buffered.stats()

        assert len(packets) >= 15
        assert stats.successful_connections >= 2
        assert stats.connection_failures >= 2
        assert stats.buffer_size <= 4

        ids = [packet.frame_id for packet in packets]
        assert ids == sorted(ids)
        assert len(ids) == len(set(ids))

        print("Frames delivered:", len(packets))
        print("Successful connections:", stats.successful_connections)
        print("Connection failures:", stats.connection_failures)
        print("Frames dropped:", stats.frames_dropped)
        print("Buffer size:", stats.buffer_size)

    print("NODE 2.4 SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
