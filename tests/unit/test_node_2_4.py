
import time

import numpy as np
import pytest

from company_vision.camera.buffered_source import (
    BufferedReconnectingSource,
)
from company_vision.core.schemas import FramePacket, current_utc


class FakeSource:
    def __init__(
        self,
        *,
        fail_open=False,
        fail_after=None,
        interval=0.002,
    ):
        self.fail_open = fail_open
        self.fail_after = fail_after
        self.interval = interval
        self.opened = False
        self.closed = False
        self.counter = 0

    def open(self):
        if self.fail_open:
            raise RuntimeError("Simulated open failure")
        self.opened = True

    def read(self):
        if not self.opened:
            raise RuntimeError("Not open")

        if (
            self.fail_after is not None
            and self.counter >= self.fail_after
        ):
            raise RuntimeError("Simulated read failure")

        time.sleep(self.interval)

        packet = FramePacket(
            camera_id="test_camera",
            frame_id=self.counter,
            timestamp=current_utc(),
            image=np.zeros((32, 48, 3), dtype=np.uint8),
        )
        self.counter += 1
        return packet

    def close(self):
        self.opened = False
        self.closed = True


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)

    return bool(predicate())


def make_buffered(factory=None, **kwargs):
    if factory is None:
        factory = FakeSource

    options = {
        "initial_backoff": 0.005,
        "max_backoff": 0.02,
    }

    options.update(kwargs)

    return BufferedReconnectingSource(
        source_factory=factory,
        **options,
    )

def test_01_invalid_buffer_size():
    with pytest.raises(ValueError):
        make_buffered(buffer_size=0)


def test_02_invalid_backoff():
    with pytest.raises(ValueError):
        make_buffered(initial_backoff=-1)


def test_03_invalid_factory():
    with pytest.raises(TypeError):
        BufferedReconnectingSource(source_factory=None)


def test_04_read_before_start():
    source = make_buffered()

    with pytest.raises(RuntimeError):
        source.read()


def test_05_receive_frame_packet():
    with make_buffered() as source:
        packet = source.read(timeout=1.0)

        assert isinstance(packet, FramePacket)
        assert packet.camera_id == "test_camera"
        assert packet.image.shape == (32, 48, 3)


def test_06_double_start_rejected():
    with make_buffered() as source, pytest.raises(RuntimeError):
        source.start()


def test_07_close_prevents_read():
    source = make_buffered()
    source.start()

    assert wait_until(
        lambda: source.stats().frames_received >= 1
    )

    source.close()

    with pytest.raises(RuntimeError):
        source.read()


def test_08_close_releases_source():
    created = []

    def factory():
        item = FakeSource()
        created.append(item)
        return item

    with make_buffered(factory) as source:
        assert wait_until(
            lambda: source.stats().frames_received >= 1
        )

    assert created
    assert all(item.closed for item in created)
    assert source.stats().state == "stopped"


def test_09_reconnect_after_open_failure():
    attempts = 0

    def factory():
        nonlocal attempts
        attempts += 1
        return FakeSource(fail_open=(attempts == 1))

    with make_buffered(factory) as source:
        assert wait_until(
            lambda: source.stats().successful_connections >= 1
        )

        stats = source.stats()
        assert stats.connection_attempts >= 2
        assert stats.connection_failures >= 1


def test_10_reconnect_after_read_failure():
    with make_buffered(
        lambda: FakeSource(fail_after=2)
    ) as source:
        assert wait_until(
            lambda: source.stats().successful_connections >= 2
        )

        assert source.stats().connection_failures >= 1


def test_11_frame_ids_continue_across_reconnects():
    with make_buffered(
        lambda: FakeSource(fail_after=3),
        buffer_size=100,
    ) as source:
        assert wait_until(
            lambda: source.stats().successful_connections >= 2
        )

        packets = []

        deadline = time.monotonic() + 2.0
        while len(packets) < 5 and time.monotonic() < deadline:
            packet = source.read(timeout=0.1)
            if packet is not None:
                packets.append(packet)

        assert len(packets) >= 5

        ids = [packet.frame_id for packet in packets]
        assert ids == sorted(ids)
        assert len(ids) == len(set(ids))


def test_12_bounded_buffer_drops_oldest():
    with make_buffered(buffer_size=2) as source:
        assert wait_until(
            lambda: source.stats().frames_received >= 20
        )

        stats = source.stats()
        assert stats.buffer_size <= 2
        assert stats.frames_dropped > 0

        first = source.read(timeout=0.1)
        assert first is not None
        assert first.frame_id > 0


def test_13_timeout_returns_none_while_reconnecting():
    with make_buffered(
        lambda: FakeSource(fail_open=True)
    ) as source:
        assert source.read(timeout=0.02) is None


def test_14_stats_are_available():
    with make_buffered() as source:
        assert wait_until(
            lambda: source.stats().frames_received >= 1
        )

        stats = source.stats()

        assert stats.connection_attempts >= 1
        assert stats.successful_connections == 1
        assert stats.frames_received >= 1


def test_15_restart_after_close_rejected():
    source = make_buffered()
    source.start()
    source.close()

    with pytest.raises(RuntimeError):
        source.start()


def test_16_nonfinite_timeout_rejected():
    with make_buffered() as source, pytest.raises(ValueError):
        source.read(timeout=float("inf"))
