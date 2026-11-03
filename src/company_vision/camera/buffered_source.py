
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from company_vision.core.schemas import FramePacket


class FrameSource(Protocol):
    def open(self) -> None: ...

    def read(self) -> FramePacket | None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class BufferStats:
    state: str
    connection_attempts: int
    successful_connections: int
    connection_failures: int
    frames_received: int
    frames_dropped: int
    frames_delivered: int
    buffer_size: int
    last_error_type: str | None


class BufferedReconnectingSource:
    """
    Read frames in a background thread.

    A bounded queue drops the oldest buffered frame
    when capacity is reached.

    A failed source is closed and recreated using
    exponential reconnect backoff.

    Frame IDs are continuous across reconnections.
    """

    def __init__(
        self,
        source_factory: Callable[[], FrameSource],
        buffer_size: int = 4,
        initial_backoff: float = 0.5,
        max_backoff: float = 8.0,
        shutdown_timeout: float = 7.0,
    ) -> None:
        if not callable(source_factory):
            raise TypeError("source_factory must be callable")

        if (
            isinstance(buffer_size, bool)
            or not isinstance(buffer_size, int)
            or buffer_size <= 0
        ):
            raise ValueError("buffer_size must be positive")

        for name, value in (
            ("initial_backoff", initial_backoff),
            ("max_backoff", max_backoff),
            ("shutdown_timeout", shutdown_timeout),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 < value < float("inf")
            ):
                raise ValueError(f"{name} must be positive and finite")

        if max_backoff < initial_backoff:
            raise ValueError(
                "max_backoff must be >= initial_backoff"
            )

        self._source_factory = source_factory
        self._queue: queue.Queue[FramePacket] = queue.Queue(
            maxsize=buffer_size
        )
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

        self._initial_backoff = float(initial_backoff)
        self._max_backoff = float(max_backoff)
        self._shutdown_timeout = float(shutdown_timeout)

        self._started = False
        self._closed = False
        self._state = "created"

        self._connection_attempts = 0
        self._successful_connections = 0
        self._connection_failures = 0
        self._frames_received = 0
        self._frames_dropped = 0
        self._frames_delivered = 0
        self._last_error_type: str | None = None

    def start(self) -> None:
        with self._lock:
            if self._started or self._closed:
                raise RuntimeError(
                    "Buffered source cannot be started again"
                )

            self._started = True
            self._state = "connecting"

            self._thread = threading.Thread(
                target=self._worker,
                name="company-vision-frame-producer",
                daemon=True,
            )

            self._thread.start()

    def _enqueue(self, packet: FramePacket) -> None:
        with self._lock:
            if self._stop_event.is_set():
                return

            normalized = FramePacket(
                camera_id=packet.camera_id,
                frame_id=self._frames_received,
                timestamp=packet.timestamp,
                image=packet.image,
            )

            self._frames_received += 1

            try:
                self._queue.put_nowait(normalized)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self._frames_dropped += 1
                except queue.Empty:
                    pass

                try:
                    self._queue.put_nowait(normalized)
                except queue.Full:
                    self._frames_dropped += 1

    def _worker(self) -> None:
        delay = self._initial_backoff

        try:
            while not self._stop_event.is_set():
                source: FrameSource | None = None

                with self._lock:
                    self._connection_attempts += 1
                    self._state = (
                        "connecting"
                        if self._successful_connections == 0
                        else "reconnecting"
                    )

                try:
                    source = self._source_factory()
                    source.open()

                    with self._lock:
                        self._successful_connections += 1
                        self._state = "streaming"
                        self._last_error_type = None

                    delay = self._initial_backoff

                    while not self._stop_event.is_set():
                        packet = source.read()

                        if packet is None:
                            raise RuntimeError(
                                "Live source returned no frame"
                            )

                        self._enqueue(packet)

                except (OSError, RuntimeError, ValueError) as exc:
                    if not self._stop_event.is_set():
                        with self._lock:
                            self._connection_failures += 1
                            self._last_error_type = type(exc).__name__
                            self._state = "reconnecting"

                finally:
                    if source is not None:
                        try:
                            source.close()
                        except (OSError, RuntimeError, ValueError):
                            pass

                if self._stop_event.is_set():
                    break

                if self._stop_event.wait(delay):
                    break

                delay = min(delay * 2, self._max_backoff)

        finally:
            with self._lock:
                self._state = "stopped"

    def read(self, timeout: float = 1.0) -> FramePacket | None:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not 0 <= timeout < float("inf")
        ):
            raise ValueError("timeout must be finite and non-negative")

        with self._lock:
            if not self._started or self._closed:
                raise RuntimeError(
                    "Buffered source is not active"
                )

        try:
            packet = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

        with self._lock:
            self._frames_delivered += 1

        return packet

    def stats(self) -> BufferStats:
        with self._lock:
            return BufferStats(
                state=self._state,
                connection_attempts=self._connection_attempts,
                successful_connections=self._successful_connections,
                connection_failures=self._connection_failures,
                frames_received=self._frames_received,
                frames_dropped=self._frames_dropped,
                frames_delivered=self._frames_delivered,
                buffer_size=self._queue.qsize(),
                last_error_type=self._last_error_type,
            )

    def close(self) -> None:
        with self._lock:
            self._closed = True
            thread = self._thread

        self._stop_event.set()

        if thread is not None:
            if thread is threading.current_thread():
                raise RuntimeError(
                    "Cannot join producer from itself"
                )

            thread.join(timeout=self._shutdown_timeout)

            if thread.is_alive():
                raise RuntimeError(
                    "Producer did not stop before timeout"
                )

        with self._lock:
            self._state = "stopped"

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
