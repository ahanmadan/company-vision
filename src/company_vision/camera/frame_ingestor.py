
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from typing import Self

import numpy as np

from company_vision.core.schemas import FramePacket


class IngestStatus(str, Enum):
    FRAME = "frame"
    EMPTY = "empty"
    EOF = "eof"


class EmptyPolicy(str, Enum):
    LIVE = "live"
    FILE = "file"


class FrameIngestError(RuntimeError):
    """A frame violates the input contract."""


@dataclass(frozen=True)
class IngestResult:
    status: IngestStatus
    packet: FramePacket | None = None


@dataclass(frozen=True)
class IngestStats:
    polls: int
    frames: int
    empty_polls: int
    end_of_stream: bool
    last_frame_id: int | None
    camera_id: str | None


class FrameIngestor:
    """
    Validate and normalize frames from any compatible source.

    The reader must return FramePacket or None.
    - FILE policy: None means EOF.
    - LIVE policy: None means temporarily unavailable.

    The adapter preserves source frame IDs and timestamp values.
    """

    def __init__(
        self,
        reader: Callable[[], FramePacket | None],
        *,
        empty_policy: EmptyPolicy,
        expected_camera_id: str | None = None,
    ) -> None:
        if not callable(reader):
            raise TypeError("reader must be callable")

        if not isinstance(empty_policy, EmptyPolicy):
            raise TypeError("empty_policy must be an EmptyPolicy")

        if expected_camera_id is not None and (
            not isinstance(expected_camera_id, str)
            or not expected_camera_id.strip()
        ):
            raise ValueError("expected_camera_id must be nonempty")

        self._reader = reader
        self._policy = empty_policy
        self._expected_camera_id = expected_camera_id

        self._polls = 0
        self._frames = 0
        self._empty_polls = 0
        self._eof = False
        self._last_frame_id: int | None = None
        self._camera_id: str | None = None

    @property
    def is_eof(self) -> bool:
        return self._eof

    def _normalize(self, packet: FramePacket) -> FramePacket:
        if not isinstance(packet, FramePacket):
            raise FrameIngestError(
                "Reader returned an unsupported frame type"
            )

        if not isinstance(packet.camera_id, str) or not packet.camera_id:
            raise FrameIngestError("Invalid camera ID")

        if (
            self._expected_camera_id is not None
            and packet.camera_id != self._expected_camera_id
        ):
            raise FrameIngestError("Unexpected camera ID")

        if self._camera_id is not None and packet.camera_id != self._camera_id:
            raise FrameIngestError(
                "Camera ID changed within one ingestor"
            )

        if (
            isinstance(packet.frame_id, bool)
            or not isinstance(packet.frame_id, int)
            or packet.frame_id < 0
        ):
            raise FrameIngestError("Invalid frame ID")

        if (
            self._last_frame_id is not None
            and packet.frame_id <= self._last_frame_id
        ):
            raise FrameIngestError(
                "Frame ID must increase strictly"
            )

        if (
            packet.timestamp.tzinfo is None
            or packet.timestamp.utcoffset() != timedelta(0)
        ):
            raise FrameIngestError(
                "Frame timestamp must use UTC"
            )

        image = packet.image

        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or image.shape[0] <= 0
            or image.shape[1] <= 0
        ):
            raise FrameIngestError(
                "Expected nonempty uint8 BGR image"
            )

        if image.flags.c_contiguous:
            return packet

        contiguous_image = np.ascontiguousarray(image)

        return FramePacket(
            camera_id=packet.camera_id,
            frame_id=packet.frame_id,
            timestamp=packet.timestamp,
            image=contiguous_image,
        )

    def read_once(self) -> IngestResult:
        if self._eof:
            return IngestResult(IngestStatus.EOF)

        # Reader errors propagate: this layer does not reconnect.
        raw_packet = self._reader()
        self._polls += 1

        if raw_packet is None:
            if self._policy is EmptyPolicy.FILE:
                self._eof = True
                return IngestResult(IngestStatus.EOF)

            self._empty_polls += 1
            return IngestResult(IngestStatus.EMPTY)

        packet = self._normalize(raw_packet)

        self._camera_id = packet.camera_id
        self._last_frame_id = packet.frame_id
        self._frames += 1

        return IngestResult(
            status=IngestStatus.FRAME,
            packet=packet,
        )

    def stats(self) -> IngestStats:
        return IngestStats(
            polls=self._polls,
            frames=self._frames,
            empty_polls=self._empty_polls,
            end_of_stream=self._eof,
            last_frame_id=self._last_frame_id,
            camera_id=self._camera_id,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        return None
