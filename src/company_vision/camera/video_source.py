from pathlib import Path
from typing import Self

import cv2

from company_vision.core.schemas import (
    FramePacket,
    current_utc,
)


class VideoSourceError(RuntimeError):
    """Raised when a video source cannot be used."""


class VideoFileSource:
    """
    Read video-file frames as standard FramePacket objects.

    Frame IDs start at zero and increment for each valid frame.
    Timestamps represent UTC read time, not recording time.
    """

    def __init__(
        self,
        path: str | Path,
        camera_id: str = "test_video_01",
    ) -> None:
        self.path = Path(path)

        if not camera_id.strip():
            raise ValueError("camera_id cannot be empty")

        self.camera_id = camera_id

        self._capture: cv2.VideoCapture | None = None
        self._frame_id = 0
        self._eof = False

        self.fps: float = 0.0
        self.width: int = 0
        self.height: int = 0
        self.frame_count: int = 0

    @property
    def is_open(self) -> bool:
        return (
            self._capture is not None
            and self._capture.isOpened()
        )

    @property
    def is_eof(self) -> bool:
        return self._eof

    @property
    def frames_read(self) -> int:
        return self._frame_id

    def open(self) -> None:
        if self.is_open:
            raise VideoSourceError(
                "Video source is already open"
            )

        if not self.path.is_file():
            raise FileNotFoundError(
                f"Video file not found: {self.path}"
            )

        self._capture = cv2.VideoCapture(
            str(self.path)
        )

        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None

            raise VideoSourceError(
                f"Cannot open video: {self.path}"
            )

        self._frame_id = 0
        self._eof = False

        self.fps = float(
            self._capture.get(cv2.CAP_PROP_FPS)
        )

        self.width = int(
            self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        self.height = int(
            self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        self.frame_count = int(
            self._capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

    def read(self) -> FramePacket | None:
        if not self.is_open:
            raise VideoSourceError(
                "Video source is not open"
            )

        if self._eof:
            return None

        assert self._capture is not None

        ok, frame = self._capture.read()

        if not ok:
            self._eof = True
            return None

        if frame is None or frame.size == 0:
            raise VideoSourceError(
                "Empty video frame received"
            )

        packet = FramePacket(
            camera_id=self.camera_id,
            frame_id=self._frame_id,
            timestamp=current_utc(),
            image=frame,
        )

        self._frame_id += 1

        return packet

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()
