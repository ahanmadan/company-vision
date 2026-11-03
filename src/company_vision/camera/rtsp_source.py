
from typing import Self
from urllib.parse import urlsplit

import cv2
import numpy as np

from company_vision.core.schemas import FramePacket, current_utc


class RTSPSourceError(RuntimeError):
    """RTSP connection or frame acquisition failure."""


class RTSPSource:
    """Receive RTSP video as standard FramePacket objects."""

    def __init__(
        self,
        url: str,
        camera_id: str = "office_rtsp_01",
        open_timeout_ms: int = 5000,
        read_timeout_ms: int = 5000,
        backend: int = cv2.CAP_FFMPEG,
    ) -> None:

        if not isinstance(url, str) or not url.strip():
            raise ValueError("RTSP URL cannot be empty")

        try:
            parsed = urlsplit(url)

            valid_url = (
                parsed.scheme.lower() == "rtsp"
                and parsed.hostname is not None
                and (
                    parsed.port is None
                    or 1 <= parsed.port <= 65535
                )
            )
        except ValueError:
            raise ValueError("Invalid RTSP URL") from None

        if not valid_url:
            raise ValueError("Invalid RTSP URL")


        if not isinstance(camera_id, str) or not camera_id.strip():
            raise ValueError("camera_id cannot be empty")

        for name, value in (
            ("open_timeout_ms", open_timeout_ms),
            ("read_timeout_ms", read_timeout_ms),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")

        if backend != cv2.CAP_FFMPEG:
            raise ValueError("This source requires FFmpeg backend")

        self._url = url
        self.camera_id = camera_id
        self.open_timeout_ms = open_timeout_ms
        self.read_timeout_ms = read_timeout_ms
        self.backend = backend

        self._capture: cv2.VideoCapture | None = None
        self._frame_id = 0

        self.width = 0
        self.height = 0
        self.fps = 0.0

    @property
    def is_open(self) -> bool:
        return (
            self._capture is not None
            and self._capture.isOpened()
        )

    @property
    def frames_read(self) -> int:
        return self._frame_id

    def open(self) -> None:
        if self.is_open:
            raise RTSPSourceError("RTSP source is already open")

        self.close()

        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            self.open_timeout_ms,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            self.read_timeout_ms,
        ]

        try:
            capture = cv2.VideoCapture(
                self._url,
                self.backend,
                params,
            )
        except (cv2.error, RuntimeError):
            raise RTSPSourceError(
                "RTSP connection could not be initialized"
            ) from None

        if not capture.isOpened():
            capture.release()
            raise RTSPSourceError(
                "RTSP connection could not be opened"
            )

        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
        except (cv2.error, RuntimeError, TypeError, ValueError, OverflowError):
            capture.release()
            raise RTSPSourceError(
                "RTSP metadata could not be read"
            ) from None

        self.width = width
        self.height = height
        self.fps = fps
        self._frame_id = 0
        self._capture = capture

    def read(self) -> FramePacket:
        if not self.is_open:
            raise RTSPSourceError("RTSP source is not open")

        assert self._capture is not None

        try:
            ok, frame = self._capture.read()
        except cv2.error:
            raise RTSPSourceError(
                "RTSP frame acquisition failed"
            ) from None

        if not ok or frame is None:
            raise RTSPSourceError(
                "RTSP stream did not deliver a frame"
            )

        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.shape[0] == 0
            or frame.shape[1] == 0
        ):
            raise RTSPSourceError(
                "RTSP stream delivered an invalid BGR frame"
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

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
