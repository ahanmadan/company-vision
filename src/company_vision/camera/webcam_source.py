from typing import Self

import cv2
import numpy as np

from company_vision.core.schemas import FramePacket, current_utc


class WebcamSourceError(RuntimeError):
    """Raised when webcam capture fails."""


class WebcamSource:
    """Read live webcam frames as FramePacket objects."""

    def __init__(
        self,
        camera_index: int = 0,
        camera_id: str = "webcam_01",
        width: int | None = None,
        height: int | None = None,
        backend: int = cv2.CAP_ANY,
    ) -> None:
        if not isinstance(camera_index, int) or isinstance(camera_index, bool):
            raise TypeError("camera_index must be an integer")

        if camera_index < 0:
            raise ValueError("camera_index cannot be negative")

        if not isinstance(camera_id, str) or not camera_id.strip():
            raise ValueError("camera_id cannot be empty")

        for name, value in (("width", width), ("height", height)):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")

        self.camera_index = camera_index
        self.camera_id = camera_id
        self.requested_width = width
        self.requested_height = height
        self.backend = backend

        self._capture: cv2.VideoCapture | None = None
        self._frame_id = 0

        self.width = 0
        self.height = 0
        self.fps = 0.0

    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    @property
    def frames_read(self) -> int:
        return self._frame_id

    def open(self) -> None:
        if self.is_open:
            raise WebcamSourceError("Webcam is already open")

        # Clear a previously closed or failed capture.
        self.close()

        capture = cv2.VideoCapture(self.camera_index, self.backend)

        if not capture.isOpened():
            capture.release()
            raise WebcamSourceError(
                f"Cannot open webcam index {self.camera_index}"
            )

        try:
            if self.requested_width is not None:
                capture.set(
                    cv2.CAP_PROP_FRAME_WIDTH,
                    self.requested_width,
                )

            if self.requested_height is not None:
                capture.set(
                    cv2.CAP_PROP_FRAME_HEIGHT,
                    self.requested_height,
                )

            self.width = int(
                capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            )
            self.height = int(
                capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )
            self.fps = float(
                capture.get(cv2.CAP_PROP_FPS)
            )

            self._frame_id = 0
            self._capture = capture

        except Exception:
            capture.release()
            raise

    def read(self) -> FramePacket:
        if not self.is_open:
            raise WebcamSourceError("Webcam is not open")

        assert self._capture is not None

        ok, frame = self._capture.read()

        if not ok or frame is None:
            raise WebcamSourceError(
                "Failed to read a webcam frame"
            )

        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.shape[0] == 0
            or frame.shape[1] == 0
        ):
            raise WebcamSourceError(
                "Webcam returned an invalid BGR frame"
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
