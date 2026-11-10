
from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import monotonic, perf_counter
from typing import Any

import numpy as np

IMPORTANT_KEYPOINTS = tuple(range(9))


@dataclass(frozen=True)
class PoseJob:
    sequence: int
    frame: np.ndarray
    captured_at: float


@dataclass(frozen=True)
class PoseResult:
    sequence: int
    frame: np.ndarray
    captured_at: float
    completed_at: float
    coordinates: np.ndarray
    poses: np.ndarray
    scores: np.ndarray
    inference_ms: float


class AsyncPosePipeline:
    """
    Single inference worker with latest-only input/output.

    The caller owns the YOLO model and must not call
    predict concurrently while this worker is running.

    Tracking and OpenCV display remain on the caller thread.
    """

    def __init__(
        self,
        model: Any,
        *,
        imgsz: int = 1280,
        conf: float = 0.30,
        gpu_precision: str = "fp16",
    ) -> None:
        if imgsz < 32:
            raise ValueError("imgsz must be >= 32")
        if not 0.0 < conf <= 1.0:
            raise ValueError("conf must be in (0, 1]")
        if gpu_precision not in ("fp16", "fp32"):
            raise ValueError("Unsupported precision")

        self.model = model
        self.imgsz = imgsz
        self.conf = conf
        self.gpu_precision = gpu_precision

        self._input: Queue[PoseJob] = Queue(maxsize=1)
        self._output: Queue[PoseResult] = Queue(maxsize=1)

        self._stop = Event()
        self._lock = Lock()
        self._thread: Thread | None = None
        self._error: BaseException | None = None

        self._submitted = 0
        self._processed = 0
        self._dropped_input = 0
        self._dropped_output = 0
        self._inference_sum_ms = 0.0

    @staticmethod
    def _replace_latest(queue: Queue, item: Any) -> bool:
        dropped = False

        try:
            queue.put_nowait(item)
            return dropped
        except Full:
            pass

        try:
            queue.get_nowait()
            dropped = True
        except Empty:
            pass

        queue.put_nowait(item)
        return dropped

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Pipeline already started")

        self._stop.clear()
        self._thread = Thread(
            target=self._run,
            name="company-vision-pose-worker",
            daemon=False,
        )
        self._thread.start()

    def submit(
        self,
        sequence: int,
        frame: np.ndarray,
        captured_at: float,
    ) -> None:
        self.check_error()

        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError("Pipeline is not running")

        if self._stop.is_set():
            raise RuntimeError("Pipeline is stopping")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected BGR frame")

        job = PoseJob(
            sequence=sequence,
            frame=frame,
            captured_at=captured_at,
        )

        with self._lock:
            dropped = self._replace_latest(self._input, job)
            self._submitted += 1
            self._dropped_input += int(dropped)

    def poll(self) -> PoseResult | None:
        self.check_error()

        try:
            return self._output.get_nowait()
        except Empty:
            return None

    def check_error(self) -> None:
        with self._lock:
            error = self._error

        if error is not None:
            raise RuntimeError(
                "Async pose inference worker failed"
            ) from error

    def _infer(self, job: PoseJob) -> PoseResult:
        started = perf_counter()

        results = self.model.predict(
            source=job.frame,
            device="cuda:0",
            quantize=(16 if self.gpu_precision == "fp16" else 32),
            imgsz=self.imgsz,
            conf=self.conf,
            classes=[0],
            verbose=False,
        )

        # CUDA -> CPU transfers below synchronize the required
        # inference results before completion is reported.
        result = results[0]
        boxes = result.boxes
        keypoints = result.keypoints

        if (
            boxes is not None
            and keypoints is not None
            and keypoints.data is not None
            and len(boxes) > 0
        ):
            coordinates = (
                boxes.xyxy.detach().cpu().numpy()
            )
            scores = boxes.conf.detach().cpu().numpy()
            coco_poses = (
                keypoints.data.detach().cpu().numpy()
            )

            if (
                coco_poses.ndim != 3
                or coco_poses.shape[1:] != (17, 3)
            ):
                raise RuntimeError(
                    "Expected COCO-17 pose structure"
                )

            poses = coco_poses[
                :, IMPORTANT_KEYPOINTS, :
            ].astype(np.float32, copy=True)

            if not (
                len(coordinates)
                == len(scores)
                == len(poses)
            ):
                raise RuntimeError(
                    "Detection/pose count mismatch"
                )
        else:
            coordinates = np.empty(
                (0, 4), dtype=np.float32
            )
            scores = np.empty(
                (0,), dtype=np.float32
            )
            poses = np.empty(
                (0, 9, 3), dtype=np.float32
            )

        completed_at = monotonic()

        return PoseResult(
            sequence=job.sequence,
            frame=job.frame,
            captured_at=job.captured_at,
            completed_at=completed_at,
            coordinates=coordinates,
            poses=poses,
            scores=scores,
            inference_ms=(perf_counter() - started) * 1000,
        )

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    job = self._input.get(timeout=0.05)
                except Empty:
                    continue

                result = self._infer(job)

                with self._lock:
                    self._processed += 1
                    self._inference_sum_ms += result.inference_ms
                    dropped = self._replace_latest(
                        self._output, result
                    )
                    self._dropped_output += int(dropped)

        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._error = exc
            self._stop.set()

    def stats(self) -> dict[str, float | int]:
        with self._lock:
            processed = self._processed
            return {
                "submitted": self._submitted,
                "processed": processed,
                "dropped_input": self._dropped_input,
                "dropped_output": self._dropped_output,
                "average_inference_ms": (
                    self._inference_sum_ms / processed
                    if processed else 0.0
                ),
            }

    def stop(self, timeout: float | None = None) -> None:
        self._stop.set()

        if self._thread is not None:
            self._thread.join(timeout=timeout)

            if self._thread.is_alive():
                raise RuntimeError(
                    "Inference worker did not stop"
                )

        self.check_error()

