"""Node 5.10.2: integrate existing tracked poses with behavior output.

Camera decoding, GPU inference and stable tracking remain owned by
the established Phase 4 pipeline.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import cv2

from company_vision.behaviors.operational_output import (
    OperationalFrameOutput,
    OperationalOutputPipeline,
)


def overlay_lines(output: OperationalFrameOutput) -> dict[int, tuple[str, ...]]:
    """Return cautious, per-track operational display labels."""

    labels = {}

    for item in output.overlays:
        if not item.show:
            continue

        lines = [f"ID {item.track_id} | {item.label}"]
        lines.extend(item.behavior_labels)

        labels[item.track_id] = tuple(lines)

    return labels


def draw_behavior_labels(
    image,
    tracked_persons: list[dict[str, Any]],
    output: OperationalFrameOutput,
):
    """Draw text only; existing Phase 4 renderer owns poses/boxes."""

    labels = overlay_lines(output)

    for person in tracked_persons:
        track_id = person["track_id"]

        if track_id not in labels:
            continue

        x1, y1 = map(int, person["bbox"][:2])

        x = max(0, min(image.shape[1] - 1, x1))
        y = max(24, y1 - 12)

        for index, line in enumerate(labels[track_id]):
            cv2.putText(
                image,
                line,
                (x, y + index * 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                2,
            )

    return image


class LiveBehaviorIntegrator:
    """Connect stabilized tracks to the existing Node 5.9.5 pipeline."""

    def __init__(
        self,
        *,
        camera_id: str = "dahua-channel-2",
        session_id: str,
    ) -> None:
        self.camera_id = camera_id
        self.session_id = session_id
        self.pipeline = OperationalOutputPipeline()
        self.frames_processed = 0
        self.results_produced = 0
        self.last_output: OperationalFrameOutput | None = None

    def process(
        self,
        *,
        frame_id: int,
        monotonic_timestamp: float,
        frame,
        tracked_persons: list[dict[str, Any]],
        objects: list[dict] | None = None,
        supplemental_wrists: dict | None = None,
    ) -> OperationalFrameOutput:
        height, width = frame.shape[:2]

        output = self.pipeline.process_tracking(
            camera_id=self.camera_id,
            session_id=self.session_id,
            frame_id=frame_id,
            timestamp_utc=datetime.now(UTC),
            monotonic_timestamp=monotonic_timestamp,
            width=width,
            height=height,
            tracked_persons=tracked_persons,
            supplemental_wrists=supplemental_wrists,
            objects=objects,
            object_detection_status=(
                "available" if objects is not None else "not_run"
            ),
            source_status="ok",
        )

        self.frames_processed += 1
        self.results_produced += len(output.results)
        self.last_output = output

        return output

    def summary(self) -> dict[str, int | str]:
        return {
            "camera_id": self.camera_id,
            "session_id": self.session_id,
            "frames_processed": self.frames_processed,
            "results_produced": self.results_produced,
        }

    def clear(self) -> None:
        self.pipeline.clear()
        self.frames_processed = 0
        self.results_produced = 0
        self.last_output = None