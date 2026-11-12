"""Node 5.9.5: canonical and overlay-ready behavior output.

This module does not draw video frames or infer employee intent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from company_vision.behaviors.behavior_result import (
    BehaviorResult,
    create_behavior_result,
    serialize_component,
)
from company_vision.behaviors.rule_event_pipeline import (
    RuleEventBatch,
    RuleEventPipeline,
)
from company_vision.behaviors.schemas import BehaviorFrame

OverlayState = Literal[
    "confirmed",
    "pending",
    "unknown",
    "no_event",
    "ended",
]


LABELS = {
    "phone_usage": "Phone evidence",
    "computer_usage": "Computer interaction evidence",
    "inactivity": "Observable stillness",
    "gathering": "Group proximity",
}

STATUS_LABELS = {
    "confirmed": "Confirmed evidence",
    "pending": "Evidence pending",
    "unknown": "Insufficient evidence",
    "no_event": "No confirmed event",
    "ended": "Event ended",
}

# Operational ordering, not an assessment of performance.
DISPLAY_ORDER = (
    "phone_usage",
    "computer_usage",
    "inactivity",
    "gathering",
)


class OperationalOutputError(ValueError):
    """Invalid output formatting input or identity."""


@dataclass(frozen=True, slots=True)
class PersonOverlay:
    """Renderer-independent instructions for one current track."""

    camera_id: str
    session_id: str
    frame_id: int
    track_id: int

    bbox_xyxy: tuple[int, int, int, int]
    state: OverlayState
    label: str
    behavior_labels: tuple[str, ...]

    observed: bool | None
    predicted: bool | None
    behavior_eligible: bool

    show: bool


@dataclass(frozen=True, slots=True)
class OperationalFrameOutput:
    """One camera/frame output plus lifecycle transition records."""

    camera_id: str
    session_id: str
    frame_id: int
    width: int
    height: int

    results: tuple[BehaviorResult, ...]
    overlays: tuple[PersonOverlay, ...]
    transitions: tuple[dict, ...]

    def to_payload(self) -> dict:
        """Return a JSON-compatible transport payload."""

        return {
            "schema_version": "1.0",
            "camera_id": self.camera_id,
            "session_id": self.session_id,
            "frame_id": self.frame_id,
            "width": self.width,
            "height": self.height,
            "results": [
                item.to_payload()
                for item in self.results
            ],
            "overlays": serialize_component(self.overlays),
            "transitions": serialize_component(self.transitions),
        }


class OperationalOutputFormatter:
    """Convert Node 5.9.4 events into conservative display output."""

    @staticmethod
    def _bbox(person, width: int, height: int):
        cx = person.bbox_center_x_norm * width
        cy = person.bbox_center_y_norm * height
        bw = person.bbox_width_norm * width
        bh = person.bbox_height_norm * height

        x1 = max(0, min(width, round(cx - bw / 2)))
        x2 = max(0, min(width, round(cx + bw / 2)))
        y1 = max(0, min(height, round(cy - bh / 2)))
        y2 = max(0, min(height, round(cy + bh / 2)))

        if x2 <= x1 or y2 <= y1:
            raise OperationalOutputError("Invalid overlay bbox")

        return (x1, y1, x2, y2)

    @staticmethod
    def _state(person, evaluations, active, transitions):
        eligible = (
            person.behavior_eligible
            and person.observed is True
            and person.predicted is not True
        )

        if not eligible:
            return "unknown"

        if any(event.status == "confirmed" for event in active):
            return "confirmed"

        if any(event.status == "pending" for event in active):
            return "active"

        if any(
            action.action == "ended"
            for action in transitions
        ):
            return "ended"

        if not evaluations:
            return "unknown"

        if any(item.outcome == "unknown" for item in evaluations):
            return "unknown"

        if any(item.outcome == "triggered" for item in evaluations):
            return "unknown"

        if all(
            item.outcome in ("not_triggered", "not_applicable")
            for item in evaluations
        ):
            return "no_event"

        return "unknown"

    @staticmethod
    def _behavior_labels(active):
        order = {
            name: index
            for index, name in enumerate(DISPLAY_ORDER)
        }

        selected = sorted(
            (
                event for event in active
                if event.status == "confirmed"
            ),
            key=lambda item: (
                order.get(item.plugin_id, len(order)),
                item.rule_id,
            ),
        )

        return tuple(dict.fromkeys(
            LABELS.get(event.plugin_id, event.plugin_id)
            for event in selected
        ))

    def format(
        self,
        batch: RuleEventBatch,
    ) -> OperationalFrameOutput:
        if not isinstance(batch, RuleEventBatch):
            raise TypeError("Expected RuleEventBatch")

        frame = batch.execution.context.frame

        results = []
        overlays = []

        people = sorted(
            (
                person
                for person in frame.persons
                if person.track_id is not None
            ),
            key=lambda item: item.track_id,
        )

        for person in people:
            track_id = person.track_id

            evaluations = tuple(
                item for item in batch.evaluations
                if item.track_id == track_id
            )

            active = tuple(
                item for item in batch.active_events
                if item.track_id == track_id
            )

            transitions = tuple(
                item for item in batch.transitions
                if item.event.track_id == track_id
                and item.event.session_id == frame.session_id
                and item.event.camera_id == frame.camera_id
            )

            state = self._state(
                person,
                evaluations,
                active,
                transitions,
            )

            labels = self._behavior_labels(active)

            status = {
                "confirmed": "confirmed",
                "active": "active",
                "unknown": "unknown",
                "no_event": "no_event",
                "ended": "ended",
            }[state]

            bbox = self._bbox(
                person,
                frame.width,
                frame.height,
            )

            evidence = {
                plugin_id: serialize_component(observation)
                for plugin_id, observation
                in batch.execution.per_track[track_id].items()
            }

            metadata = {
                "bbox_xyxy": bbox,
                "observed": person.observed,
                "predicted": person.predicted,
                "behavior_eligible": person.behavior_eligible,
                "display_labels": labels,
                "source_status": frame.source_status,
                "object_detection_status": (
                    frame.object_detection_status
                ),
            }

            result = create_behavior_result(
                camera_id=frame.camera_id,
                session_id=frame.session_id,
                frame_id=frame.frame_id,
                track_id=track_id,
                status=status,
                events=active + tuple(
                    item.event for item in transitions
                ),
                rule_evaluations=evaluations,
                evidence=evidence,
                metadata=metadata,
                created_at=frame.timestamp_utc,
            )

            results.append(result)

            # Predicted tracks remain hidden. The overlay
            # does not display guessed worker positions.
            show = (
                person.observed is True
                and person.predicted is not True
            )

            label = STATUS_LABELS[state if state != "active" else "pending"]

            overlays.append(PersonOverlay(
                camera_id=frame.camera_id,
                session_id=frame.session_id,
                frame_id=frame.frame_id,
                track_id=track_id,
                bbox_xyxy=bbox,
                state=(
                    "pending" if state == "active" else state
                ),
                label=label,
                behavior_labels=labels,
                observed=person.observed,
                predicted=person.predicted,
                behavior_eligible=person.behavior_eligible,
                show=show,
            ))

        return OperationalFrameOutput(
            camera_id=frame.camera_id,
            session_id=frame.session_id,
            frame_id=frame.frame_id,
            width=frame.width,
            height=frame.height,
            results=tuple(results),
            overlays=tuple(overlays),
            transitions=tuple(
                serialize_component(item)
                for item in batch.transitions
            ),
        )


class OperationalOutputPipeline:
    """Full logical path from tracking observations to final output."""

    def __init__(
        self,
        *,
        rules: RuleEventPipeline | None = None,
        formatter: OperationalOutputFormatter | None = None,
    ) -> None:
        self.rules = rules or RuleEventPipeline()
        self.formatter = formatter or OperationalOutputFormatter()

    def process(self, frame: BehaviorFrame):
        return self.formatter.format(self.rules.process(frame))

    def process_tracking(self, **kwargs):
        return self.formatter.format(
            self.rules.process_tracking(**kwargs)
        )

    def clear(self):
        self.rules.clear()

    def clear_session(
        self,
        camera_id,
        session_id,
        *,
        frame_id,
        timestamp_utc,
    ):
        return self.rules.clear_session(
            camera_id,
            session_id,
            frame_id=frame_id,
            timestamp_utc=timestamp_utc,
        )