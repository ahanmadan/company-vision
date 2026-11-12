"""Node 5.9.4: connect behavior evidence, rules and event lifecycles.

This module does not execute cameras or models and does not infer
employee productivity, intent or misconduct.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from threading import RLock

from company_vision.behaviors.behavior_execution_pipeline import (
    BehaviorExecutionBatch,
    BehaviorExecutionPipeline,
)
from company_vision.behaviors.event_manager import (
    BehaviorEvent,
    EventManager,
    EventPolicy,
    EventTransition,
)
from company_vision.behaviors.rule_engine import (
    RuleDefinition,
    RuleEngine,
    RuleEvaluation,
)
from company_vision.behaviors.schemas import BehaviorFrame

DEFAULT_RULES = (
    RuleDefinition(
        rule_id="phone_usage_detected",
        plugin_id="phone_usage",
        metric_key="score",
        operator="gte",
        threshold=0.75,
        min_confidence=0.5,
    ),
    RuleDefinition(
        rule_id="computer_usage_detected",
        plugin_id="computer_usage",
        metric_key="score",
        operator="gte",
        threshold=0.75,
        min_confidence=0.5,
    ),
    RuleDefinition(
        rule_id="inactivity_detected",
        plugin_id="inactivity",
        metric_key="score",
        operator="gte",
        threshold=0.8,
        min_confidence=0.5,
    ),
    RuleDefinition(
        rule_id="gathering_detected",
        plugin_id="gathering",
        metric_key="score",
        operator="gte",
        threshold=0.8,
        min_confidence=0.5,
    ),
)

# Display ordering only, not an inference about misconduct.
PRIORITY_ORDER = (
    "phone_usage",
    "computer_usage",
    "inactivity",
    "gathering",
)


class RuleEventPipelineError(ValueError):
    """Invalid integrated rule/event input."""


@dataclass(frozen=True, slots=True)
class RuleEventBatch:
    """Output for one frame, with separate evidence and event records."""

    execution: BehaviorExecutionBatch
    evaluations: tuple[RuleEvaluation, ...]
    transitions: tuple[EventTransition, ...]
    active_events: tuple[BehaviorEvent, ...]

    @property
    def camera_id(self) -> str:
        return self.execution.camera_id

    @property
    def session_id(self) -> str:
        return self.execution.session_id

    @property
    def frame_id(self) -> int:
        return self.execution.frame_id

    def evaluations_for_track(
        self, track_id: int
    ) -> tuple[RuleEvaluation, ...]:
        """Sorted evaluations; never collapse unknown into negative."""

        order = {
            plugin_id: i
            for i, plugin_id in enumerate(PRIORITY_ORDER)
        }
        return tuple(sorted(
            (
                e for e in self.evaluations
                if e.track_id == track_id
            ),
            key=lambda e: (
                order.get(e.plugin_id, len(order)),
                e.rule_id,
            ),
        ))


class RuleEventPipeline:
    """Evaluate evidence and manage events without duplicate creation."""

    _TERMINAL = frozenset({
        "ended", "lost", "removed", "expired",
    })

    def __init__(
        self,
        *,
        execution: BehaviorExecutionPipeline | None = None,
        rules: Sequence[RuleDefinition] | None = None,
        policy: EventPolicy | None = None,
    ) -> None:
        self.execution = (
            execution if execution is not None
            else BehaviorExecutionPipeline()
        )

        selected = DEFAULT_RULES if rules is None else tuple(rules)

        if not selected:
            raise ValueError("At least one rule is required")

        self.rule_engine = RuleEngine(selected)
        self.events = EventManager(policy=policy)
        self._sessions: dict[str, str] = {}
        self._lock = RLock()

        # This node requires complete evidence from Node 5.9.3.
        supported = set(PRIORITY_ORDER)

        if any(rule.plugin_id not in supported for rule in selected):
            raise ValueError("Rules reference unsupported behavior plugins")

    @classmethod
    def _terminal_ids(cls, frame: BehaviorFrame) -> tuple[int, ...]:
        ids = [
            signal.track_id
            for signal in frame.track_lifecycle_events
            if signal.event_type.strip().lower() in cls._TERMINAL
        ]

        if any(track_id is None for track_id in ids):
            raise RuleEventPipelineError(
                "Terminal signal requires a track ID"
            )

        if len(ids) != len(set(ids)):
            raise RuleEventPipelineError(
                "Duplicate terminal track signals"
            )

        return tuple(sorted(ids))

    def _evaluate_batch(
        self,
        execution: BehaviorExecutionBatch,
        *,
        terminal_ids: Sequence[int] = (),
    ) -> RuleEventBatch:
        context = execution.context
        frame = context.frame

        evaluations = self.rule_engine.evaluate(
            context,
            execution.results,
        )

        for item in evaluations:
            if (
                item.camera_id != frame.camera_id
                or item.session_id != frame.session_id
                or item.frame_id != frame.frame_id
            ):
                raise RuleEventPipelineError(
                    "Evaluation frame identity mismatch"
                )

        old_session = self._sessions.get(frame.camera_id)

        transitions = []

        # End previous-session events before publishing new events.
        # Existing event snapshots retain their original identities.
        if old_session is not None and old_session != frame.session_id:
            transitions.extend(self.events.end_session(
                frame.camera_id,
                old_session,
                frame_id=frame.frame_id,
                timestamp_utc=frame.timestamp_utc,
                reason="camera_session_changed",
            ))

        # EventManager requires a non-empty evaluation sequence.
        # An empty frame must not fabricate a negative observation.
        if evaluations:
            transitions.extend(self.events.process(
                evaluations,
                timestamp_utc=frame.timestamp_utc,
            ))

        for track_id in terminal_ids:
            transitions.extend(self.events.end_track(
                frame.camera_id,
                frame.session_id,
                track_id,
                frame_id=frame.frame_id,
                timestamp_utc=frame.timestamp_utc,
                reason="track_terminated",
            ))

        self._sessions[frame.camera_id] = frame.session_id

        active = self.events.get_active_events(
            camera_id=frame.camera_id,
            session_id=frame.session_id,
        )

        return RuleEventBatch(
            execution=execution,
            evaluations=evaluations,
            transitions=tuple(transitions),
            active_events=active,
        )

    def process_batch(
        self,
        execution: BehaviorExecutionBatch,
        *,
        terminal_track_ids: Sequence[int] = (),
    ) -> RuleEventBatch:
        """Accept an already processed Node 5.9.3 batch."""

        if not isinstance(execution, BehaviorExecutionBatch):
            raise TypeError("Expected BehaviorExecutionBatch")

        ids = tuple(terminal_track_ids)

        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in ids
        ):
            raise RuleEventPipelineError("Invalid terminal track ID")

        if len(ids) != len(set(ids)):
            raise RuleEventPipelineError(
                "Duplicate terminal track ID"
            )

        observed = {
            person.track_id
            for person in execution.context.frame.persons
            if person.track_id is not None
        }

        if set(ids) & observed:
            raise RuleEventPipelineError(
                "Terminal track is still present in the frame"
            )

        with self._lock:
            return self._evaluate_batch(
                execution,
                terminal_ids=ids,
            )

    def process(self, frame: BehaviorFrame) -> RuleEventBatch:
        """Run plugins, rules and event lifecycle for one frame."""

        if not isinstance(frame, BehaviorFrame):
            raise TypeError("Expected BehaviorFrame")

        # Validate lifecycle inputs before advancing temporal state.
        ids = self._terminal_ids(frame)

        present = {
            item.track_id
            for item in frame.persons
            if item.track_id is not None
        }

        if set(ids) & present:
            raise RuleEventPipelineError(
                "Terminal track is present in the same frame"
            )

        with self._lock:
            execution = self.execution.process(frame)
            return self._evaluate_batch(
                execution,
                terminal_ids=ids,
            )

    def process_tracking(self, **kwargs) -> RuleEventBatch:
        """Accept tracking inputs from Node 5.9.1."""

        with self._lock:
            frame = self.execution.temporal.bridge.build(**kwargs)
            return self.process(frame)

    def clear_session(
        self,
        camera_id: str,
        session_id: str,
        *,
        frame_id: int,
        timestamp_utc,
    ) -> tuple[EventTransition, ...]:
        """Explicitly terminate a camera session."""

        with self._lock:
            transitions = self.events.end_session(
                camera_id,
                session_id,
                frame_id=frame_id,
                timestamp_utc=timestamp_utc,
                reason="session_cleared",
            )

            self.execution.clear_session(camera_id, session_id)

            if self._sessions.get(camera_id) == session_id:
                del self._sessions[camera_id]

            return transitions

    def clear(self) -> None:
        """Reset histories, event state and session markers."""

        with self._lock:
            self.execution.clear()
            self.events.clear()
            self._sessions.clear()

    @property
    def active_event_count(self) -> int:
        return self.events.active_event_count