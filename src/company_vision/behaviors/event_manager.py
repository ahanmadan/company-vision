from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import RLock
from types import MappingProxyType
from typing import Literal

from company_vision.behaviors.rule_engine import RuleEvaluation

EventStatus = Literal["pending", "confirmed"]
EventAction = Literal["started", "updated", "confirmed", "ended"]


class EventManagerError(ValueError):
    """Base event lifecycle error."""


class EventContractError(EventManagerError):
    """Invalid event input or identity."""


class EventConfigurationError(EventManagerError):
    """Invalid event lifecycle configuration."""


@dataclass(frozen=True, slots=True)
class EventPolicy:
    """Explicit frame-based event lifecycle policy."""

    confirm_after_triggered_frames: int = 2
    end_after_not_triggered_frames: int = 2

    def __post_init__(self):
        for name in (
            "confirm_after_triggered_frames",
            "end_after_not_triggered_frames",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise EventConfigurationError(
                    f"{name} must be a positive integer"
                )


@dataclass(frozen=True, slots=True)
class BehaviorEvent:
    """Immutable public snapshot of one behavior event."""

    event_id: str
    rule_id: str
    plugin_id: str

    camera_id: str
    session_id: str
    track_id: int | None

    status: EventStatus

    started_frame_id: int
    last_frame_id: int
    ended_frame_id: int | None

    triggered_count: int
    consecutive_clear_count: int

    observed_value: float | None
    threshold: float
    operator: str
    confidence: float | None

    reason: str
    end_reason: str | None

    started_at_utc: datetime
    updated_at_utc: datetime
    ended_at_utc: datetime | None

    evidence: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.evidence, Mapping):
            raise EventContractError("Evidence must be a mapping")

        object.__setattr__(
            self,
            "evidence",
            MappingProxyType(dict(self.evidence)),
        )


@dataclass(frozen=True, slots=True)
class EventTransition:
    """One lifecycle action and its immutable event snapshot."""

    action: EventAction
    event: BehaviorEvent


@dataclass(slots=True)
class _MutableEvent:
    event_id: str
    rule_id: str
    plugin_id: str

    camera_id: str
    session_id: str
    track_id: int | None

    status: EventStatus

    started_frame_id: int
    last_frame_id: int

    triggered_count: int
    consecutive_clear_count: int

    observed_value: float | None
    threshold: float
    operator: str
    confidence: float | None

    reason: str

    started_at_utc: datetime
    updated_at_utc: datetime

    evidence: Mapping[str, object]


class EventManager:
    """
    Deterministic event lifecycle manager.

    Input:
        Sequence[RuleEvaluation]

    Responsibilities:
        - Create events from triggered evaluations.
        - Update existing events without duplication.
        - Confirm events using explicit frame policies.
        - End events after consecutive negative evaluations.
        - Preserve unknown and not_applicable semantics.
        - Isolate camera, session, rule and track identities.
        - Support explicit track/session termination.

    Non-responsibilities:
        - Feature extraction.
        - Plugin execution.
        - Behavior classification.
        - Rule evaluation.
        - Database persistence.
        - Final BehaviorResult formatting.
    """

    def __init__(
        self,
        policy: EventPolicy | None = None,
    ):
        self.policy = policy or EventPolicy()

        if not isinstance(self.policy, EventPolicy):
            raise TypeError("policy must be EventPolicy")

        self._active: dict[
            tuple[str, str, str, int | None],
            _MutableEvent,
        ] = {}

        self._last_frames: dict[
            tuple[str, str],
            int,
        ] = {}

        self._lock = RLock()

    @staticmethod
    def _key(evaluation: RuleEvaluation):
        return (
            evaluation.camera_id,
            evaluation.session_id,
            evaluation.rule_id,
            evaluation.track_id,
        )

    @staticmethod
    def _event_id(key, frame_id: int) -> str:
        raw = repr((*key, frame_id)).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()[:24]
        return f"evt_{digest}"

    @staticmethod
    def _validate_timestamp(timestamp: datetime):
        if not isinstance(timestamp, datetime):
            raise EventContractError(
                "timestamp_utc must be datetime"
            )

        if (
            timestamp.tzinfo is None
            or timestamp.utcoffset() != timedelta(0)
        ):
            raise EventContractError(
                "timestamp_utc must be timezone-aware UTC"
            )

    @classmethod
    def _validate_batch(
        cls,
        evaluations: Sequence[RuleEvaluation],
        timestamp: datetime,
    ):
        cls._validate_timestamp(timestamp)

        if not isinstance(evaluations, Sequence):
            raise TypeError("evaluations must be a sequence")

        if isinstance(evaluations, (str, bytes)):
            raise TypeError("Invalid evaluations sequence")

        if not evaluations:
            raise EventContractError(
                "At least one RuleEvaluation is required"
            )

        if any(
            not isinstance(item, RuleEvaluation)
            for item in evaluations
        ):
            raise EventContractError(
                "All inputs must be RuleEvaluation instances"
            )

        first = evaluations[0]

        identity = (
            first.camera_id,
            first.session_id,
            first.frame_id,
        )

        seen = set()

        for item in evaluations:
            current = (
                item.camera_id,
                item.session_id,
                item.frame_id,
            )

            if current != identity:
                raise EventContractError(
                    "Mixed camera/session/frame batch"
                )

            key = cls._key(item)

            if key in seen:
                raise EventContractError(
                    "Duplicate event evaluation in one frame"
                )

            seen.add(key)

            if not item.camera_id or not item.session_id:
                raise EventContractError(
                    "Camera and session IDs are required"
                )

            if not item.rule_id or not item.plugin_id:
                raise EventContractError(
                    "Rule and plugin IDs are required"
                )

            if (
                item.track_id is not None
                and (
                    isinstance(item.track_id, bool)
                    or not isinstance(item.track_id, int)
                    or item.track_id < 0
                )
            ):
                raise EventContractError("Invalid track ID")

        return identity

    @staticmethod
    def _snapshot(
        event: _MutableEvent,
        *,
        ended_frame_id=None,
        ended_at_utc=None,
        end_reason=None,
    ) -> BehaviorEvent:

        return BehaviorEvent(
            event_id=event.event_id,
            rule_id=event.rule_id,
            plugin_id=event.plugin_id,
            camera_id=event.camera_id,
            session_id=event.session_id,
            track_id=event.track_id,
            status=event.status,
            started_frame_id=event.started_frame_id,
            last_frame_id=event.last_frame_id,
            ended_frame_id=ended_frame_id,
            triggered_count=event.triggered_count,
            consecutive_clear_count=event.consecutive_clear_count,
            observed_value=event.observed_value,
            threshold=event.threshold,
            operator=event.operator,
            confidence=event.confidence,
            reason=event.reason,
            end_reason=end_reason,
            started_at_utc=event.started_at_utc,
            updated_at_utc=event.updated_at_utc,
            ended_at_utc=ended_at_utc,
            evidence=event.evidence,
        )

    @staticmethod
    def _update_evidence(
        event: _MutableEvent,
        evaluation: RuleEvaluation,
        timestamp: datetime,
    ):
        event.last_frame_id = evaluation.frame_id
        event.updated_at_utc = timestamp
        event.observed_value = evaluation.observed_value
        event.confidence = evaluation.confidence
        event.reason = evaluation.reason
        event.evidence = MappingProxyType(
            dict(evaluation.evidence)
        )

    def process(
        self,
        evaluations: Sequence[RuleEvaluation],
        *,
        timestamp_utc: datetime,
    ) -> tuple[EventTransition, ...]:
        """
        Process one camera/session/frame evaluation batch.

        Unknown and not_applicable never create or close events.
        They reset the consecutive-clear counter of existing events.
        """

        identity = self._validate_batch(
            evaluations,
            timestamp_utc,
        )

        camera_id, session_id, frame_id = identity
        session_key = (camera_id, session_id)

        with self._lock:

            previous = self._last_frames.get(session_key)

            if previous is not None and frame_id <= previous:
                raise EventContractError(
                    "Frame IDs must increase strictly"
                )

            transitions = []

            for evaluation in sorted(
                evaluations,
                key=lambda item: (
                    item.rule_id,
                    item.track_id is None,
                    item.track_id if item.track_id is not None else -1,
                ),
            ):
                key = self._key(evaluation)
                event = self._active.get(key)

                if evaluation.outcome == "triggered":

                    if event is None:
                        event = _MutableEvent(
                            event_id=self._event_id(
                                key,
                                evaluation.frame_id,
                            ),
                            rule_id=evaluation.rule_id,
                            plugin_id=evaluation.plugin_id,
                            camera_id=evaluation.camera_id,
                            session_id=evaluation.session_id,
                            track_id=evaluation.track_id,
                            status="pending",
                            started_frame_id=evaluation.frame_id,
                            last_frame_id=evaluation.frame_id,
                            triggered_count=1,
                            consecutive_clear_count=0,
                            observed_value=evaluation.observed_value,
                            threshold=evaluation.threshold,
                            operator=evaluation.operator,
                            confidence=evaluation.confidence,
                            reason=evaluation.reason,
                            started_at_utc=timestamp_utc,
                            updated_at_utc=timestamp_utc,
                            evidence=MappingProxyType(
                                dict(evaluation.evidence)
                            ),
                        )

                        self._active[key] = event

                        transitions.append(
                            EventTransition(
                                "started",
                                self._snapshot(event),
                            )
                        )

                    else:
                        event.triggered_count += 1
                        event.consecutive_clear_count = 0

                        self._update_evidence(
                            event,
                            evaluation,
                            timestamp_utc,
                        )

                        if (
                            event.status == "pending"
                            and event.triggered_count
                            >= self.policy.confirm_after_triggered_frames
                        ):
                            event.status = "confirmed"

                            transitions.append(
                                EventTransition(
                                    "confirmed",
                                    self._snapshot(event),
                                )
                            )

                        else:
                            transitions.append(
                                EventTransition(
                                    "updated",
                                    self._snapshot(event),
                                )
                            )

                    if (
                        event.status == "pending"
                        and event.triggered_count
                        >= self.policy.confirm_after_triggered_frames
                    ):
                        event.status = "confirmed"

                        transitions.append(
                            EventTransition(
                                "confirmed",
                                self._snapshot(event),
                            )
                        )

                elif evaluation.outcome == "not_triggered":

                    if event is None:
                        continue

                    event.consecutive_clear_count += 1

                    self._update_evidence(
                        event,
                        evaluation,
                        timestamp_utc,
                    )

                    if (
                        event.consecutive_clear_count
                        >= self.policy.end_after_not_triggered_frames
                    ):
                        snapshot = self._snapshot(
                            event,
                            ended_frame_id=frame_id,
                            ended_at_utc=timestamp_utc,
                            end_reason="consecutive_not_triggered",
                        )

                        transitions.append(
                            EventTransition("ended", snapshot)
                        )

                        del self._active[key]

                    else:
                        transitions.append(
                            EventTransition(
                                "updated",
                                self._snapshot(event),
                            )
                        )

                else:
                    # Unknown / not_applicable is not negative evidence.
                    if event is not None:
                        event.consecutive_clear_count = 0

                        self._update_evidence(
                            event,
                            evaluation,
                            timestamp_utc,
                        )

                        transitions.append(
                            EventTransition(
                                "updated",
                                self._snapshot(event),
                            )
                        )

            self._last_frames[session_key] = frame_id

            return tuple(transitions)

    def _end_matching(
        self,
        predicate,
        *,
        frame_id: int,
        timestamp_utc: datetime,
        reason: str,
    ) -> tuple[EventTransition, ...]:

        self._validate_timestamp(timestamp_utc)

        if not isinstance(reason, str) or not reason.strip():
            raise EventContractError("End reason is required")

        if (
            isinstance(frame_id, bool)
            or not isinstance(frame_id, int)
            or frame_id < 0
        ):
            raise EventContractError("Invalid ending frame ID")

        with self._lock:

            keys = sorted(
                (
                    key for key in self._active
                    if predicate(key)
                ),
                key=repr,
            )

            transitions = []

            for key in keys:
                event = self._active.pop(key)

                event.last_frame_id = max(
                    event.last_frame_id,
                    frame_id,
                )
                event.updated_at_utc = timestamp_utc

                transitions.append(
                    EventTransition(
                        "ended",
                        self._snapshot(
                            event,
                            ended_frame_id=frame_id,
                            ended_at_utc=timestamp_utc,
                            end_reason=reason,
                        ),
                    )
                )

            return tuple(transitions)

    def end_track(
        self,
        camera_id: str,
        session_id: str,
        track_id: int,
        *,
        frame_id: int,
        timestamp_utc: datetime,
        reason: str = "track_lost",
    ) -> tuple[EventTransition, ...]:

        return self._end_matching(
            lambda key: (
                key[0] == camera_id
                and key[1] == session_id
                and key[3] == track_id
            ),
            frame_id=frame_id,
            timestamp_utc=timestamp_utc,
            reason=reason,
        )

    def end_session(
        self,
        camera_id: str,
        session_id: str,
        *,
        frame_id: int,
        timestamp_utc: datetime,
        reason: str = "session_ended",
    ) -> tuple[EventTransition, ...]:

        return self._end_matching(
            lambda key: (
                key[0] == camera_id
                and key[1] == session_id
            ),
            frame_id=frame_id,
            timestamp_utc=timestamp_utc,
            reason=reason,
        )

    def get_active_events(
        self,
        *,
        camera_id: str | None = None,
        session_id: str | None = None,
    ) -> tuple[BehaviorEvent, ...]:

        with self._lock:

            events = []

            for key in sorted(self._active, key=repr):
                event = self._active[key]

                if (
                    camera_id is not None
                    and event.camera_id != camera_id
                ):
                    continue

                if (
                    session_id is not None
                    and event.session_id != session_id
                ):
                    continue

                events.append(self._snapshot(event))

            return tuple(events)

    @property
    def active_event_count(self) -> int:

        with self._lock:
            return len(self._active)

    def clear(self) -> None:

        with self._lock:
            self._active.clear()
            self._last_frames.clear()