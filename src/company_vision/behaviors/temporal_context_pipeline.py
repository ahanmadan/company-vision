"""Node 5.9.2: safe frame-to-temporal-plugin-context integration.

This module does not run inference, cameras, plugins or rules.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from threading import RLock

from company_vision.behaviors.feature_extractor import (
    FeatureExtractor,
)
from company_vision.behaviors.plugin_registry import BehaviorPluginContext
from company_vision.behaviors.pose_tracking_bridge import PoseTrackingBridge
from company_vision.behaviors.schemas import BehaviorFrame
from company_vision.behaviors.temporal_state_manager import (
    TemporalStateManager,
)


class TemporalPipelineError(ValueError):
    """Rejected frame, lifecycle signal or timeline transition."""


@dataclass(frozen=True, slots=True)
class FrameWatermark:
    session_id: str
    frame_id: int
    timestamp_utc: datetime
    monotonic_timestamp: float | None


class BehaviorTemporalPipeline:
    """Build plugin contexts with isolated, bounded track histories."""

    _ENDED_EVENTS = frozenset({
        "ended", "lost", "removed", "expired",
    })

    def __init__(
        self,
        max_history_per_track: int = 120,
        *,
        bridge: PoseTrackingBridge | None = None,
    ) -> None:
        self.bridge = bridge or PoseTrackingBridge()
        self.extractor = FeatureExtractor()
        self.manager = TemporalStateManager(
            max_history_per_track=max_history_per_track,
        )
        self._watermarks: dict[str, FrameWatermark] = {}
        self._lock = RLock()

    @staticmethod
    def _eligible(person) -> bool:
        return (
            person.track_id is not None
            and person.behavior_eligible
            and person.observed is True
            and person.predicted is not True
        )

    @classmethod
    def _ended_ids(cls, frame: BehaviorFrame) -> tuple[int, ...]:
        ended = []

        for signal in frame.track_lifecycle_events:
            event_type = signal.event_type.strip().lower()

            if event_type not in cls._ENDED_EVENTS:
                continue

            if signal.track_id is None:
                raise TemporalPipelineError(
                    "Terminal track event requires track_id"
                )

            ended.append(signal.track_id)

        if len(ended) != len(set(ended)):
            raise TemporalPipelineError(
                "Duplicate terminal track signals"
            )

        return tuple(ended)

    def _validate_timeline(self, frame: BehaviorFrame) -> None:
        previous = self._watermarks.get(frame.camera_id)

        if previous is None:
            return

        # New sessions deliberately start a fresh camera timeline.
        if previous.session_id != frame.session_id:
            return

        if frame.frame_id <= previous.frame_id:
            raise TemporalPipelineError(
                "Duplicate or out-of-order frame_id"
            )

        if frame.timestamp_utc < previous.timestamp_utc:
            raise TemporalPipelineError(
                "UTC timestamp regression"
            )

        earlier = previous.monotonic_timestamp
        current = frame.monotonic_timestamp

        if (
            earlier is not None
            and current is not None
            and current < earlier
        ):
            raise TemporalPipelineError(
                "Monotonic timestamp regression"
            )

    def process(self, frame: BehaviorFrame) -> BehaviorPluginContext:
        """Commit one frame and return histories for eligible people."""

        if not isinstance(frame, BehaviorFrame):
            raise TypeError("Expected BehaviorFrame")

        with self._lock:
            self._validate_timeline(frame)

            features = self.extractor.extract(frame)
            terminal = set(self._ended_ids(frame))

            eligible = tuple(
                person
                for person in features.persons
                if self._eligible(person)
            )

            eligible_ids = {
                person.track_id for person in eligible
            }

            if terminal & eligible_ids:
                raise TemporalPipelineError(
                    "Track cannot be ended and eligible in same frame"
                )

            # Predicted and unconfirmed tracks must not create
            # continuous valid behavior history.
            invalid_ids = {
                person.track_id
                for person in features.persons
                if (
                    person.track_id is not None
                    and not self._eligible(person)
                )
            }

            ended = terminal | invalid_ids

            # The existing TemporalStateManager performs
            # a staged/atomic update and manages camera sessions.
            valid_features = replace(
                features,
                persons=eligible,
            )

            self.manager.update(
                valid_features,
                ended_track_ids=tuple(sorted(ended)),
            )

            # Make history available only for current,
            # eligible identities. Missing persons are not
            # implicitly marked lost.
            histories = tuple(
                history
                for history in self.manager.get_active_histories(
                    frame.camera_id,
                    frame.session_id,
                )
                if history.track_id in eligible_ids
            )

            context = BehaviorPluginContext(
                frame=features,
                histories=histories,
            )

            self._watermarks[frame.camera_id] = FrameWatermark(
                session_id=frame.session_id,
                frame_id=frame.frame_id,
                timestamp_utc=frame.timestamp_utc,
                monotonic_timestamp=frame.monotonic_timestamp,
            )

            return context

    def process_tracking(self, **kwargs) -> BehaviorPluginContext:
        """Accept exactly the Node 5.9.1 bridge input contract."""

        frame = self.bridge.build(**kwargs)
        return self.process(frame)

    def get_history(
        self,
        camera_id: str,
        session_id: str,
        track_id: int,
    ):
        with self._lock:
            return self.manager.get_history(
                camera_id,
                session_id,
                track_id,
            )

    def clear_session(
        self,
        camera_id: str,
        session_id: str,
    ) -> int:
        with self._lock:
            removed = self.manager.clear_session(
                camera_id,
                session_id,
            )

            watermark = self._watermarks.get(camera_id)

            if (
                watermark is not None
                and watermark.session_id == session_id
            ):
                del self._watermarks[camera_id]

            return removed

    def clear(self) -> None:
        with self._lock:
            self.manager.clear()
            self._watermarks.clear()

    @property
    def active_track_count(self) -> int:
        return self.manager.active_track_count