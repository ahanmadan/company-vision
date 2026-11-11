from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from threading import RLock

from company_vision.behaviors.feature_extractor import (
    FrameFeatures,
    PersonFeatures,
)


class TemporalStateError(ValueError):
    """Raised when temporal history cannot be updated safely."""


@dataclass(frozen=True, slots=True)
class TemporalObservation:
    """One immutable observation of a tracked person."""

    frame_id: int
    timestamp_utc: datetime
    monotonic_timestamp: float | None
    person: PersonFeatures


@dataclass(frozen=True, slots=True)
class TemporalTrackHistory:
    """Bounded chronological history for one camera/session/track."""

    camera_id: str
    session_id: str
    track_id: int
    observations: tuple[TemporalObservation, ...]

    @property
    def observation_count(self) -> int:
        return len(self.observations)

    @property
    def latest(self) -> TemporalObservation:
        return self.observations[-1]

    @property
    def earliest(self) -> TemporalObservation:
        return self.observations[0]


class TemporalStateManager:
    """
    Maintain independent temporal histories for tracked persons.

    History identity:
        camera_id + session_id + track_id

    Responsibilities:
        - Maintain chronological per-track observations.
        - Bound observations per track.
        - Isolate camera and session state.
        - Reset old camera history on session changes.
        - Remove explicitly ended tracks.
        - Reject duplicate or out-of-order observations.
        - Preserve observed/predicted/eligibility metadata.

    Non-responsibilities:
        - Tracking or generating track IDs.
        - Behavior classification.
        - Rule evaluation.
        - Object-person association.
        - Event generation.
        - Camera or model execution.

    Predicted observations may be retained for continuity,
    but their behavior_eligible flag remains unchanged.
    """

    def __init__(self, max_history_per_track: int = 120) -> None:

        if (
            isinstance(max_history_per_track, bool)
            or not isinstance(max_history_per_track, int)
            or max_history_per_track < 1
        ):
            raise ValueError(
                "max_history_per_track must be a positive integer"
            )

        self.max_history_per_track = max_history_per_track

        self._histories: dict[
            tuple[str, str, int],
            TemporalTrackHistory,
        ] = {}

        self._camera_sessions: dict[str, str] = {}

        self._lock = RLock()

    @staticmethod
    def _validate_frame(frame: FrameFeatures) -> None:

        if not isinstance(frame, FrameFeatures):
            raise TypeError("frame must be a FrameFeatures instance")

        if not frame.camera_id or not frame.session_id:
            raise TemporalStateError(
                "Camera and session IDs must not be empty"
            )

        if (
            not isinstance(frame.frame_id, int)
            or isinstance(frame.frame_id, bool)
            or frame.frame_id < 0
        ):
            raise TemporalStateError("Invalid frame_id")

        timestamp = frame.timestamp_utc

        if (
            timestamp.tzinfo is None
            or timestamp.utcoffset() is None
            or timestamp.utcoffset().total_seconds() != 0
        ):
            raise TemporalStateError(
                "timestamp_utc must be timezone-aware UTC"
            )

        if frame.monotonic_timestamp is not None and (
            not isfinite(frame.monotonic_timestamp)
            or frame.monotonic_timestamp < 0
        ):
            raise TemporalStateError(
                "Invalid monotonic timestamp"
            )

        seen = set()

        for person in frame.persons:

            if (
                person.camera_id != frame.camera_id
                or person.session_id != frame.session_id
                or person.frame_id != frame.frame_id
            ):
                raise TemporalStateError(
                    "Person identity does not match its frame"
                )

            track_id = person.track_id

            if track_id is None:
                continue

            if (
                isinstance(track_id, bool)
                or not isinstance(track_id, int)
                or track_id < 0
            ):
                raise TemporalStateError("Invalid track_id")

            if track_id in seen:
                raise TemporalStateError(
                    f"Duplicate track ID: {track_id}"
                )

            seen.add(track_id)

    @staticmethod
    def _validate_ended_ids(
        ended_track_ids: Iterable[int],
    ) -> tuple[int, ...]:

        if ended_track_ids is None:
            raise TemporalStateError(
                "ended_track_ids must be an iterable"
            )

        try:
            values = tuple(ended_track_ids)
        except TypeError as exc:
            raise TemporalStateError(
                "ended_track_ids must be iterable"
            ) from exc

        for track_id in values:

            if (
                isinstance(track_id, bool)
                or not isinstance(track_id, int)
                or track_id < 0
            ):
                raise TemporalStateError(
                    "Invalid ended track ID"
                )

        if len(values) != len(set(values)):
            raise TemporalStateError(
                "Duplicate ended track IDs"
            )

        return values

    @staticmethod
    def _validate_chronology(
        previous: TemporalObservation,
        current: TemporalObservation,
    ) -> None:

        if current.frame_id <= previous.frame_id:
            raise TemporalStateError(
                "Frame IDs must increase for each track"
            )

        if current.timestamp_utc < previous.timestamp_utc:
            raise TemporalStateError(
                "UTC timestamp regression detected"
            )

        previous_mono = previous.monotonic_timestamp
        current_mono = current.monotonic_timestamp

        if (
            previous_mono is not None
            and current_mono is not None
            and current_mono < previous_mono
        ):
            raise TemporalStateError(
                "Monotonic timestamp regression detected"
            )

    def update(
        self,
        frame: FrameFeatures,
        ended_track_ids: Iterable[int] = (),
    ) -> tuple[TemporalTrackHistory, ...]:
        """
        Append current frame observations.

        ended_track_ids:
            Track IDs explicitly declared lost/ended by the
            upstream tracking lifecycle.

        An absent person in an ordinary frame does not imply
        track termination.
        """

        self._validate_frame(frame)

        ended = self._validate_ended_ids(ended_track_ids)

        persons = {
            person.track_id: person
            for person in frame.persons
            if person.track_id is not None
        }

        overlap = set(ended).intersection(persons)

        if overlap:
            raise TemporalStateError(
                "A track cannot be ended and observed in the same frame"
            )

        camera_id = frame.camera_id
        session_id = frame.session_id

        with self._lock:

            staged = dict(self._histories)

            previous_session = self._camera_sessions.get(camera_id)

            session_changed = (
                previous_session is not None
                and previous_session != session_id
            )

            if session_changed:

                staged = {
                    key: history
                    for key, history in staged.items()
                    if key[0] != camera_id
                }

            for track_id in ended:

                staged.pop(
                    (camera_id, session_id, track_id),
                    None,
                )

            updated = []

            for track_id, person in persons.items():

                key = (camera_id, session_id, track_id)

                observation = TemporalObservation(
                    frame_id=frame.frame_id,
                    timestamp_utc=frame.timestamp_utc,
                    monotonic_timestamp=frame.monotonic_timestamp,
                    person=person,
                )

                previous = staged.get(key)

                if previous is not None:

                    self._validate_chronology(
                        previous.latest,
                        observation,
                    )

                    observations = (
                        previous.observations + (observation,)
                    )

                else:
                    observations = (observation,)

                observations = observations[
                    -self.max_history_per_track:
                ]

                history = TemporalTrackHistory(
                    camera_id=camera_id,
                    session_id=session_id,
                    track_id=track_id,
                    observations=observations,
                )

                staged[key] = history
                updated.append(history)

            # Commit only after all observations are validated.
            self._histories = staged
            self._camera_sessions[camera_id] = session_id

            return tuple(updated)

    def get_history(
        self,
        camera_id: str,
        session_id: str,
        track_id: int,
    ) -> TemporalTrackHistory | None:

        key = (camera_id, session_id, track_id)

        with self._lock:
            return self._histories.get(key)

    def get_active_histories(
        self,
        camera_id: str,
        session_id: str,
    ) -> tuple[TemporalTrackHistory, ...]:

        with self._lock:

            result = [
                history
                for key, history in self._histories.items()
                if key[0] == camera_id and key[1] == session_id
            ]

            return tuple(
                sorted(result, key=lambda item: item.track_id)
            )

    def end_track(
        self,
        camera_id: str,
        session_id: str,
        track_id: int,
    ) -> bool:
        """Remove one explicitly terminated track."""

        key = (camera_id, session_id, track_id)

        with self._lock:
            return self._histories.pop(key, None) is not None

    def clear_session(
        self,
        camera_id: str,
        session_id: str,
    ) -> int:
        """Remove all history belonging to one camera session."""

        with self._lock:

            keys = [
                key
                for key in self._histories
                if key[0] == camera_id and key[1] == session_id
            ]

            for key in keys:
                del self._histories[key]

            if self._camera_sessions.get(camera_id) == session_id:
                del self._camera_sessions[camera_id]

            return len(keys)

    def clear(self) -> None:
        """Clear all temporal state."""

        with self._lock:
            self._histories.clear()
            self._camera_sessions.clear()

    @property
    def active_track_count(self) -> int:

        with self._lock:
            return len(self._histories)