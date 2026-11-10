from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TrackLifecycleState:
    track_id: int
    last_observed_at: float
    last_seen_at: float
    consecutive_observations: int = 0
    consecutive_misses: int = 0


class TrackLifecycleGuard:
    """
    Separate tracking identity retention from visibility and
    behavior eligibility.

    Observed tracks may be published.
    Predicted tracks are hidden from the operational output.
    """

    def __init__(
        self,
        max_hidden_seconds: float = 0.75,
        min_confirmed_observations: int = 2,
    ) -> None:
        if max_hidden_seconds <= 0:
            raise ValueError("max_hidden_seconds must be positive")
        if min_confirmed_observations < 1:
            raise ValueError(
                "min_confirmed_observations must be >= 1"
            )

        self.max_hidden_seconds = max_hidden_seconds
        self.min_confirmed_observations = (
            min_confirmed_observations
        )
        self._states: dict[int, TrackLifecycleState] = {}

    def update(
        self,
        tracks: list[dict[str, Any]],
        timestamp: float,
    ) -> list[dict[str, Any]]:
        if timestamp < 0:
            raise ValueError("timestamp must be nonnegative")

        visible: list[dict[str, Any]] = []
        received_ids: set[int] = set()

        for track in tracks:
            track_id = int(track["track_id"])

            if track_id in received_ids:
                raise ValueError(
                    f"Duplicate track_id: {track_id}"
                )

            received_ids.add(track_id)

            observed = bool(track.get("observed", False))
            predicted = bool(track.get("predicted", False))

            if observed and predicted:
                raise ValueError(
                    "Track cannot be observed and predicted"
                )

            state = self._states.get(track_id)

            if state is None:
                if not observed:
                    # Never publish an unknown predicted track.
                    continue

                state = TrackLifecycleState(
                    track_id=track_id,
                    last_observed_at=timestamp,
                    last_seen_at=timestamp,
                )
                self._states[track_id] = state

            if timestamp < state.last_seen_at:
                raise ValueError(
                    "Non-monotonic timestamp for track"
                )

            state.last_seen_at = timestamp

            if not observed:
                state.consecutive_misses += 1
                state.consecutive_observations = 0
                continue

            if (
                timestamp - state.last_observed_at
                > self.max_hidden_seconds
            ):
                # Reconfirmation after a long observation gap.
                state.consecutive_observations = 0

            state.last_observed_at = timestamp
            state.consecutive_misses = 0
            state.consecutive_observations += 1

            if (
                state.consecutive_observations
                < self.min_confirmed_observations
            ):
                continue

            published = dict(track)
            published["behavior_eligible"] = True
            published["track_lifecycle"] = "observed"
            visible.append(published)

        expired_ids = [
            track_id
            for track_id, state in self._states.items()
            if (
                timestamp - state.last_observed_at
                > self.max_hidden_seconds
            )
        ]

        for track_id in expired_ids:
            del self._states[track_id]

        return visible

    def reset(self) -> None:
        self._states.clear()

    @property
    def retained_tracks(self) -> int:
        return len(self._states)
