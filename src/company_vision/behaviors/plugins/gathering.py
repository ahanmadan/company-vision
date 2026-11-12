"""Node 5.8.4: sustained spatial grouping evidence.

Camera-perspective heuristics are provisional.
Proximity is not evidence of conversation, intent or misconduct.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite

from company_vision.behaviors.feature_extractor import PersonFeatures
from company_vision.behaviors.plugin_registry import (
    GATHERING_PLUGIN_ID,
    BehaviorPluginContext,
    PluginResult,
)


@dataclass(frozen=True, slots=True)
class GatheringConfig:
    min_people: int = 3
    min_duration_seconds: float = 5.0
    max_sample_gap_seconds: float = 2.5
    max_time_skew_seconds: float = 0.6
    max_distance_body_heights: float = 1.4
    max_height_ratio: float = 1.8
    min_observations: int = 3

    def __post_init__(self) -> None:
        if (
            isinstance(self.min_people, bool)
            or not isinstance(self.min_people, int)
            or self.min_people < 2
        ):
            raise ValueError("min_people must be >= 2")

        if (
            isinstance(self.min_observations, bool)
            or not isinstance(self.min_observations, int)
            or self.min_observations < 3
        ):
            raise ValueError("min_observations must be >= 3")

        values = (
            self.min_duration_seconds,
            self.max_sample_gap_seconds,
            self.max_time_skew_seconds,
            self.max_distance_body_heights,
            self.max_height_ratio,
        )

        if any(not isfinite(x) or x <= 0 for x in values):
            raise ValueError("Configuration values must be finite and positive")

        if self.max_height_ratio < 1:
            raise ValueError("max_height_ratio must be >= 1")


class GatheringPlugin:
    """Evaluate per-track group proximity and temporal persistence."""

    plugin_id = GATHERING_PLUGIN_ID
    version = "1.0"

    def __init__(self, config: GatheringConfig | None = None):
        self.config = config or GatheringConfig()

    @staticmethod
    def _eligible(person: PersonFeatures) -> bool:
        return (
            person.track_id is not None
            and person.behavior_eligible
            and person.observed is True
            and person.predicted is not True
            and person.bbox_height_norm > 0
            and person.bbox_width_norm > 0
        )

    def _near(
        self,
        a: PersonFeatures,
        b: PersonFeatures,
        width: int,
        height: int,
    ) -> tuple[bool, float, float]:
        """Use bottom-centre image anchors and apparent body height."""

        height_a = a.bbox_height_norm * height
        height_b = b.bbox_height_norm * height

        if min(height_a, height_b) <= 0:
            return False, float("inf"), float("inf")

        ratio = max(height_a, height_b) / min(height_a, height_b)

        if ratio > self.config.max_height_ratio:
            return False, float("inf"), ratio

        ax = a.bbox_center_x_norm * width
        ay = (a.bbox_center_y_norm + a.bbox_height_norm / 2) * height
        bx = b.bbox_center_x_norm * width
        by = (b.bbox_center_y_norm + b.bbox_height_norm / 2) * height

        distance = hypot(ax - bx, ay - by)
        normalized = distance / ((height_a + height_b) / 2)

        return (
            normalized <= self.config.max_distance_body_heights,
            normalized,
            ratio,
        )

    def _groups(self, people, width, height):
        """Conservative complete-link groups, not chain clusters."""

        remaining = sorted(
            people,
            key=lambda item: item.track_id,
        )

        groups = []

        while remaining:
            seed = remaining.pop(0)
            group = [seed]

            for candidate in remaining[:]:
                if all(
                    self._near(member, candidate, width, height)[0]
                    for member in group
                ):
                    group.append(candidate)
                    remaining.remove(candidate)

            if len(group) >= self.config.min_people:
                groups.append(tuple(
                    sorted(member.track_id for member in group)
                ))

        return tuple(groups)

    @staticmethod
    def _obs(
        track_id,
        reason,
        *,
        score=None,
        confidence=None,
        group=None,
        duration=None,
    ):
        result = {
            "track_id": track_id,
            "status": "ok" if score is not None else "unknown",
            "metrics": (
                {"score": round(score, 6)}
                if score is not None else {}
            ),
            "evidence": {
                "reason": reason,
                "group_members": list(group) if group else [],
                "duration_seconds": (
                    round(duration, 3)
                    if duration is not None else None
                ),
            },
        }

        if confidence is not None:
            result["confidence"] = round(confidence, 6)

        return result

    def _history_records(self, context, track_id, now):
        history = next(
            (
                value for value in context.histories
                if value.track_id == track_id
            ),
            None,
        )

        if history is None:
            return None

        current = next(
            (p for p in context.frame.persons if p.track_id == track_id),
            None,
        )

        if current is None or not self._eligible(current):
            return None

        records = []

        for item in history.observations:
            stamp = item.monotonic_timestamp
            if (
                stamp is not None
                and isfinite(stamp)
                and stamp <= now
                and item.frame_id <= context.frame.frame_id
                and self._eligible(item.person)
            ):
                records.append((stamp, item.person))

        records.append((now, current))

        # Current observation replaces any same-time history sample.
        by_time = dict(records)
        ordered = sorted(by_time.items())

        if len(ordered) < self.config.min_observations:
            return None

        start = now - self.config.min_duration_seconds

        # Need a real observation sufficiently close to the
        # beginning of the evaluation interval.
        before = [
            item for item in ordered
            if item[0] <= start
            and start - item[0] <= self.config.max_sample_gap_seconds
        ]

        if not before:
            return None

        baseline = before[-1][0]
        window = [
            item for item in ordered
            if baseline <= item[0] <= now
        ]

        if len(window) < self.config.min_observations:
            return None

        for i in range(1, len(window)):
            if (
                window[i][0] - window[i - 1][0]
                > self.config.max_sample_gap_seconds
            ):
                return None

        return window

    def _persistent(self, context, member_ids, now):
        """Check three aligned group snapshots across the time window."""

        records = {}

        for track_id in member_ids:
            values = self._history_records(context, track_id, now)

            if values is None:
                return False, "insufficient_temporal_evidence", None

            records[track_id] = values

        start = now - self.config.min_duration_seconds
        middle = start + self.config.min_duration_seconds / 2

        probes = (start, middle, now)

        for probe in probes:
            selected = []

            for track_id in member_ids:
                candidates = [
                    (abs(stamp - probe), sample)
                    for stamp, sample in records[track_id]
                    if abs(stamp - probe) <= self.config.max_time_skew_seconds
                ]

                if not candidates:
                    return False, "missing_aligned_observation", None

                _, sample = min(candidates, key=lambda item: item[0])
                selected.append(sample)

            for i, a in enumerate(selected):
                for b in selected[i + 1:]:
                    near, _, _ = self._near(
                        a,
                        b,
                        context.frame.width,
                        context.frame.height,
                    )

                    if not near:
                        return False, "group_not_persistent", None

        return True, "sustained_group_proximity", self.config.min_duration_seconds

    def execute(self, context: BehaviorPluginContext) -> PluginResult:
        if not isinstance(context, BehaviorPluginContext):
            raise TypeError("Expected BehaviorPluginContext")

        frame = context.frame
        now = frame.monotonic_timestamp

        eligible = [
            p for p in frame.persons if self._eligible(p)
        ]

        groups = (
            self._groups(eligible, frame.width, frame.height)
            if frame.source_status == "ok" else ()
        )

        group_by_track = {
            track_id: group
            for group in groups
            for track_id in group
        }

        observations = []

        for person in frame.persons:
            track_id = person.track_id

            if track_id is None:
                continue

            if not self._eligible(person):
                observations.append(
                    self._obs(track_id, "ineligible_track")
                )
                continue

            if now is None or frame.source_status != "ok":
                observations.append(
                    self._obs(track_id, "timing_or_source_unavailable")
                )
                continue

            group = group_by_track.get(track_id)

            if group is None:
                observations.append(self._obs(
                    track_id,
                    "no_current_group",
                    score=0.0,
                    confidence=0.75,
                ))
                continue

            persistent, reason, duration = self._persistent(
                context,
                group,
                now,
            )

            if not persistent:
                observations.append(self._obs(
                    track_id,
                    reason,
                    group=group,
                ))
                continue

            observations.append(self._obs(
                track_id,
                reason,
                score=0.9,
                confidence=0.8,
                group=group,
                duration=duration,
            ))

        return PluginResult(
            plugin_id=self.plugin_id,
            camera_id=frame.camera_id,
            session_id=frame.session_id,
            frame_id=frame.frame_id,
            status="ok",
            payload={"observations": observations},
        )