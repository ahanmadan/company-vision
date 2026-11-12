"""Node 5.8.3: observable inactivity and posture evidence.

No medical, cognitive, sleep or productivity diagnosis is made.
This plugin does not operate a camera or infer intent.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite

from company_vision.behaviors.feature_extractor import PersonFeatures
from company_vision.behaviors.plugin_registry import (
    INACTIVITY_PLUGIN_ID,
    BehaviorPluginContext,
    PluginResult,
)


@dataclass(frozen=True, slots=True)
class InactivityConfig:
    min_duration_seconds: float = 6.0
    max_sample_gap_seconds: float = 4.0
    min_observations: int = 3
    max_body_movement: float = 0.035
    max_bbox_movement: float = 0.025
    min_point_quality: float = 0.25
    lowered_head_margin: float = 0.08

    def __post_init__(self) -> None:
        positive = (
            self.min_duration_seconds,
            self.max_sample_gap_seconds,
            self.max_body_movement,
            self.max_bbox_movement,
            self.lowered_head_margin,
        )

        if any(not isfinite(v) or v <= 0 for v in positive):
            raise ValueError("Positive finite configuration required")

        if (
            isinstance(self.min_observations, bool)
            or not isinstance(self.min_observations, int)
            or self.min_observations < 3
        ):
            raise ValueError("At least three observations are required")

        if not (
            isfinite(self.min_point_quality)
            and 0 < self.min_point_quality <= 1
        ):
            raise ValueError("Invalid keypoint quality threshold")


class InactivityPlugin:
    """Evaluate observable stillness per tracked person."""

    plugin_id = INACTIVITY_PLUGIN_ID
    version = "1.0"

    def __init__(self, config: InactivityConfig | None = None) -> None:
        self.config = config or InactivityConfig()

    @staticmethod
    def _eligible(person: PersonFeatures) -> bool:
        return (
            person.track_id is not None
            and person.behavior_eligible
            and person.observed is True
            and person.predicted is not True
        )

    def _landmarks(self, person: PersonFeatures):
        """Return trustworthy person-local landmark coordinates."""

        points = {}

        for item in person.keypoints:
            if (
                item.available
                and item.x_norm is not None
                and item.y_norm is not None
                and isfinite(item.x_norm)
                and isfinite(item.y_norm)
                and item.quality >= self.config.min_point_quality
            ):
                points[item.name] = (item.x_norm, item.y_norm)

        return points

    def _posture(self, person: PersonFeatures) -> dict:
        """Describe visible head position without inferring sleep."""

        if person.head_center is None or person.shoulder_center is None:
            return {
                "head_lowered": None,
                "head_shoulder_vertical_offset": None,
            }

        offset = (
            person.head_center[1] - person.shoulder_center[1]
        )

        if not isfinite(offset):
            return {
                "head_lowered": None,
                "head_shoulder_vertical_offset": None,
            }

        return {
            "head_lowered": offset >= self.config.lowered_head_margin,
            "head_shoulder_vertical_offset": round(offset, 6),
        }

    @staticmethod
    def _result(
        track_id: int,
        reason: str,
        *,
        score: float | None = None,
        confidence: float | None = None,
        evidence: dict | None = None,
    ) -> dict:
        result = {
            "track_id": track_id,
            "status": "ok" if score is not None else "unknown",
            "metrics": (
                {"score": round(score, 6)}
                if score is not None else {}
            ),
            "evidence": {
                "reason": reason,
                **(evidence or {}),
            },
        }

        if confidence is not None:
            result["confidence"] = round(confidence, 6)

        return result

    def _evaluate(self, context, person):
        track_id = person.track_id

        if not self._eligible(person):
            return self._result(track_id, "ineligible_track")

        now = context.frame.monotonic_timestamp

        if now is None or context.frame.source_status != "ok":
            return self._result(track_id, "invalid_frame_timing")

        history = next(
            (
                item for item in context.histories
                if item.track_id == track_id
            ),
            None,
        )

        if history is None:
            return self._result(track_id, "history_missing")

        records = [
            (
                item.monotonic_timestamp,
                item.person,
            )
            for item in history.observations
            if (
                item.monotonic_timestamp is not None
                and self._eligible(item.person)
                and item.monotonic_timestamp <= now
                and item.frame_id <= context.frame.frame_id
            )
        ]

        if (
            not records
            or records[-1][0] != now
            or records[-1][1].frame_id != person.frame_id
        ):
            records.append((now, person))

        # Sort and deduplicate by time; overlapping current
        # history samples must never count twice.
        by_time = {stamp: sample for stamp, sample in records}
        ordered = sorted(by_time.items())

        # Only uninterrupted recent observations count.
        contiguous = [ordered[-1]] if ordered else []

        for record in reversed(ordered[:-1]):
            newer = contiguous[0][0]

            if newer - record[0] > self.config.max_sample_gap_seconds:
                break

            contiguous.insert(0, record)

        if len(contiguous) < self.config.min_observations:
            return self._result(track_id, "insufficient_observations")

        duration = contiguous[-1][0] - contiguous[0][0]

        if duration < self.config.min_duration_seconds:
            return self._result(
                track_id,
                "insufficient_duration",
                evidence={"duration_seconds": round(duration, 3)},
            )

        # Head/shoulder alone is not enough; require a visible
        # arm landmark to distinguish meaningful upper-body motion.
        required = (
            "left_shoulder",
            "right_shoulder",
        )

        landmarks = [
            self._landmarks(sample)
            for _, sample in contiguous
        ]

        common = set.intersection(
            *(set(values) for values in landmarks)
        )

        if not set(required).issubset(common):
            return self._result(track_id, "shoulders_not_observable")

        arm_points = common.intersection({
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
        })

        if not arm_points:
            return self._result(track_id, "arms_not_observable")

        active_names = sorted(set(required).union(arm_points))

        # Compare every observation to every other observation.
        # This avoids missing movement that later returns
        # to its initial position.
        body_movement = 0.0
        bbox_movement = 0.0

        for i in range(len(contiguous)):
            person_a = contiguous[i][1]

            for j in range(i + 1, len(contiguous)):
                person_b = contiguous[j][1]

                bbox_distance = hypot(
                    person_a.bbox_center_x_norm
                    - person_b.bbox_center_x_norm,
                    person_a.bbox_center_y_norm
                    - person_b.bbox_center_y_norm,
                )

                bbox_movement = max(
                    bbox_movement,
                    bbox_distance,
                )

                for name in active_names:
                    a = landmarks[i][name]
                    b = landmarks[j][name]

                    movement = hypot(
                        a[0] - b[0],
                        a[1] - b[1],
                    )

                    body_movement = max(
                        body_movement,
                        movement,
                    )

        still = (
            body_movement <= self.config.max_body_movement
            and bbox_movement <= self.config.max_bbox_movement
        )

        posture = self._posture(person)

        confidence = min(
            0.95,
            0.65 + 0.025 * len(contiguous),
        )

        evidence = {
            "duration_seconds": round(duration, 3),
            "observation_count": len(contiguous),
            "body_movement": round(body_movement, 6),
            "bbox_movement": round(bbox_movement, 6),
            "landmarks": active_names,
            "posture": posture,
            "sleep_confirmed": False,
        }

        if still:
            return self._result(
                track_id,
                "sustained_observable_stillness",
                score=0.85,
                confidence=confidence,
                evidence=evidence,
            )

        return self._result(
            track_id,
            "observable_body_movement",
            score=0.0,
            confidence=confidence,
            evidence=evidence,
        )

    def execute(self, context: BehaviorPluginContext) -> PluginResult:
        if not isinstance(context, BehaviorPluginContext):
            raise TypeError("Expected BehaviorPluginContext")

        observations = []

        for person in context.frame.persons:
            if person.track_id is None:
                continue

            observations.append(
                self._evaluate(context, person)
            )

        return PluginResult(
            plugin_id=self.plugin_id,
            camera_id=context.frame.camera_id,
            session_id=context.frame.session_id,
            frame_id=context.frame.frame_id,
            status="ok",
            payload={"observations": observations},
        )