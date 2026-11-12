"""Phone usage evidence plugin.

This module does not run an object detector or a camera.
It consumes previously detected objects and person features.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite

from company_vision.behaviors.feature_extractor import (
    ObjectFeatures,
    PersonFeatures,
)
from company_vision.behaviors.plugin_registry import (
    PHONE_USAGE_PLUGIN_ID,
    BehaviorPluginContext,
    PluginResult,
)


@dataclass(frozen=True, slots=True)
class PhoneUsageConfig:
    min_phone_confidence: float = 0.35
    max_arm_distance: float = 0.42
    min_person_margin: float = 0.20

    def __post_init__(self) -> None:
        for value in (
            self.min_phone_confidence,
            self.max_arm_distance,
            self.min_person_margin,
        ):
            if not isfinite(value) or value < 0:
                raise ValueError("Invalid phone usage configuration")

        if not 0 < self.min_phone_confidence <= 1:
            raise ValueError("Invalid phone confidence threshold")

        if self.max_arm_distance <= 0:
            raise ValueError("Invalid arm distance threshold")


class PhoneUsagePlugin:
    """Conservative phone-to-person association detector."""

    plugin_id = PHONE_USAGE_PLUGIN_ID
    version = "1.0"

    _PHONE_NAMES = frozenset({
        "cell phone",
        "mobile phone",
        "cellphone",
        "smartphone",
    })

    def __init__(self, config: PhoneUsageConfig | None = None):
        self.config = config or PhoneUsageConfig()

    @classmethod
    def _is_phone(cls, obj: ObjectFeatures) -> bool:
        return (
            obj.class_id == 67
            or obj.class_name.strip().lower() in cls._PHONE_NAMES
        )

    @staticmethod
    def _global_point(person: PersonFeatures, name: str):
        for point in person.keypoints:
            if point.name != name or not point.available:
                continue

            if point.x_norm is None or point.y_norm is None:
                continue

            x1 = (
                person.bbox_center_x_norm
                - person.bbox_width_norm / 2
            )
            y1 = (
                person.bbox_center_y_norm
                - person.bbox_height_norm / 2
            )

            return (
                x1 + point.x_norm * person.bbox_width_norm,
                y1 + point.y_norm * person.bbox_height_norm,
            )

        return None

    def _association(
        self,
        person: PersonFeatures,
        phone: ObjectFeatures,
    ):
        """Return candidate distance and wrist availability."""

        px = phone.center_x_norm
        py = phone.center_y_norm

        cx = person.bbox_center_x_norm
        cy = person.bbox_center_y_norm

        w = person.bbox_width_norm
        h = person.bbox_height_norm

        if w <= 0 or h <= 0:
            return None

        margin_x = self.config.min_person_margin * w
        margin_y = self.config.min_person_margin * h

        if (
            abs(px - cx) > w / 2 + margin_x
            or abs(py - cy) > h / 2 + margin_y
        ):
            return None

        body_scale = max(hypot(w, h), 1e-6)

        candidates = []

        for name in (
            "left_wrist",
            "right_wrist",
            "left_elbow",
            "right_elbow",
        ):
            point = self._global_point(person, name)

            if point is None:
                continue

            distance = hypot(
                px - point[0], py - point[1]
            ) / body_scale

            candidates.append((
                distance,
                name.endswith("wrist"),
                name,
            ))

        if not candidates:
            return ("unknown", None, False, None)

        distance, wrist, landmark = min(
            candidates,
            key=lambda item: item[0],
        )

        if distance > self.config.max_arm_distance:
            return None

        return ("candidate", distance, wrist, landmark)

    @staticmethod
    def _observation(
        track_id: int,
        status: str,
        *,
        score: float | None = None,
        confidence: float | None = None,
        evidence: dict | None = None,
    ) -> dict:
        record = {
            "track_id": track_id,
            "status": status,
            "metrics": {},
            "evidence": evidence or {},
        }

        if score is not None:
            record["metrics"]["score"] = round(score, 6)

        if confidence is not None:
            record["confidence"] = round(confidence, 6)

        return record

    def execute(
        self,
        context: BehaviorPluginContext,
    ) -> PluginResult:
        if not isinstance(context, BehaviorPluginContext):
            raise TypeError("Expected BehaviorPluginContext")

        frame = context.frame

        observations = []

        # If object detection did not run, absence of a phone
        # must never be treated as negative evidence.
        detection_available = (
            frame.object_detection_status == "available"
            and frame.source_status == "ok"
        )

        phones = [
            obj
            for obj in frame.objects
            if self._is_phone(obj)
            and obj.confidence >= self.config.min_phone_confidence
        ]

        eligible = [
            person for person in frame.persons
            if (
                person.track_id is not None
                and person.behavior_eligible
                and person.observed is True
                and person.predicted is not True
            )
        ]

        candidates = []

        if detection_available:
            for phone_index, phone in enumerate(phones):
                for person in eligible:
                    match = self._association(person, phone)

                    if match is None or match[0] != "candidate":
                        continue

                    _, distance, wrist, landmark = match

                    candidates.append((
                        distance,
                        phone_index,
                        person.track_id,
                        wrist,
                        landmark,
                        phone.confidence,
                    ))

        # One phone cannot be assigned to multiple people.
        candidates.sort(key=lambda item: item[0])

        assignments = {}
        assigned_phones = set()

        for candidate in candidates:
            (
                distance,
                phone_index,
                track_id,
                wrist,
                landmark,
                phone_confidence,
            ) = candidate

            if track_id in assignments:
                continue

            if phone_index in assigned_phones:
                continue

            # Ambiguous equally-close association is not
            # accepted as positive evidence.
            alternatives = [
                item
                for item in candidates
                if (
                    item[1] == phone_index
                    and item[2] != track_id
                    and abs(item[0] - distance) < 0.08
                )
            ]

            if alternatives:
                continue

            assignments[track_id] = candidate
            assigned_phones.add(phone_index)

        for person in frame.persons:
            track_id = person.track_id

            if track_id is None:
                continue

            if (
                not person.behavior_eligible
                or person.observed is not True
                or person.predicted is True
            ):
                observations.append(self._observation(
                    track_id, "unknown",
                    evidence={"reason": "ineligible_track"},
                ))
                continue

            if not detection_available:
                observations.append(self._observation(
                    track_id, "unknown",
                    evidence={"reason": "object_detection_unavailable"},
                ))
                continue

            candidate = assignments.get(track_id)

            if candidate is not None:
                (
                    distance,
                    phone_index,
                    _,
                    wrist,
                    landmark,
                    phone_confidence,
                ) = candidate

                closeness = max(
                    0.0,
                    1.0 - distance / self.config.max_arm_distance,
                )

                score = min(
                    1.0,
                    0.78 + 0.14 * closeness
                    + (0.06 if wrist else 0.0),
                )

                confidence = min(
                    1.0,
                    phone_confidence * (
                        0.82 + (0.16 if wrist else 0.0)
                    ),
                )

                observations.append(self._observation(
                    track_id,
                    "ok",
                    score=score,
                    confidence=confidence,
                    evidence={
                        "reason": "phone_near_arm",
                        "phone_index": phone_index,
                        "landmark": landmark,
                        "normalized_distance": round(distance, 6),
                        "wrist_evidence": wrist,
                        "phone_confidence": phone_confidence,
                    },
                ))
                continue

            # A visible phone without reliable association
            # is not proof that this person is using it.
            nearby = any(
                self._association(person, phone) is not None
                for phone in phones
            )

            if nearby:
                observations.append(self._observation(
                    track_id,
                    "unknown",
                    evidence={"reason": "ambiguous_phone_association"},
                ))
            else:
                observations.append(self._observation(
                    track_id,
                    "ok",
                    score=0.0,
                    confidence=0.8,
                    evidence={"reason": "no_associated_phone"},
                ))

        return PluginResult(
            plugin_id=self.plugin_id,
            camera_id=frame.camera_id,
            session_id=frame.session_id,
            frame_id=frame.frame_id,
            status="ok",
            payload={"observations": observations},
        )