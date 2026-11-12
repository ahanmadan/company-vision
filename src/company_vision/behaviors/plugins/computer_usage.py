"""Evidence-based computer activity plugin; no productivity inference."""
from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite
from typing import ClassVar

from company_vision.behaviors.feature_extractor import PersonFeatures
from company_vision.behaviors.plugin_registry import BehaviorPluginContext, PluginResult


@dataclass(frozen=True, slots=True)
class ComputerUsageConfig:
    min_object_confidence: float = 0.4
    max_hand_distance: float = 0.32
    min_hand_movement: float = 0.018
    min_history_seconds: float = 0.4
    max_history_seconds: float = 10.0
    ambiguity_margin: float = 0.06

    def __post_init__(self) -> None:
        values = (
            self.min_object_confidence, self.max_hand_distance,
            self.min_hand_movement, self.min_history_seconds,
            self.max_history_seconds, self.ambiguity_margin,
        )
        if any(not isfinite(v) or v < 0 for v in values):
            raise ValueError("Configuration requires finite nonnegative values")
        if not 0 < self.min_object_confidence <= 1:
            raise ValueError("min_object_confidence must be in (0,1]")
        if self.max_hand_distance <= 0 or self.min_history_seconds <= 0:
            raise ValueError("Distance and duration must be positive")
        if self.max_history_seconds < self.min_history_seconds:
            raise ValueError("Invalid history interval")


class ComputerUsagePlugin:
    plugin_id = "computer_usage"
    version = "1.0"

    # COCO-80: mouse=64, laptop=63, keyboard=66. A monitor alone is insufficient.
    _INPUT_DEVICES: ClassVar[dict[int, str]] = {64: "mouse", 66: "keyboard"}
    _LAPTOP = 63

    def __init__(self, config: ComputerUsageConfig | None = None) -> None:
        self.config = config or ComputerUsageConfig()

    @staticmethod
    def _hand(person: PersonFeatures, name: str) -> tuple[float, float] | None:
        for point in person.keypoints:
            if point.name != name or not point.available:
                continue
            if point.x_norm is None or point.y_norm is None or point.quality <= 0:
                continue
            return (
                person.bbox_center_x_norm
                + (point.x_norm - 0.5) * person.bbox_width_norm,
                person.bbox_center_y_norm
                + (point.y_norm - 0.5) * person.bbox_height_norm,
            )
        return None

    def _movement(self, context: BehaviorPluginContext, person: PersonFeatures,
                  hand_name: str) -> float | None:
        current = self._hand(person, hand_name)
        now = context.frame.monotonic_timestamp
        if current is None or now is None:
            return None
        scale = max(hypot(person.bbox_width_norm, person.bbox_height_norm), 1e-6)
        for history in context.histories:
            if history.track_id != person.track_id:
                continue
            # Use the oldest eligible sample in the configured time window.
            for observation in history.observations:
                then = observation.monotonic_timestamp
                if then is None:
                    continue
                age = now - then
                if not self.config.min_history_seconds <= age <= self.config.max_history_seconds:
                    continue
                previous = observation.person
                if previous.observed is not True or not previous.behavior_eligible:
                    continue
                old_point = self._hand(previous, hand_name)
                if old_point is not None:
                    return hypot(current[0] - old_point[0], current[1] - old_point[1]) / scale
        return None

    @staticmethod
    def _obs(track_id: int, reason: str, score: float | None = None,
             confidence: float | None = None, evidence: dict | None = None) -> dict:
        result = {
            "track_id": track_id,
            "status": "ok" if score is not None else "unknown",
            "metrics": {"score": round(score, 6)} if score is not None else {},
            "evidence": {"reason": reason, "work_state": (
                "computer_activity_observed" if score is not None and score >= 0.7
                else "default_work_assumed"
            ), **(evidence or {})},
        }
        if confidence is not None:
            result["confidence"] = round(confidence, 6)
        return result

    def execute(self, context: BehaviorPluginContext) -> PluginResult:
        if not isinstance(context, BehaviorPluginContext):
            raise TypeError("Expected BehaviorPluginContext")
        frame = context.frame
        observations = []
        available = frame.object_detection_status == "available" and frame.source_status == "ok"
        devices = [obj for obj in frame.objects if obj.confidence >= self.config.min_object_confidence
                   and (obj.class_id in self._INPUT_DEVICES or obj.class_id == self._LAPTOP)]
        eligible = [p for p in frame.persons if p.track_id is not None and p.behavior_eligible
                    and p.observed is True and p.predicted is not True]
        # Each detected device belongs to at most one person; ambiguity remains unknown.
        candidates = []
        for i, device in enumerate(devices):
            for person in eligible:
                scale = max(hypot(person.bbox_width_norm, person.bbox_height_norm), 1e-6)
                for name in ("left_wrist", "right_wrist"):
                    hand = self._hand(person, name)
                    if hand is None:
                        continue
                    distance = hypot(hand[0] - device.center_x_norm,
                                     hand[1] - device.center_y_norm) / scale
                    if distance <= self.config.max_hand_distance:
                        candidates.append((distance, i, person.track_id, name))
        candidates.sort()
        assignments = {}
        used_devices = set()
        for distance, device_index, track_id, hand_name in candidates:
            if device_index in used_devices or track_id in assignments:
                continue
            rivals = [c for c in candidates if c[1] == device_index and c[2] != track_id
                      and abs(c[0] - distance) < self.config.ambiguity_margin]
            if rivals:
                continue
            assignments[track_id] = (device_index, hand_name, distance)
            used_devices.add(device_index)

        for person in frame.persons:
            if person.track_id is None:
                continue
            track_id = person.track_id
            if person not in eligible:
                observations.append(self._obs(track_id, "ineligible_track"))
                continue
            if not available:
                observations.append(self._obs(track_id, "object_detection_unavailable"))
                continue
            assignment = assignments.get(track_id)
            if assignment is None:
                observations.append(self._obs(track_id, "interaction_not_verified"))
                continue
            device_index, hand_name, distance = assignment
            device = devices[device_index]
            movement = self._movement(context, person, hand_name)
            if movement is None:
                observations.append(self._obs(track_id, "temporal_evidence_missing",
                                              evidence={"device": device.class_name}))
                continue
            if movement < self.config.min_hand_movement:
                observations.append(self._obs(track_id, "hand_motion_not_verified",
                                              evidence={"movement": round(movement, 6)}))
                continue
            closeness = max(0.0, 1.0 - distance / self.config.max_hand_distance)
            score = min(1.0, 0.75 + 0.18 * closeness)
            observations.append(self._obs(
                track_id, "hand_motion_near_input_device", score=score,
                confidence=device.confidence * 0.9,
                evidence={"device": device.class_name, "hand": hand_name,
                          "normalized_distance": round(distance, 6),
                          "movement": round(movement, 6)},
            ))
        return PluginResult(plugin_id=self.plugin_id, camera_id=frame.camera_id,
                            session_id=frame.session_id, frame_id=frame.frame_id,
                            status="ok", payload={"observations": observations})