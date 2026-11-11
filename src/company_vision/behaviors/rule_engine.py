from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Literal

from company_vision.behaviors.plugin_registry import (
    BehaviorPluginContext,
    PluginResult,
)

RuleOperator = Literal["gt", "gte", "lt", "lte", "eq"]

RuleOutcome = Literal[
    "triggered",
    "not_triggered",
    "unknown",
    "not_applicable",
]


class RuleEngineError(ValueError):
    """Base error for rule engine failures."""


class RuleConfigurationError(RuleEngineError):
    """Invalid rule configuration."""


class RuleContractError(RuleEngineError):
    """Invalid plugin evidence or execution contract."""


@dataclass(frozen=True, slots=True)
class RuleDefinition:
    """Immutable configurable numeric behavior rule."""

    rule_id: str
    plugin_id: str
    metric_key: str
    operator: RuleOperator
    threshold: float

    min_confidence: float = 0.0
    requires_eligible_track: bool = True
    enabled: bool = True

    def __post_init__(self) -> None:

        for name in ("rule_id", "plugin_id", "metric_key"):
            value = getattr(self, name)

            if not isinstance(value, str) or not value.strip():
                raise RuleConfigurationError(
                    f"{name} must be a non-empty string"
                )

        if self.operator not in ("gt", "gte", "lt", "lte", "eq"):
            raise RuleConfigurationError("Unsupported operator")

        if isinstance(self.threshold, bool) or not isfinite(
            self.threshold
        ):
            raise RuleConfigurationError("Invalid threshold")

        if (
            isinstance(self.min_confidence, bool)
            or not isfinite(self.min_confidence)
            or not 0 <= self.min_confidence <= 1
        ):
            raise RuleConfigurationError("Invalid minimum confidence")

        if not isinstance(self.enabled, bool):
            raise RuleConfigurationError("enabled must be bool")

        if not isinstance(self.requires_eligible_track, bool):
            raise RuleConfigurationError(
                "requires_eligible_track must be bool"
            )


@dataclass(frozen=True, slots=True)
class RuleEvaluation:
    """Immutable evaluation output for one rule and observation."""

    rule_id: str
    plugin_id: str

    camera_id: str
    session_id: str
    frame_id: int

    track_id: int | None

    outcome: RuleOutcome

    metric_key: str
    observed_value: float | None
    threshold: float
    operator: RuleOperator

    confidence: float | None = None

    reason: str = ""

    evidence: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:

        if self.outcome not in (
            "triggered",
            "not_triggered",
            "unknown",
            "not_applicable",
        ):
            raise RuleContractError("Invalid rule outcome")

        if not isinstance(self.evidence, Mapping):
            raise RuleContractError("Evidence must be a mapping")

        object.__setattr__(
            self,
            "evidence",
            MappingProxyType(dict(self.evidence)),
        )


class RuleEngine:
    """
    Deterministic rule evaluator.

    Input:
        BehaviorPluginContext
        Sequence[PluginResult]

    Plugin payload contract:

        {
            "observations": [
                {
                    "track_id": 12,
                    "status": "ok",
                    "metrics": {
                        "score": 0.85
                    },
                    "confidence": 0.92,
                    "evidence": {}
                }
            ]
        }

    Observation status:
        ok
        unknown
        not_applicable

    Missing evidence never becomes a negative result.
    """

    def __init__(
        self,
        rules: Sequence[RuleDefinition],
    ) -> None:

        rules = tuple(rules)

        if not rules:
            raise RuleConfigurationError(
                "At least one rule is required"
            )

        if any(not isinstance(rule, RuleDefinition) for rule in rules):
            raise RuleConfigurationError(
                "All rules must be RuleDefinition instances"
            )

        identifiers = [rule.rule_id for rule in rules]

        if len(identifiers) != len(set(identifiers)):
            raise RuleConfigurationError(
                "Duplicate rule identifiers"
            )

        self._rules = tuple(
            sorted(rules, key=lambda rule: rule.rule_id)
        )

    @staticmethod
    def _compare(
        value: float,
        operator: RuleOperator,
        threshold: float,
    ) -> bool:

        if operator == "gt":
            return value > threshold

        if operator == "gte":
            return value >= threshold

        if operator == "lt":
            return value < threshold

        if operator == "lte":
            return value <= threshold

        return value == threshold

    @staticmethod
    def _evaluation(
        rule: RuleDefinition,
        context: BehaviorPluginContext,
        outcome: RuleOutcome,
        *,
        track_id: int | None = None,
        value: float | None = None,
        confidence: float | None = None,
        reason: str = "",
        evidence: Mapping[str, object] | None = None,
    ) -> RuleEvaluation:

        frame = context.frame

        return RuleEvaluation(
            rule_id=rule.rule_id,
            plugin_id=rule.plugin_id,
            camera_id=frame.camera_id,
            session_id=frame.session_id,
            frame_id=frame.frame_id,
            track_id=track_id,
            outcome=outcome,
            metric_key=rule.metric_key,
            observed_value=value,
            threshold=rule.threshold,
            operator=rule.operator,
            confidence=confidence,
            reason=reason,
            evidence=evidence or {},
        )

    @staticmethod
    def _validate_observation(
        observation: object,
    ) -> Mapping:

        if not isinstance(observation, Mapping):
            raise RuleContractError(
                "Each observation must be a mapping"
            )

        status = observation.get("status", "ok")

        if status not in ("ok", "unknown", "not_applicable"):
            raise RuleContractError(
                "Invalid observation status"
            )

        metrics = observation.get("metrics", {})

        if not isinstance(metrics, Mapping):
            raise RuleContractError(
                "Observation metrics must be a mapping"
            )

        evidence = observation.get("evidence", {})

        if not isinstance(evidence, Mapping):
            raise RuleContractError(
                "Observation evidence must be a mapping"
            )

        return observation

    def _evaluate_observation(
        self,
        rule: RuleDefinition,
        context: BehaviorPluginContext,
        observation: Mapping,
    ) -> RuleEvaluation:

        track_id = observation.get("track_id")

        if track_id is not None and (
            isinstance(track_id, bool)
            or not isinstance(track_id, int)
            or track_id < 0
        ):
            raise RuleContractError("Invalid observation track_id")

        status = observation.get("status", "ok")

        if status == "unknown":
            return self._evaluation(
                rule,
                context,
                "unknown",
                track_id=track_id,
                reason="Plugin observation is unknown",
            )

        if status == "not_applicable":
            return self._evaluation(
                rule,
                context,
                "not_applicable",
                track_id=track_id,
                reason="Plugin observation is not applicable",
            )

        if rule.requires_eligible_track:

            if track_id is None:
                return self._evaluation(
                    rule,
                    context,
                    "unknown",
                    reason="Missing track identity",
                )

            person = next(
                (
                    item
                    for item in context.frame.persons
                    if item.track_id == track_id
                ),
                None,
            )

            if person is None:
                return self._evaluation(
                    rule,
                    context,
                    "unknown",
                    track_id=track_id,
                    reason="Track absent from current frame",
                )

            if (
                not person.behavior_eligible
                or person.observed is not True
                or person.predicted is True
            ):
                return self._evaluation(
                    rule,
                    context,
                    "unknown",
                    track_id=track_id,
                    reason="Track is not eligible for behavior evaluation",
                )

        metrics = observation["metrics"]

        if rule.metric_key not in metrics:
            return self._evaluation(
                rule,
                context,
                "unknown",
                track_id=track_id,
                reason="Required metric is missing",
            )

        value = metrics[rule.metric_key]

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
        ):
            raise RuleContractError(
                f"Invalid numeric metric: {rule.metric_key}"
            )

        confidence = observation.get("confidence")

        if confidence is not None:

            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not isfinite(confidence)
                or not 0 <= confidence <= 1
            ):
                raise RuleContractError(
                    "Invalid observation confidence"
                )

            confidence = float(confidence)

        if (
            rule.min_confidence > 0
            and (confidence is None or confidence < rule.min_confidence)
        ):
            return self._evaluation(
                rule,
                context,
                "unknown",
                track_id=track_id,
                value=float(value),
                confidence=confidence,
                reason="Insufficient evidence confidence",
            )

        matched = self._compare(
            float(value),
            rule.operator,
            rule.threshold,
        )

        return self._evaluation(
            rule,
            context,
            "triggered" if matched else "not_triggered",
            track_id=track_id,
            value=float(value),
            confidence=confidence,
            reason=(
                "Rule threshold satisfied"
                if matched
                else "Rule threshold not satisfied"
            ),
            evidence=observation.get("evidence", {}),
        )

    def evaluate(
        self,
        context: BehaviorPluginContext,
        plugin_results: Sequence[PluginResult],
    ) -> tuple[RuleEvaluation, ...]:
        """Evaluate all configured rules against plugin evidence."""

        if not isinstance(context, BehaviorPluginContext):
            raise TypeError(
                "context must be BehaviorPluginContext"
            )

        results = tuple(plugin_results)

        if any(
            not isinstance(item, PluginResult)
            for item in results
        ):
            raise RuleContractError(
                "All inputs must be PluginResult instances"
            )

        frame = context.frame
        by_plugin = {}

        for result in results:

            if (
                result.camera_id != frame.camera_id
                or result.session_id != frame.session_id
                or result.frame_id != frame.frame_id
            ):
                raise RuleContractError(
                    "Plugin result identity does not match frame"
                )

            if result.plugin_id in by_plugin:
                raise RuleContractError(
                    "Duplicate plugin result"
                )

            by_plugin[result.plugin_id] = result

        evaluations = []

        for rule in self._rules:

            if not rule.enabled:
                evaluations.append(
                    self._evaluation(
                        rule,
                        context,
                        "not_applicable",
                        reason="Rule is disabled",
                    )
                )
                continue

            result = by_plugin.get(rule.plugin_id)

            if result is None:
                evaluations.append(
                    self._evaluation(
                        rule,
                        context,
                        "unknown",
                        reason="Plugin result is missing",
                    )
                )
                continue

            if result.status == "unknown":
                evaluations.append(
                    self._evaluation(
                        rule,
                        context,
                        "unknown",
                        reason="Plugin execution returned unknown",
                    )
                )
                continue

            if result.status == "not_applicable":
                evaluations.append(
                    self._evaluation(
                        rule,
                        context,
                        "not_applicable",
                        reason="Plugin is not applicable",
                    )
                )
                continue

            observations = result.payload.get("observations")

            if observations is None:
                evaluations.append(
                    self._evaluation(
                        rule,
                        context,
                        "unknown",
                        reason="Plugin supplied no observations",
                    )
                )
                continue

            if (
                not isinstance(observations, Sequence)
                or isinstance(observations, (str, bytes))
            ):
                raise RuleContractError(
                    "Plugin observations must be a sequence"
                )

            if not observations:
                evaluations.append(
                    self._evaluation(
                        rule,
                        context,
                        "unknown",
                        reason="No observations available",
                    )
                )
                continue

            for raw in observations:

                observation = self._validate_observation(raw)

                evaluations.append(
                    self._evaluate_observation(
                        rule,
                        context,
                        observation,
                    )
                )

        return tuple(evaluations)

    @property
    def rules(self) -> tuple[RuleDefinition, ...]:
        return self._rules

    @property
    def rule_count(self) -> int:
        return len(self._rules)
