"""Node 5.9.3: execute four behavior plugins on a shared context.

No video capture, inference, rules, event generation or
final employee-state arbitration occurs in this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from company_vision.behaviors.plugin_registry import (
    BehaviorPluginContext,
    BehaviorPluginRegistry,
    PluginContractError,
    PluginResult,
)
from company_vision.behaviors.plugins.computer_usage import ComputerUsagePlugin
from company_vision.behaviors.plugins.gathering import GatheringPlugin
from company_vision.behaviors.plugins.inactivity import InactivityPlugin
from company_vision.behaviors.plugins.phone_usage import PhoneUsagePlugin
from company_vision.behaviors.schemas import BehaviorFrame
from company_vision.behaviors.temporal_context_pipeline import (
    BehaviorTemporalPipeline,
)

PLUGIN_ORDER = (
    "phone_usage",
    "computer_usage",
    "inactivity",
    "gathering",
)


class BehaviorExecutionError(RuntimeError):
    """Plugin results do not satisfy the integrated frame contract."""


@dataclass(frozen=True, slots=True)
class BehaviorExecutionBatch:
    """Evidence-only outputs for one camera/session/frame."""

    context: BehaviorPluginContext
    results: tuple[PluginResult, ...]
    per_track: Mapping[int, Mapping[str, Mapping]]

    @property
    def camera_id(self) -> str:
        return self.context.frame.camera_id

    @property
    def session_id(self) -> str:
        return self.context.frame.session_id

    @property
    def frame_id(self) -> int:
        return self.context.frame.frame_id

    @property
    def track_count(self) -> int:
        return len(self.per_track)

    @property
    def plugin_count(self) -> int:
        return len(self.results)

    def get_observation(self, track_id: int, plugin_id: str):
        """Return evidence for an existing track and plugin."""

        return self.per_track[track_id][plugin_id]


class BehaviorExecutionPipeline:
    """Run all four independent detectors without altering their logic."""

    def __init__(
        self,
        *,
        temporal: BehaviorTemporalPipeline | None = None,
        plugins: Sequence | None = None,
    ) -> None:
        self.temporal = (
            temporal if temporal is not None
            else BehaviorTemporalPipeline()
        )

        if plugins is None:
            plugins = (
                PhoneUsagePlugin(),
                ComputerUsagePlugin(),
                InactivityPlugin(),
                GatheringPlugin(),
            )

        plugins = tuple(plugins)

        if tuple(plugin.plugin_id for plugin in plugins) != PLUGIN_ORDER:
            raise ValueError(
                "Exactly four plugins in PLUGIN_ORDER are required"
            )

        self.registry = BehaviorPluginRegistry()
        self.registry.register_many(plugins)

    @staticmethod
    def _track_ids(context: BehaviorPluginContext) -> tuple[int, ...]:
        return tuple(
            sorted(
                person.track_id
                for person in context.frame.persons
                if person.track_id is not None
            )
        )

    @staticmethod
    def _validate_observation(
        item,
        *,
        expected_track_ids: set[int],
        plugin_id: str,
    ) -> int:
        if not isinstance(item, Mapping):
            raise BehaviorExecutionError(
                f"{plugin_id}: observation must be a mapping"
            )

        track_id = item.get("track_id")

        if (
            isinstance(track_id, bool)
            or not isinstance(track_id, int)
            or track_id not in expected_track_ids
        ):
            raise BehaviorExecutionError(
                f"{plugin_id}: unexpected track_id"
            )

        if item.get("status") not in ("ok", "unknown"):
            raise BehaviorExecutionError(
                f"{plugin_id}: invalid observation status"
            )

        metrics = item.get("metrics")

        if not isinstance(metrics, Mapping):
            raise BehaviorExecutionError(
                f"{plugin_id}: metrics must be a mapping"
            )

        evidence = item.get("evidence")

        if not isinstance(evidence, Mapping):
            raise BehaviorExecutionError(
                f"{plugin_id}: evidence must be a mapping"
            )

        if item["status"] == "unknown" and "score" in metrics:
            raise BehaviorExecutionError(
                f"{plugin_id}: unknown must not have a score"
            )

        return track_id

    @classmethod
    def _collect(
        cls,
        context: BehaviorPluginContext,
        results: tuple[PluginResult, ...],
    ) -> Mapping[int, Mapping[str, Mapping]]:
        expected_ids = cls._track_ids(context)
        expected_set = set(expected_ids)

        if tuple(result.plugin_id for result in results) != PLUGIN_ORDER:
            raise BehaviorExecutionError(
                "Incomplete or unordered plugin results"
            )

        by_track = {
            track_id: {}
            for track_id in expected_ids
        }

        for result in results:
            observations = result.payload.get("observations")

            if not isinstance(observations, (list, tuple)):
                raise BehaviorExecutionError(
                    f"{result.plugin_id}: missing observations"
                )

            seen = set()

            for observation in observations:
                track_id = cls._validate_observation(
                    observation,
                    expected_track_ids=expected_set,
                    plugin_id=result.plugin_id,
                )

                if track_id in seen:
                    raise BehaviorExecutionError(
                        f"{result.plugin_id}: duplicate observation"
                    )

                seen.add(track_id)

                # Keep each plugin's own status, metrics and
                # evidence. No cross-plugin classification.
                by_track[track_id][result.plugin_id] = (
                    MappingProxyType(dict(observation))
                )

            if seen != expected_set:
                missing = sorted(expected_set - seen)
                raise BehaviorExecutionError(
                    f"{result.plugin_id}: missing tracks {missing}"
                )

        return MappingProxyType({
            track_id: MappingProxyType(items)
            for track_id, items in by_track.items()
        })

    def execute_context(
        self,
        context: BehaviorPluginContext,
    ) -> BehaviorExecutionBatch:
        """Run every detector on one identical immutable context."""

        if not isinstance(context, BehaviorPluginContext):
            raise TypeError("Expected BehaviorPluginContext")

        results = self.registry.execute_all(
            context,
            plugin_ids=PLUGIN_ORDER,
        )

        # A disabled plugin must not silently disappear.
        if len(results) != len(PLUGIN_ORDER):
            raise PluginContractError(
                "All four behavior plugins must be enabled"
            )

        return BehaviorExecutionBatch(
            context=context,
            results=results,
            per_track=self._collect(context, results),
        )

    def process(
        self,
        frame: BehaviorFrame,
    ) -> BehaviorExecutionBatch:
        """Update the temporal pipeline, then execute all detectors."""

        context = self.temporal.process(frame)
        return self.execute_context(context)

    def process_tracking(self, **kwargs) -> BehaviorExecutionBatch:
        """Process inputs accepted by the Node 5.9.1 bridge."""

        context = self.temporal.process_tracking(**kwargs)
        return self.execute_context(context)

    def clear(self) -> None:
        """Reset histories and frame watermarks."""

        self.temporal.clear()

    def clear_session(self, camera_id: str, session_id: str) -> int:
        return self.temporal.clear_session(camera_id, session_id)