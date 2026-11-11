from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from threading import RLock
from types import MappingProxyType
from typing import Literal, Protocol

from company_vision.behaviors.feature_extractor import FrameFeatures
from company_vision.behaviors.temporal_state_manager import (
    TemporalTrackHistory,
)

PHONE_USAGE_PLUGIN_ID = "phone_usage"
INACTIVITY_PLUGIN_ID = "inactivity"
GATHERING_PLUGIN_ID = "gathering"

INITIAL_PLUGIN_IDS = (
    PHONE_USAGE_PLUGIN_ID,
    INACTIVITY_PLUGIN_ID,
    GATHERING_PLUGIN_ID,
)


class PluginRegistryError(RuntimeError):
    """Base exception for behavior plugin registry errors."""


class PluginContractError(PluginRegistryError):
    """Raised when a plugin violates its contract."""


class DuplicatePluginError(PluginRegistryError):
    """Raised when a plugin ID is already registered."""


class PluginNotFoundError(PluginRegistryError):
    """Raised when a requested plugin does not exist."""


class PluginDisabledError(PluginRegistryError):
    """Raised when an inactive plugin is executed directly."""


class PluginExecutionError(PluginRegistryError):
    """Raised when a plugin fails during execution."""

    def __init__(self, plugin_id: str, message: str):
        self.plugin_id = plugin_id
        super().__init__(
            f"Plugin '{plugin_id}' execution failed: {message}"
        )


@dataclass(frozen=True, slots=True)
class BehaviorPluginContext:
    """Immutable input context shared with behavior plugins."""

    frame: FrameFeatures
    histories: tuple[TemporalTrackHistory, ...] = ()

    def __post_init__(self) -> None:

        if not isinstance(self.frame, FrameFeatures):
            raise TypeError("frame must be FrameFeatures")

        histories = tuple(self.histories)

        seen: set[int] = set()

        for history in histories:

            if not isinstance(history, TemporalTrackHistory):
                raise TypeError(
                    "histories must contain TemporalTrackHistory"
                )

            if (
                history.camera_id != self.frame.camera_id
                or history.session_id != self.frame.session_id
            ):
                raise PluginContractError(
                    "History camera/session does not match frame"
                )

            if history.latest.frame_id > self.frame.frame_id:
                raise PluginContractError(
                    "History cannot contain future observations"
                )

            if history.track_id in seen:
                raise PluginContractError(
                    "Duplicate track history"
                )

            seen.add(history.track_id)

        object.__setattr__(self, "histories", histories)


@dataclass(frozen=True, slots=True)
class PluginResult:
    """Standardized result returned by a behavior plugin."""

    plugin_id: str
    camera_id: str
    session_id: str
    frame_id: int

    status: Literal[
        "ok",
        "unknown",
        "not_applicable",
    ] = "ok"

    payload: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:

        if not self.plugin_id:
            raise PluginContractError("plugin_id is required")

        if not self.camera_id or not self.session_id:
            raise PluginContractError(
                "Camera and session IDs are required"
            )

        if (
            isinstance(self.frame_id, bool)
            or not isinstance(self.frame_id, int)
            or self.frame_id < 0
        ):
            raise PluginContractError("Invalid frame_id")

        if self.status not in (
            "ok",
            "unknown",
            "not_applicable",
        ):
            raise PluginContractError("Invalid plugin result status")

        if not isinstance(self.payload, Mapping):
            raise PluginContractError(
                "Plugin payload must be a mapping"
            )

        object.__setattr__(
            self,
            "payload",
            MappingProxyType(dict(self.payload)),
        )


class BehaviorPlugin(Protocol):
    """Structural interface implemented by every behavior plugin."""

    @property
    def plugin_id(self) -> str:
        ...

    @property
    def version(self) -> str:
        ...

    def execute(
        self,
        context: BehaviorPluginContext,
    ) -> PluginResult:
        ...


@dataclass(frozen=True, slots=True)
class PluginDescriptor:
    plugin_id: str
    version: str
    enabled: bool


@dataclass(slots=True)
class _RegistryEntry:
    plugin: BehaviorPlugin
    enabled: bool


class BehaviorPluginRegistry:
    """
    Thread-safe registry for independent behavior plugins.

    Responsibilities:
        - Register plugins.
        - Prevent duplicate IDs.
        - Retrieve plugins.
        - Enable and disable plugins.
        - Unregister plugins.
        - Execute individual plugins.
        - Execute all enabled plugins deterministically.

    Non-responsibilities:
        - Behavior classification.
        - Rule evaluation.
        - Threshold management.
        - Event lifecycle.
        - Camera or model execution.
    """

    _PLUGIN_ID_PATTERN = re.compile(
        r"^[a-z][a-z0-9_.-]*$"
    )

    def __init__(self) -> None:

        self._plugins: dict[str, _RegistryEntry] = {}
        self._lock = RLock()

    @classmethod
    def _validate_plugin(cls, plugin: object) -> tuple[str, str]:

        plugin_id = getattr(plugin, "plugin_id", None)
        version = getattr(plugin, "version", None)

        if (
            not isinstance(plugin_id, str)
            or not cls._PLUGIN_ID_PATTERN.fullmatch(plugin_id)
        ):
            raise PluginContractError(
                "Plugin ID must use lowercase identifier format"
            )

        if not isinstance(version, str) or not version.strip():
            raise PluginContractError(
                "Plugin version must be a non-empty string"
            )

        if not callable(getattr(plugin, "execute", None)):
            raise PluginContractError(
                "Plugin must implement execute(context)"
            )

        return plugin_id, version

    @staticmethod
    def _descriptor(entry: _RegistryEntry) -> PluginDescriptor:

        return PluginDescriptor(
            plugin_id=entry.plugin.plugin_id,
            version=entry.plugin.version,
            enabled=entry.enabled,
        )

    def register(
        self,
        plugin: BehaviorPlugin,
        *,
        enabled: bool = True,
    ) -> PluginDescriptor:

        if not isinstance(enabled, bool):
            raise TypeError("enabled must be bool")

        plugin_id, _ = self._validate_plugin(plugin)

        with self._lock:

            if plugin_id in self._plugins:
                raise DuplicatePluginError(
                    f"Plugin already registered: {plugin_id}"
                )

            entry = _RegistryEntry(
                plugin=plugin,
                enabled=enabled,
            )

            self._plugins[plugin_id] = entry

            return self._descriptor(entry)

    def register_many(
        self,
        plugins: Sequence[BehaviorPlugin],
        *,
        enabled: bool = True,
    ) -> tuple[PluginDescriptor, ...]:

        if not isinstance(enabled, bool):
            raise TypeError("enabled must be bool")

        plugins = tuple(plugins)

        validated = [
            (plugin, self._validate_plugin(plugin)[0])
            for plugin in plugins
        ]

        ids = [plugin_id for _, plugin_id in validated]

        if len(ids) != len(set(ids)):
            raise DuplicatePluginError(
                "Duplicate IDs inside registration batch"
            )

        with self._lock:

            duplicates = set(ids).intersection(self._plugins)

            if duplicates:
                raise DuplicatePluginError(
                    f"Already registered: {sorted(duplicates)}"
                )

            staged = dict(self._plugins)

            for plugin, plugin_id in validated:
                staged[plugin_id] = _RegistryEntry(
                    plugin=plugin,
                    enabled=enabled,
                )

            self._plugins = staged

            return tuple(
                self._descriptor(self._plugins[plugin_id])
                for plugin_id in ids
            )

    def get(self, plugin_id: str) -> BehaviorPlugin:

        with self._lock:

            entry = self._plugins.get(plugin_id)

            if entry is None:
                raise PluginNotFoundError(
                    f"Unknown plugin: {plugin_id}"
                )

            return entry.plugin

    def get_descriptor(self, plugin_id: str) -> PluginDescriptor:

        with self._lock:

            entry = self._plugins.get(plugin_id)

            if entry is None:
                raise PluginNotFoundError(
                    f"Unknown plugin: {plugin_id}"
                )

            return self._descriptor(entry)

    def list_plugins(
        self,
        *,
        enabled_only: bool = False,
    ) -> tuple[PluginDescriptor, ...]:

        if not isinstance(enabled_only, bool):
            raise TypeError("enabled_only must be bool")

        with self._lock:

            entries = []

            for plugin_id in sorted(self._plugins):

                entry = self._plugins[plugin_id]

                if enabled_only and not entry.enabled:
                    continue

                entries.append(self._descriptor(entry))

            return tuple(entries)

    def set_enabled(
        self,
        plugin_id: str,
        enabled: bool,
    ) -> PluginDescriptor:

        if not isinstance(enabled, bool):
            raise TypeError("enabled must be bool")

        with self._lock:

            entry = self._plugins.get(plugin_id)

            if entry is None:
                raise PluginNotFoundError(
                    f"Unknown plugin: {plugin_id}"
                )

            entry.enabled = enabled

            return self._descriptor(entry)

    def unregister(self, plugin_id: str) -> bool:

        with self._lock:
            return self._plugins.pop(plugin_id, None) is not None

    @staticmethod
    def _validate_result(
        plugin_id: str,
        context: BehaviorPluginContext,
        result: object,
    ) -> PluginResult:

        if not isinstance(result, PluginResult):
            raise PluginContractError(
                "Plugin must return PluginResult"
            )

        frame = context.frame

        if result.plugin_id != plugin_id:
            raise PluginContractError(
                "Result plugin ID mismatch"
            )

        if (
            result.camera_id != frame.camera_id
            or result.session_id != frame.session_id
            or result.frame_id != frame.frame_id
        ):
            raise PluginContractError(
                "Plugin result frame identity mismatch"
            )

        return result

    def execute(
        self,
        plugin_id: str,
        context: BehaviorPluginContext,
    ) -> PluginResult:

        if not isinstance(context, BehaviorPluginContext):
            raise TypeError(
                "context must be BehaviorPluginContext"
            )

        with self._lock:

            entry = self._plugins.get(plugin_id)

            if entry is None:
                raise PluginNotFoundError(
                    f"Unknown plugin: {plugin_id}"
                )

            if not entry.enabled:
                raise PluginDisabledError(
                    f"Plugin is disabled: {plugin_id}"
                )

            plugin = entry.plugin

        try:

            result = plugin.execute(context)

            return self._validate_result(
                plugin_id,
                context,
                result,
            )

        except PluginRegistryError:
            raise

        except Exception as exc:
            raise PluginExecutionError(
                plugin_id,
                str(exc),
            ) from exc

    def execute_all(
        self,
        context: BehaviorPluginContext,
        *,
        plugin_ids: Sequence[str] | None = None,
    ) -> tuple[PluginResult, ...]:

        if not isinstance(context, BehaviorPluginContext):
            raise TypeError(
                "context must be BehaviorPluginContext"
            )

        with self._lock:

            if plugin_ids is None:
                selected = tuple(sorted(self._plugins))

            else:
                selected = tuple(plugin_ids)

                if len(selected) != len(set(selected)):
                    raise PluginContractError(
                        "Duplicate plugin IDs in execution request"
                    )

                missing = set(selected) - set(self._plugins)

                if missing:
                    raise PluginNotFoundError(
                        f"Unknown plugins: {sorted(missing)}"
                    )

            enabled_ids = tuple(
                plugin_id
                for plugin_id in selected
                if self._plugins[plugin_id].enabled
            )

        return tuple(
            self.execute(plugin_id, context)
            for plugin_id in enabled_ids
        )

    @property
    def plugin_count(self) -> int:

        with self._lock:
            return len(self._plugins)

    @property
    def enabled_count(self) -> int:

        with self._lock:
            return sum(
                entry.enabled
                for entry in self._plugins.values()
            )