from datetime import UTC, datetime

import pytest

from company_vision.behaviors.feature_extractor import (
    FeatureExtractor,
)
from company_vision.behaviors.plugin_registry import (
    BehaviorPluginContext,
    BehaviorPluginRegistry,
    DuplicatePluginError,
    PluginContractError,
    PluginDisabledError,
    PluginExecutionError,
    PluginNotFoundError,
    PluginResult,
)
from company_vision.behaviors.schemas import BehaviorFrame


def make_context(
    camera_id="camera_01",
    session_id="session_01",
    frame_id=1,
):

    frame = BehaviorFrame(
        camera_id=camera_id,
        session_id=session_id,
        frame_id=frame_id,
        timestamp_utc=datetime(
            2026, 11, 11, tzinfo=UTC
        ),
        width=1920,
        height=1080,
        persons=[],
        source="tracking_output",
        source_status="ok",
    )

    features = FeatureExtractor().extract(frame)

    return BehaviorPluginContext(frame=features)


class SamplePlugin:

    def __init__(self, plugin_id="sample", version="1.0"):
        self.plugin_id = plugin_id
        self.version = version
        self.calls = 0

    def execute(self, context):

        self.calls += 1

        frame = context.frame

        return PluginResult(
            plugin_id=self.plugin_id,
            camera_id=frame.camera_id,
            session_id=frame.session_id,
            frame_id=frame.frame_id,
            status="ok",
            payload={"calls": self.calls},
        )


class FailingPlugin(SamplePlugin):

    def execute(self, context):
        raise RuntimeError("intentional test failure")


class InvalidResultPlugin(SamplePlugin):

    def execute(self, context):
        return {"invalid": True}


class WrongIdentityPlugin(SamplePlugin):

    def execute(self, context):

        return PluginResult(
            plugin_id="wrong",
            camera_id=context.frame.camera_id,
            session_id=context.frame.session_id,
            frame_id=context.frame.frame_id,
        )


def test_register_plugin():

    registry = BehaviorPluginRegistry()

    descriptor = registry.register(SamplePlugin())

    assert descriptor.plugin_id == "sample"
    assert descriptor.version == "1.0"
    assert descriptor.enabled is True
    assert registry.plugin_count == 1


def test_register_disabled_plugin():

    registry = BehaviorPluginRegistry()

    descriptor = registry.register(
        SamplePlugin(),
        enabled=False,
    )

    assert descriptor.enabled is False
    assert registry.enabled_count == 0


def test_duplicate_registration_is_rejected():

    registry = BehaviorPluginRegistry()

    registry.register(SamplePlugin())

    with pytest.raises(DuplicatePluginError):
        registry.register(SamplePlugin())


def test_invalid_plugin_id_is_rejected():

    registry = BehaviorPluginRegistry()

    with pytest.raises(PluginContractError):
        registry.register(SamplePlugin("Invalid ID"))


def test_invalid_plugin_version_is_rejected():

    registry = BehaviorPluginRegistry()

    with pytest.raises(PluginContractError):
        registry.register(SamplePlugin(version=""))


def test_missing_execute_method_is_rejected():

    class InvalidPlugin:
        plugin_id = "invalid"
        version = "1.0"

    registry = BehaviorPluginRegistry()

    with pytest.raises(PluginContractError):
        registry.register(InvalidPlugin())


def test_get_plugin():

    registry = BehaviorPluginRegistry()
    plugin = SamplePlugin()

    registry.register(plugin)

    assert registry.get("sample") is plugin


def test_unknown_plugin_lookup():

    registry = BehaviorPluginRegistry()

    with pytest.raises(PluginNotFoundError):
        registry.get("missing")


def test_get_descriptor():

    registry = BehaviorPluginRegistry()

    registry.register(SamplePlugin())

    descriptor = registry.get_descriptor("sample")

    assert descriptor.plugin_id == "sample"
    assert descriptor.enabled is True


def test_list_plugins_is_deterministic():

    registry = BehaviorPluginRegistry()

    registry.register(SamplePlugin("zeta"))
    registry.register(SamplePlugin("alpha"))
    registry.register(SamplePlugin("beta"))

    ids = [
        item.plugin_id
        for item in registry.list_plugins()
    ]

    assert ids == ["alpha", "beta", "zeta"]


def test_enabled_only_listing():

    registry = BehaviorPluginRegistry()

    registry.register(SamplePlugin("alpha"))
    registry.register(
        SamplePlugin("beta"),
        enabled=False,
    )

    result = registry.list_plugins(enabled_only=True)

    assert [item.plugin_id for item in result] == ["alpha"]


def test_disable_plugin():

    registry = BehaviorPluginRegistry()

    registry.register(SamplePlugin())

    descriptor = registry.set_enabled("sample", False)

    assert descriptor.enabled is False
    assert registry.enabled_count == 0


def test_enable_plugin():

    registry = BehaviorPluginRegistry()

    registry.register(
        SamplePlugin(),
        enabled=False,
    )

    descriptor = registry.set_enabled("sample", True)

    assert descriptor.enabled is True
    assert registry.enabled_count == 1


def test_unknown_plugin_enable_is_rejected():

    registry = BehaviorPluginRegistry()

    with pytest.raises(PluginNotFoundError):
        registry.set_enabled("missing", True)


def test_disabled_plugin_cannot_execute_directly():

    registry = BehaviorPluginRegistry()

    registry.register(
        SamplePlugin(),
        enabled=False,
    )

    with pytest.raises(PluginDisabledError):
        registry.execute("sample", make_context())


def test_unregister_plugin():

    registry = BehaviorPluginRegistry()

    registry.register(SamplePlugin())

    assert registry.unregister("sample") is True
    assert registry.plugin_count == 0


def test_unregister_unknown_plugin_is_safe():

    registry = BehaviorPluginRegistry()

    assert registry.unregister("missing") is False


def test_execute_plugin():

    registry = BehaviorPluginRegistry()
    plugin = SamplePlugin()

    registry.register(plugin)

    result = registry.execute(
        "sample",
        make_context(),
    )

    assert result.plugin_id == "sample"
    assert result.status == "ok"
    assert result.payload["calls"] == 1
    assert plugin.calls == 1


def test_execute_requires_valid_context():

    registry = BehaviorPluginRegistry()

    registry.register(SamplePlugin())

    with pytest.raises(TypeError):
        registry.execute("sample", {})


def test_execute_invalid_return_type():

    registry = BehaviorPluginRegistry()

    registry.register(InvalidResultPlugin())

    with pytest.raises(PluginContractError):
        registry.execute("sample", make_context())


def test_execute_wrong_result_identity():

    registry = BehaviorPluginRegistry()

    registry.register(WrongIdentityPlugin())

    with pytest.raises(PluginContractError):
        registry.execute("sample", make_context())


def test_plugin_runtime_error_is_wrapped():

    registry = BehaviorPluginRegistry()

    registry.register(FailingPlugin())

    with pytest.raises(PluginExecutionError) as error:
        registry.execute("sample", make_context())

    assert error.value.plugin_id == "sample"


def test_execute_all_runs_enabled_plugins():

    registry = BehaviorPluginRegistry()

    first = SamplePlugin("alpha")
    second = SamplePlugin("beta")

    registry.register(first)
    registry.register(second)

    results = registry.execute_all(make_context())

    assert [item.plugin_id for item in results] == [
        "alpha",
        "beta",
    ]

    assert first.calls == 1
    assert second.calls == 1


def test_execute_all_skips_disabled_plugins():

    registry = BehaviorPluginRegistry()

    active = SamplePlugin("active")
    inactive = SamplePlugin("inactive")

    registry.register(active)
    registry.register(inactive, enabled=False)

    results = registry.execute_all(make_context())

    assert [item.plugin_id for item in results] == ["active"]
    assert active.calls == 1
    assert inactive.calls == 0


def test_execute_all_respects_requested_order():

    registry = BehaviorPluginRegistry()

    registry.register(SamplePlugin("alpha"))
    registry.register(SamplePlugin("beta"))

    results = registry.execute_all(
        make_context(),
        plugin_ids=("beta", "alpha"),
    )

    assert [item.plugin_id for item in results] == [
        "beta",
        "alpha",
    ]


def test_execute_all_rejects_duplicate_ids():

    registry = BehaviorPluginRegistry()

    registry.register(SamplePlugin())

    with pytest.raises(PluginContractError):
        registry.execute_all(
            make_context(),
            plugin_ids=("sample", "sample"),
        )


def test_execute_all_rejects_unknown_ids():

    registry = BehaviorPluginRegistry()

    with pytest.raises(PluginNotFoundError):
        registry.execute_all(
            make_context(),
            plugin_ids=("missing",),
        )


def test_register_many_is_atomic():

    registry = BehaviorPluginRegistry()

    registry.register(SamplePlugin("existing"))

    with pytest.raises(DuplicatePluginError):
        registry.register_many(
            (
                SamplePlugin("new"),
                SamplePlugin("existing"),
            )
        )

    assert registry.plugin_count == 1
    assert registry.get("existing") is not None

    with pytest.raises(PluginNotFoundError):
        registry.get("new")


def test_register_many_success():

    registry = BehaviorPluginRegistry()

    descriptors = registry.register_many(
        (
            SamplePlugin("alpha"),
            SamplePlugin("beta"),
        )
    )

    assert len(descriptors) == 2
    assert registry.plugin_count == 2


def test_plugin_result_unknown_status():

    result = PluginResult(
        plugin_id="sample",
        camera_id="camera_01",
        session_id="session_01",
        frame_id=1,
        status="unknown",
    )

    assert result.status == "unknown"


def test_plugin_result_not_applicable_status():

    result = PluginResult(
        plugin_id="sample",
        camera_id="camera_01",
        session_id="session_01",
        frame_id=1,
        status="not_applicable",
    )

    assert result.status == "not_applicable"


def test_plugin_result_payload_is_read_only():

    result = PluginResult(
        plugin_id="sample",
        camera_id="camera_01",
        session_id="session_01",
        frame_id=1,
        payload={"score": 0.8},
    )

    with pytest.raises(TypeError):
        result.payload["score"] = 0.1


def test_context_requires_frame_features():

    with pytest.raises(TypeError):
        BehaviorPluginContext(frame={})


def test_context_accepts_empty_histories():

    context = make_context()

    assert context.histories == ()


def test_context_normalizes_history_collection():

    context = make_context()

    assert isinstance(context.histories, tuple)


def test_initial_plugin_ids_are_unique():

    from company_vision.behaviors.plugin_registry import (
        INITIAL_PLUGIN_IDS,
    )

    assert len(INITIAL_PLUGIN_IDS) == 3
    assert len(set(INITIAL_PLUGIN_IDS)) == 3


def test_registry_counts():

    registry = BehaviorPluginRegistry()

    registry.register(SamplePlugin("alpha"))
    registry.register(
        SamplePlugin("beta"),
        enabled=False,
    )

    assert registry.plugin_count == 2
    assert registry.enabled_count == 1