from datetime import UTC, datetime

import pytest

from company_vision.behaviors.feature_extractor import (
    FrameFeatures,
    ObjectFeatures,
    PersonFeatures,
    PointFeature,
)
from company_vision.behaviors.plugin_registry import (
    BehaviorPluginContext,
    BehaviorPluginRegistry,
)
from company_vision.behaviors.plugins.phone_usage import (
    PhoneUsageConfig,
    PhoneUsagePlugin,
)
from company_vision.behaviors.rule_engine import (
    RuleDefinition,
    RuleEngine,
)


def point(name, x, y):
    return PointFeature(
        name=name,
        available=True,
        x_norm=x,
        y_norm=y,
        confidence=0.95,
        weight=0.95,
        quality=0.95,
        reason="available",
    )


def person(track_id=1, center=0.5, eligible=True):
    return PersonFeatures(
        camera_id="camera-1",
        session_id="session-1",
        frame_id=1,
        track_id=track_id,
        behavior_eligible=eligible,
        observed=eligible,
        predicted=not eligible,
        bbox_center_x_norm=center,
        bbox_center_y_norm=0.5,
        bbox_width_norm=0.25,
        bbox_height_norm=0.6,
        keypoints=(
            point("left_elbow", 0.25, 0.55),
            point("right_elbow", 0.75, 0.55),
        ),
        head_center=None,
        shoulder_center=None,
        shoulder_width_norm=None,
        shoulder_line=None,
        head_to_shoulder=None,
        left_arm=None,
        right_arm=None,
        total_keypoints=9,
        available_keypoints=2,
        unavailable_keypoints=7,
        mean_keypoint_quality=0.95,
    )


def phone(x=0.4375, confidence=0.96):
    return ObjectFeatures(
        object_id=10,
        class_id=67,
        class_name="cell phone",
        confidence=confidence,
        center_x_norm=x,
        center_y_norm=0.53,
        width_norm=0.02,
        height_norm=0.04,
    )


def context(
    persons=None,
    objects=None,
    detection="available",
    source_status="ok",
):
    frame = FrameFeatures(
        schema_version="1.0",
        camera_id="camera-1",
        session_id="session-1",
        frame_id=1,
        timestamp_utc=datetime.now(UTC),
        monotonic_timestamp=1.0,
        width=1920,
        height=1080,
        persons=tuple(persons if persons is not None else [person()]),
        objects=tuple(objects if objects is not None else [phone()]),
        object_detection_status=detection,
        source="tracking_output",
        source_status=source_status,
    )
    return BehaviorPluginContext(frame=frame)


def observation(result, index=0):
    return result.payload["observations"][index]


def test_positive_phone_near_elbow():
    result = PhoneUsagePlugin().execute(context())
    item = observation(result)
    assert item["status"] == "ok"
    assert item["metrics"]["score"] >= 0.7


def test_rule_engine_compatibility():
    ctx = context()
    registry = BehaviorPluginRegistry()
    registry.register(PhoneUsagePlugin())
    result = registry.execute("phone_usage", ctx)
    rule = RuleDefinition(
        rule_id="phone-use",
        plugin_id="phone_usage",
        metric_key="score",
        operator="gte",
        threshold=0.7,
        min_confidence=0.5,
    )
    evaluated = RuleEngine([rule]).evaluate(ctx, [result])
    assert evaluated[0].outcome == "triggered"


def test_no_phone_is_negative():
    result = PhoneUsagePlugin().execute(context(objects=[]))
    item = observation(result)
    assert item["status"] == "ok"
    assert item["metrics"]["score"] == 0.0


def test_object_detector_not_run_is_unknown():
    result = PhoneUsagePlugin().execute(
        context(objects=[], detection="not_run")
    )
    assert observation(result)["status"] == "unknown"


def test_partial_source_is_unknown():
    result = PhoneUsagePlugin().execute(
        context(source_status="partial")
    )
    assert observation(result)["status"] == "unknown"


def test_predicted_track_is_unknown():
    result = PhoneUsagePlugin().execute(
        context(persons=[person(eligible=False)])
    )
    assert observation(result)["status"] == "unknown"


def test_phone_far_away_is_not_positive():
    result = PhoneUsagePlugin().execute(
        context(objects=[phone(x=0.95)])
    )
    assert observation(result)["metrics"]["score"] == 0.0


def test_multiple_people_independent():
    result = PhoneUsagePlugin().execute(
        context(
            persons=[person(1, 0.5), person(2, 0.85)],
            objects=[phone(0.4375)],
        )
    )
    by_id = {
        item["track_id"]: item
        for item in result.payload["observations"]
    }
    assert by_id[1]["metrics"]["score"] >= 0.7
    assert by_id[2]["metrics"]["score"] == 0.0


def test_no_person_returns_empty_observations():
    result = PhoneUsagePlugin().execute(
        context(persons=[])
    )
    assert result.payload["observations"] == []


def test_low_confidence_phone_does_not_trigger():
    result = PhoneUsagePlugin().execute(
        context(objects=[phone(confidence=0.1)])
    )
    assert observation(result)["metrics"]["score"] == 0.0


def test_invalid_configuration():
    with pytest.raises(ValueError):
        PhoneUsageConfig(min_phone_confidence=-1)


def test_registry_descriptor():
    registry = BehaviorPluginRegistry()
    descriptor = registry.register(PhoneUsagePlugin())
    assert descriptor.plugin_id == "phone_usage"


def test_deterministic_output():
    plugin = PhoneUsagePlugin()
    ctx = context()
    first = plugin.execute(ctx)
    second = plugin.execute(ctx)
    assert first.payload == second.payload


def test_no_duplicate_phone_assignment():
    plugin = PhoneUsagePlugin()
    result = plugin.execute(
        context(
            persons=[person(1, 0.5), person(2, 0.52)],
            objects=[phone(0.46)],
        )
    )
    positive = [
        item
        for item in result.payload["observations"]
        if item["metrics"].get("score", 0) >= 0.7
    ]
    assert len(positive) <= 1