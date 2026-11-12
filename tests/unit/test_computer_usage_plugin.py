"""Node 5.8.2 isolated contract tests."""
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from company_vision.behaviors.feature_extractor import (
    FrameFeatures,
    ObjectFeatures,
    PersonFeatures,
    PointFeature,
)
from company_vision.behaviors.plugin_registry import BehaviorPluginContext, BehaviorPluginRegistry
from company_vision.behaviors.plugins.computer_usage import ComputerUsageConfig, ComputerUsagePlugin
from company_vision.behaviors.rule_engine import RuleDefinition, RuleEngine
from company_vision.behaviors.temporal_state_manager import (
    TemporalObservation,
    TemporalTrackHistory,
)


def person(track=1, x=0.50, hand_x=0.75, eligible=True):
    return PersonFeatures(
        camera_id="cam", session_id="session", frame_id=10, track_id=track,
        behavior_eligible=eligible, observed=eligible, predicted=not eligible,
        bbox_center_x_norm=x, bbox_center_y_norm=0.50,
        bbox_width_norm=0.30, bbox_height_norm=0.60,
        keypoints=(PointFeature("right_wrist", True, hand_x, 0.55, 0.95, None, 0.95, "available"),),
        head_center=None, shoulder_center=None, shoulder_width_norm=None,
        shoulder_line=None, head_to_shoulder=None, left_arm=None, right_arm=None,
        total_keypoints=1, available_keypoints=1, unavailable_keypoints=0,
        mean_keypoint_quality=0.95,
    )


def device(x=0.58, class_id=66, confidence=0.98):
    return ObjectFeatures(
        object_id="obj", class_id=class_id,
        class_name={66: "keyboard", 64: "mouse", 63: "laptop", 62: "tv"}[class_id],
        confidence=confidence, center_x_norm=x, center_y_norm=0.53,
        width_norm=0.04, height_norm=0.02,
    )


def context(persons=None, objects=None, detection="available", history=True, source_status="ok"):
    persons = [person()] if persons is None else persons
    objects = [device()] if objects is None else objects
    frame = FrameFeatures(
        schema_version="1.0", camera_id="cam", session_id="session", frame_id=10,
        timestamp_utc=datetime.now(UTC), monotonic_timestamp=10.0,
        width=1920, height=1080, persons=tuple(persons), objects=tuple(objects),
        object_detection_status=detection, source="tracking_output", source_status=source_status,
    )
    histories = ()
    if history:
        items = []
        for p in persons:
            if p.track_id is None:
                continue
            old = replace(p, frame_id=9, keypoints=(replace(p.keypoints[0], x_norm=0.55),))
            items.append(TemporalTrackHistory(
                camera_id="cam", session_id="session", track_id=p.track_id,
                observations=(TemporalObservation(9, datetime.now(UTC), 9.0, old),),
            ))
        histories = tuple(items)
    return BehaviorPluginContext(frame=frame, histories=histories)


def single(ctx):
    return ComputerUsagePlugin().execute(ctx).payload["observations"][0]


def test_observed_computer_activity():
    result = single(context())
    assert result["status"] == "ok"
    assert result["metrics"]["score"] >= 0.7
    assert result["evidence"]["work_state"] == "computer_activity_observed"


def test_registry_and_rule_engine():
    ctx = context()
    registry = BehaviorPluginRegistry()
    registry.register(ComputerUsagePlugin())
    result = registry.execute("computer_usage", ctx)
    evaluated = RuleEngine([RuleDefinition(
        rule_id="computer-active", plugin_id="computer_usage",
        metric_key="score", operator="gte", threshold=0.7,
        min_confidence=0.5,
    )]).evaluate(ctx, [result])
    assert evaluated[0].outcome == "triggered"


def test_default_work_is_not_fabricated_positive():
    result = single(context(objects=[]))
    assert result["status"] == "unknown"
    assert result["metrics"] == {}
    assert result["evidence"]["work_state"] == "default_work_assumed"


def test_missing_detector_is_unknown():
    assert single(context(objects=[], detection="not_run"))["status"] == "unknown"


def test_partial_source_is_unknown():
    assert single(context(source_status="partial"))["status"] == "unknown"


def test_predicted_person_is_unknown():
    assert single(context(persons=[person(eligible=False)]))["status"] == "unknown"


def test_monitor_alone_is_insufficient():
    assert single(context(objects=[device(class_id=62)]))["status"] == "unknown"


def test_phone_or_unrelated_object_not_computer():
    assert single(context(objects=[]))["status"] == "unknown"


def test_no_history_no_confirmation():
    assert single(context(history=False))["status"] == "unknown"


def test_stationary_hand_no_confirmation():
    p = person(hand_x=0.55)
    assert single(context(persons=[p]))["status"] == "unknown"


def test_wrong_device_location_no_confirmation():
    assert single(context(objects=[device(x=0.95)]))["status"] == "unknown"


def test_low_confidence_device_no_confirmation():
    assert single(context(objects=[device(confidence=0.1)]))["status"] == "unknown"


def test_multiple_people_independent():
    result = ComputerUsagePlugin().execute(context(
        persons=[person(1, 0.5), person(2, 0.9)], objects=[device()],
    ))
    by_track = {item["track_id"]: item for item in result.payload["observations"]}
    assert by_track[1]["status"] == "ok"
    assert by_track[2]["status"] == "unknown"


def test_no_people():
    assert ComputerUsagePlugin().execute(context(persons=[])).payload["observations"] == []


def test_invalid_config():
    with pytest.raises(ValueError):
        ComputerUsageConfig(max_hand_distance=-1)


def test_same_input_deterministic():
    ctx = context()
    plugin = ComputerUsagePlugin()
    assert plugin.execute(ctx).payload == plugin.execute(ctx).payload