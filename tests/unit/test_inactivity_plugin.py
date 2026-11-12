"""Node 5.8.3 isolated inactivity detector tests."""

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from company_vision.behaviors.feature_extractor import (
    FrameFeatures,
    PersonFeatures,
    PointFeature,
)
from company_vision.behaviors.plugin_registry import (
    BehaviorPluginContext,
    BehaviorPluginRegistry,
)
from company_vision.behaviors.plugins.inactivity import (
    InactivityConfig,
    InactivityPlugin,
)
from company_vision.behaviors.rule_engine import RuleDefinition, RuleEngine
from company_vision.behaviors.temporal_state_manager import (
    TemporalObservation,
    TemporalTrackHistory,
)


def point(name, x=0.5, y=0.5, quality=0.95):
    return PointFeature(
        name=name,
        available=quality > 0,
        x_norm=x,
        y_norm=y,
        confidence=quality,
        weight=None,
        quality=quality,
        reason="available" if quality > 0 else "invalid",
    )


def person(
    track=1,
    frame_id=10,
    center=0.5,
    elbow_x=0.5,
    eligible=True,
    head_y=0.3,
    quality=0.95,
):
    return PersonFeatures(
        camera_id="cam",
        session_id="session",
        frame_id=frame_id,
        track_id=track,
        behavior_eligible=eligible,
        observed=eligible,
        predicted=not eligible,
        bbox_center_x_norm=center,
        bbox_center_y_norm=0.5,
        bbox_width_norm=0.3,
        bbox_height_norm=0.6,
        keypoints=(
            point("left_shoulder", 0.3, 0.5, quality),
            point("right_shoulder", 0.7, 0.5, quality),
            point("left_elbow", elbow_x, 0.7, quality),
        ),
        head_center=(0.5, head_y),
        shoulder_center=(0.5, 0.5),
        shoulder_width_norm=0.4,
        shoulder_line=None,
        head_to_shoulder=None,
        left_arm=None,
        right_arm=None,
        total_keypoints=3,
        available_keypoints=3 if quality > 0 else 0,
        unavailable_keypoints=0 if quality > 0 else 3,
        mean_keypoint_quality=quality,
    )


def context(
    current=None,
    earlier=None,
    times=None,
    source_status="ok",
):
    current = current or person()

    frame = FrameFeatures(
        schema_version="1.0",
        camera_id="cam",
        session_id="session",
        frame_id=10,
        timestamp_utc=datetime.now(UTC),
        monotonic_timestamp=10.0,
        width=1920,
        height=1080,
        persons=(current,),
        objects=(),
        object_detection_status="not_run",
        source="tracking_output",
        source_status=source_status,
    )

    if earlier is None:
        earlier = [
            person(frame_id=1),
            person(frame_id=4),
            person(frame_id=7),
        ]

    if times is None:
        times = (1.0, 4.0, 7.0)

    history = TemporalTrackHistory(
        camera_id="cam",
        session_id="session",
        track_id=current.track_id,
        observations=tuple(
            TemporalObservation(
                frame_id=sample.frame_id,
                timestamp_utc=datetime.now(UTC),
                monotonic_timestamp=stamp,
                person=sample,
            )
            for sample, stamp in zip(earlier, times)
        ),
    )

    return BehaviorPluginContext(
        frame=frame,
        histories=(history,),
    )


def result(ctx):
    return InactivityPlugin().execute(ctx).payload["observations"][0]


def test_sustained_stillness():
    item = result(context())
    assert item["status"] == "ok"
    assert item["metrics"]["score"] >= 0.8
    assert item["evidence"]["sleep_confirmed"] is False


def test_motion_is_negative():
    current = person(elbow_x=0.9)
    item = result(context(current=current))
    assert item["metrics"]["score"] == 0.0


def test_bbox_motion_is_negative():
    current = person(center=0.7)
    item = result(context(current=current))
    assert item["metrics"]["score"] == 0.0


def test_missing_history_is_unknown():
    ctx = context()
    ctx = replace(ctx, histories=())
    item = result(ctx)
    assert item["status"] == "unknown"


def test_short_duration_unknown():
    earlier = [
        person(frame_id=7),
        person(frame_id=8),
        person(frame_id=9),
    ]
    item = result(context(
        earlier=earlier,
        times=(7.0, 8.0, 9.0),
    ))
    assert item["status"] == "unknown"


def test_history_gap_unknown():
    item = result(context(times=(1.0, 2.0, 3.0)))
    assert item["status"] == "unknown"


def test_predicted_track_unknown():
    current = person(eligible=False)
    item = result(context(current=current))
    assert item["status"] == "unknown"


def test_partial_source_unknown():
    item = result(context(source_status="partial"))
    assert item["status"] == "unknown"


def test_low_quality_points_unknown():
    current = person(quality=0.1)
    earlier = [
        person(frame_id=1, quality=0.1),
        person(frame_id=4, quality=0.1),
        person(frame_id=7, quality=0.1),
    ]
    item = result(context(current=current, earlier=earlier))
    assert item["status"] == "unknown"


def test_head_lowered_is_not_sleep_confirmation():
    current = person(head_y=0.65)
    item = result(context(current=current))
    assert item["evidence"]["posture"]["head_lowered"] is True
    assert item["evidence"]["sleep_confirmed"] is False


def test_registry_and_rule_engine():
    ctx = context()
    registry = BehaviorPluginRegistry()
    registry.register(InactivityPlugin())

    plugin_result = registry.execute("inactivity", ctx)

    engine = RuleEngine([
        RuleDefinition(
            rule_id="stillness",
            plugin_id="inactivity",
            metric_key="score",
            operator="gte",
            threshold=0.8,
            min_confidence=0.5,
        )
    ])

    evaluations = engine.evaluate(ctx, [plugin_result])
    assert evaluations[0].outcome == "triggered"


def test_invalid_config():
    with pytest.raises(ValueError):
        InactivityConfig(min_duration_seconds=0)


def test_invalid_min_observations():
    with pytest.raises(ValueError):
        InactivityConfig(min_observations=2)


def test_deterministic():
    ctx = context()
    plugin = InactivityPlugin()
    assert plugin.execute(ctx).payload == plugin.execute(ctx).payload


def test_no_people():
    ctx = context()
    empty = replace(ctx.frame, persons=())
    ctx = replace(ctx, frame=empty, histories=())

    output = InactivityPlugin().execute(ctx)
    assert output.payload["observations"] == []


def test_separate_people():
    first = person(track=1)
    second = person(track=2, center=0.8)

    original = context(current=first)

    second_history = TemporalTrackHistory(
        camera_id="cam",
        session_id="session",
        track_id=2,
        observations=(
            TemporalObservation(
                frame_id=7,
                timestamp_utc=datetime.now(UTC),
                monotonic_timestamp=7.0,
                person=person(track=2, frame_id=7, center=0.8),
            ),
        ),
    )

    frame = replace(original.frame, persons=(first, second))

    ctx = BehaviorPluginContext(
        frame=frame,
        histories=original.histories + (second_history,),
    )

    records = InactivityPlugin().execute(ctx).payload["observations"]
    by_id = {item["track_id"]: item for item in records}

    assert by_id[1]["status"] == "ok"
    assert by_id[2]["status"] == "unknown"