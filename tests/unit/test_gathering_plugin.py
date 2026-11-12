"""Node 5.8.4 gathering detector contract tests."""

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from company_vision.behaviors.feature_extractor import (
    FrameFeatures,
    PersonFeatures,
)
from company_vision.behaviors.plugin_registry import (
    BehaviorPluginContext,
    BehaviorPluginRegistry,
)
from company_vision.behaviors.plugins.gathering import (
    GatheringConfig,
    GatheringPlugin,
)
from company_vision.behaviors.rule_engine import RuleDefinition, RuleEngine
from company_vision.behaviors.temporal_state_manager import (
    TemporalObservation,
    TemporalTrackHistory,
)


def person(track, x, frame_id=10, eligible=True, y=0.5, h=0.5):
    return PersonFeatures(
        camera_id="cam", session_id="session", frame_id=frame_id,
        track_id=track,
        behavior_eligible=eligible,
        observed=eligible,
        predicted=not eligible,
        bbox_center_x_norm=x,
        bbox_center_y_norm=y,
        bbox_width_norm=0.12,
        bbox_height_norm=h,
        keypoints=(),
        head_center=None,
        shoulder_center=None,
        shoulder_width_norm=None,
        shoulder_line=None,
        head_to_shoulder=None,
        left_arm=None,
        right_arm=None,
        total_keypoints=0,
        available_keypoints=0,
        unavailable_keypoints=0,
        mean_keypoint_quality=0.0,
    )


def context(
    people=None,
    history=True,
    source_status="ok",
    missing_track=None,
    broken_track=None,
    moved_track=None,
):
    if people is None:
        people = [
            person(1, 0.40),
            person(2, 0.48),
            person(3, 0.56),
        ]

    frame = FrameFeatures(
        schema_version="1.0",
        camera_id="cam", session_id="session",
        frame_id=10,
        timestamp_utc=datetime.now(UTC),
        monotonic_timestamp=10.0,
        width=1920, height=1080,
        persons=tuple(people),
        objects=(),
        object_detection_status="not_run",
        source="tracking_output",
        source_status=source_status,
    )

    histories = []

    if history:
        for p in people:
            if p.track_id == missing_track:
                continue

            times = (4.0, 5.0, 7.0, 9.0)

            if p.track_id == broken_track:
                times = (1.0, 2.0, 3.0, 9.0)

            entries = []

            for index, stamp in enumerate(times):
                old = replace(p, frame_id=index + 1)

                if p.track_id == moved_track and index == 2:
                    old = replace(old, bbox_center_x_norm=0.90)

                entries.append(TemporalObservation(
                    frame_id=index + 1,
                    timestamp_utc=datetime.now(UTC),
                    monotonic_timestamp=stamp,
                    person=old,
                ))

            histories.append(TemporalTrackHistory(
                camera_id="cam",
                session_id="session",
                track_id=p.track_id,
                observations=tuple(entries),
            ))

    return BehaviorPluginContext(
        frame=frame,
        histories=tuple(histories),
    )


def by_track(ctx):
    result = GatheringPlugin().execute(ctx)
    return {
        item["track_id"]: item
        for item in result.payload["observations"]
    }


def test_three_people_sustained_group():
    records = by_track(context())
    assert len(records) == 3
    assert all(r["metrics"]["score"] >= 0.8 for r in records.values())


def test_group_members_preserved():
    records = by_track(context())
    assert records[1]["evidence"]["group_members"] == [1, 2, 3]


def test_two_people_not_group():
    records = by_track(context(people=[
        person(1, 0.40), person(2, 0.48),
    ]))
    assert all(r["metrics"]["score"] == 0 for r in records.values())


def test_distant_people_not_group():
    records = by_track(context(people=[
        person(1, 0.1), person(2, 0.5), person(3, 0.9),
    ]))
    assert all(r["metrics"]["score"] == 0 for r in records.values())


def test_new_group_without_history_unknown():
    records = by_track(context(history=False))
    assert all(r["status"] == "unknown" for r in records.values())


def test_missing_one_member_history_unknown():
    records = by_track(context(missing_track=3))
    assert all(r["status"] == "unknown" for r in records.values())


def test_broken_history_unknown():
    records = by_track(context(broken_track=2))
    assert all(r["status"] == "unknown" for r in records.values())


def test_temporary_group_is_not_confirmed():
    records = by_track(context(moved_track=3))
    assert all(r["status"] == "unknown" for r in records.values())


def test_ineligible_track_unknown():
    people = [
        person(1, 0.40),
        person(2, 0.48),
        person(3, 0.56, eligible=False),
    ]
    records = by_track(context(people=people))
    assert records[3]["status"] == "unknown"
    assert records[1]["metrics"]["score"] == 0.0


def test_partial_source_unknown():
    records = by_track(context(source_status="partial"))
    assert all(r["status"] == "unknown" for r in records.values())


def test_missing_timestamp_unknown():
    ctx = context()
    modified = replace(ctx.frame, monotonic_timestamp=None)
    ctx = replace(ctx, frame=modified)
    assert all(
        r["status"] == "unknown"
        for r in by_track(ctx).values()
    )


def test_perspective_height_difference():
    people = [
        person(1, 0.40, h=0.25),
        person(2, 0.48, h=0.50),
        person(3, 0.56, h=0.50),
    ]
    records = by_track(context(people=people))
    assert all(r["metrics"]["score"] == 0 for r in records.values())


def test_no_people():
    result = GatheringPlugin().execute(
        context(people=[])
    )
    assert result.payload["observations"] == []


def test_registry_and_rule_engine():
    ctx = context()
    registry = BehaviorPluginRegistry()
    registry.register(GatheringPlugin())

    result = registry.execute("gathering", ctx)

    engine = RuleEngine([
        RuleDefinition(
            rule_id="gathering-observed",
            plugin_id="gathering",
            metric_key="score",
            operator="gte",
            threshold=0.8,
            min_confidence=0.5,
        )
    ])

    evaluations = engine.evaluate(ctx, [result])
    assert len(evaluations) == 3
    assert all(e.outcome == "triggered" for e in evaluations)


def test_invalid_people_config():
    with pytest.raises(ValueError):
        GatheringConfig(min_people=1)


def test_invalid_duration():
    with pytest.raises(ValueError):
        GatheringConfig(min_duration_seconds=-1)


def test_invalid_sample_count():
    with pytest.raises(ValueError):
        GatheringConfig(min_observations=2)


def test_repeatable_output():
    ctx = context()
    plugin = GatheringPlugin()
    assert plugin.execute(ctx).payload == plugin.execute(ctx).payload


def test_four_person_group():
    records = by_track(context(people=[
        person(1, 0.40),
        person(2, 0.46),
        person(3, 0.52),
        person(4, 0.58),
    ]))
    assert len(records) == 4
    assert all(r["status"] == "ok" for r in records.values())