"""Node 5.9.3: four-plugin execution integration tests."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from company_vision.behaviors.behavior_execution_pipeline import (
    PLUGIN_ORDER,
    BehaviorExecutionBatch,
    BehaviorExecutionError,
    BehaviorExecutionPipeline,
)
from company_vision.behaviors.plugin_registry import (
    PluginContractError,
    PluginResult,
)
from company_vision.behaviors.plugins.computer_usage import ComputerUsagePlugin
from company_vision.behaviors.plugins.gathering import GatheringPlugin
from company_vision.behaviors.plugins.inactivity import InactivityPlugin
from company_vision.behaviors.plugins.phone_usage import PhoneUsagePlugin

BASE = datetime(2026, 11, 12, tzinfo=UTC)


def tracked(track_id=1, x=40.0, observed=True, eligible=True):
    points = np.array(
        [[x + 18, 20 + index * 5, 0.9] for index in range(9)],
        dtype=np.float32,
    )

    return {
        "track_id": track_id,
        "bbox": np.array(
            [x, 10, x + 50, 130],
            dtype=np.float32,
        ),
        "keypoints": points,
        "raw_keypoints": points.copy() if observed else None,
        "keypoint_weights": np.ones(9, dtype=np.float32) * 0.85,
        "confidence": 0.95,
        "observed": observed,
        "predicted": not observed,
        "behavior_eligible": eligible if observed else False,
        "missed": 0 if observed else 1,
    }


def frame_args(
    frame_id=1,
    *,
    people=None,
    objects=None,
    camera="cam-a",
    session="session-a",
    seconds=None,
    source_status="ok",
):
    if people is None:
        people = [tracked()]

    if seconds is None:
        seconds = float(frame_id)

    return {
        "camera_id": camera,
        "session_id": session,
        "frame_id": frame_id,
        "timestamp_utc": BASE + timedelta(seconds=seconds),
        "monotonic_timestamp": seconds,
        "width": 640,
        "height": 480,
        "tracked_persons": people,
        "objects": objects,
        "source_status": source_status,
    }


def test_four_plugins_are_registered():
    pipeline = BehaviorExecutionPipeline()

    assert pipeline.registry.plugin_count == 4
    assert pipeline.registry.enabled_count == 4


def test_execution_order_is_stable():
    pipeline = BehaviorExecutionPipeline()
    batch = pipeline.process_tracking(**frame_args())

    assert tuple(r.plugin_id for r in batch.results) == PLUGIN_ORDER


def test_standard_batch_metadata():
    pipeline = BehaviorExecutionPipeline()
    batch = pipeline.process_tracking(**frame_args())

    assert isinstance(batch, BehaviorExecutionBatch)
    assert batch.camera_id == "cam-a"
    assert batch.session_id == "session-a"
    assert batch.frame_id == 1
    assert batch.plugin_count == 4


def test_per_track_has_every_plugin():
    pipeline = BehaviorExecutionPipeline()
    batch = pipeline.process_tracking(**frame_args())

    assert batch.track_count == 1
    assert tuple(batch.per_track[1]) == PLUGIN_ORDER


def test_missing_object_detector_is_not_positive():
    batch = BehaviorExecutionPipeline().process_tracking(
        **frame_args()
    )

    assert batch.get_observation(1, "phone_usage")["status"] == "unknown"
    assert batch.get_observation(1, "computer_usage")["status"] == "unknown"


def test_computer_default_is_not_fabricated_score():
    batch = BehaviorExecutionPipeline().process_tracking(
        **frame_args()
    )

    item = batch.get_observation(1, "computer_usage")

    assert item["metrics"] == {}
    assert item["evidence"]["work_state"] == "default_work_assumed"


def test_inactivity_unknown_with_single_frame():
    batch = BehaviorExecutionPipeline().process_tracking(
        **frame_args()
    )

    assert batch.get_observation(1, "inactivity")["status"] == "unknown"


def test_gathering_negative_with_single_person():
    batch = BehaviorExecutionPipeline().process_tracking(
        **frame_args()
    )

    item = batch.get_observation(1, "gathering")

    assert item["status"] == "ok"
    assert item["metrics"]["score"] == 0.0


def test_multiple_people_have_independent_results():
    batch = BehaviorExecutionPipeline().process_tracking(
        **frame_args(people=[
            tracked(1, x=30),
            tracked(2, x=150),
            tracked(3, x=270),
        ])
    )

    assert batch.track_count == 3
    assert set(batch.per_track) == {1, 2, 3}
    assert all(len(items) == 4 for items in batch.per_track.values())


def test_empty_frame_keeps_four_plugin_results():
    batch = BehaviorExecutionPipeline().process_tracking(
        **frame_args(people=[])
    )

    assert batch.track_count == 0
    assert batch.plugin_count == 4
    assert all(
        result.payload["observations"] == []
        for result in batch.results
    )


def test_predicted_track_has_no_positive_claims():
    batch = BehaviorExecutionPipeline().process_tracking(
        **frame_args(people=[
            tracked(observed=False, eligible=False)
        ])
    )

    assert all(
        item["status"] == "unknown"
        for item in batch.per_track[1].values()
    )


def test_unconfirmed_track_has_no_positive_claims():
    batch = BehaviorExecutionPipeline().process_tracking(
        **frame_args(people=[
            tracked(eligible=False)
        ])
    )

    assert all(
        item["status"] == "unknown"
        for item in batch.per_track[1].values()
    )


def test_partial_source_keeps_unverified_observations_unknown():
    batch = BehaviorExecutionPipeline().process_tracking(
        **frame_args(source_status="partial")
    )

    assert batch.get_observation(1, "phone_usage")["status"] == "unknown"
    assert batch.get_observation(1, "computer_usage")["status"] == "unknown"
    assert batch.get_observation(1, "gathering")["status"] == "unknown"


def test_real_phone_detector_output_preserved():
    phone = {
        "object_id": "phone-1",
        "class_id": 67,
        "class_name": "cell phone",
        "confidence": 0.96,
        "bbox": [56, 65, 72, 90],
    }

    batch = BehaviorExecutionPipeline().process_tracking(
        **frame_args(objects=[phone])
    )

    item = batch.get_observation(1, "phone_usage")
    assert item["status"] in ("ok", "unknown")
    assert item["evidence"]["reason"] != "object_detection_unavailable"


def test_same_context_deterministic():
    pipeline = BehaviorExecutionPipeline()
    context = pipeline.temporal.process_tracking(**frame_args())

    first = pipeline.execute_context(context)
    second = pipeline.execute_context(context)

    assert first.per_track == second.per_track


def test_multiple_frames_accumulate_one_history():
    pipeline = BehaviorExecutionPipeline()

    pipeline.process_tracking(**frame_args(1))
    batch = pipeline.process_tracking(**frame_args(2))

    assert batch.context.histories[0].observation_count == 2


def test_session_isolation():
    pipeline = BehaviorExecutionPipeline()

    pipeline.process_tracking(**frame_args(
        1, camera="cam-a", session="old"
    ))

    batch = pipeline.process_tracking(**frame_args(
        1, camera="cam-a", session="new"
    ))

    assert batch.context.histories[0].observation_count == 1


def test_camera_isolation():
    pipeline = BehaviorExecutionPipeline()

    pipeline.process_tracking(**frame_args(camera="cam-a"))
    pipeline.process_tracking(**frame_args(camera="cam-b"))

    assert pipeline.temporal.active_track_count == 2


def test_invalid_plugin_order_rejected():
    with pytest.raises(ValueError):
        BehaviorExecutionPipeline(plugins=(
            ComputerUsagePlugin(),
            PhoneUsagePlugin(),
            InactivityPlugin(),
            GatheringPlugin(),
        ))


def test_missing_plugin_rejected():
    with pytest.raises(ValueError):
        BehaviorExecutionPipeline(plugins=(
            PhoneUsagePlugin(),
            InactivityPlugin(),
            GatheringPlugin(),
        ))


def test_disabled_plugin_cannot_disappear_silently():
    pipeline = BehaviorExecutionPipeline()
    pipeline.registry.set_enabled("phone_usage", False)

    context = pipeline.temporal.process_tracking(**frame_args())

    with pytest.raises(PluginContractError):
        pipeline.execute_context(context)


def test_duplicate_observation_is_rejected():
    pipeline = BehaviorExecutionPipeline()
    context = pipeline.temporal.process_tracking(**frame_args())
    results = list(pipeline.registry.execute_all(
        context, plugin_ids=PLUGIN_ORDER
    ))

    first = results[0]
    observation = first.payload["observations"][0]

    results[0] = PluginResult(
        plugin_id=first.plugin_id,
        camera_id=first.camera_id,
        session_id=first.session_id,
        frame_id=first.frame_id,
        payload={"observations": [observation, observation]},
    )

    with pytest.raises(BehaviorExecutionError):
        pipeline._collect(context, tuple(results))


def test_missing_track_observation_is_rejected():
    pipeline = BehaviorExecutionPipeline()
    context = pipeline.temporal.process_tracking(**frame_args())
    results = list(pipeline.registry.execute_all(
        context, plugin_ids=PLUGIN_ORDER
    ))

    first = results[0]

    results[0] = PluginResult(
        plugin_id=first.plugin_id,
        camera_id=first.camera_id,
        session_id=first.session_id,
        frame_id=first.frame_id,
        payload={"observations": []},
    )

    with pytest.raises(BehaviorExecutionError):
        pipeline._collect(context, tuple(results))


def test_unknown_must_not_have_score():
    pipeline = BehaviorExecutionPipeline()
    context = pipeline.temporal.process_tracking(**frame_args())
    results = list(pipeline.registry.execute_all(
        context, plugin_ids=PLUGIN_ORDER
    ))

    first = results[0]

    results[0] = PluginResult(
        plugin_id=first.plugin_id,
        camera_id=first.camera_id,
        session_id=first.session_id,
        frame_id=first.frame_id,
        payload={"observations": [{
            "track_id": 1,
            "status": "unknown",
            "metrics": {"score": 0.9},
            "evidence": {},
        }]},
    )

    with pytest.raises(BehaviorExecutionError):
        pipeline._collect(context, tuple(results))


def test_reset_allows_reuse_of_frame_ids():
    pipeline = BehaviorExecutionPipeline()

    pipeline.process_tracking(**frame_args(5))
    pipeline.clear()

    batch = pipeline.process_tracking(**frame_args(1))

    assert batch.context.histories[0].observation_count == 1


def test_no_rules_or_events_created():
    batch = BehaviorExecutionPipeline().process_tracking(
        **frame_args()
    )

    assert not hasattr(batch, "events")
    assert not hasattr(batch, "rule_evaluations")


def test_context_type_rejected():
    pipeline = BehaviorExecutionPipeline()

    with pytest.raises(TypeError):
        pipeline.execute_context({})


def test_results_are_plugin_results():
    batch = BehaviorExecutionPipeline().process_tracking(
        **frame_args()
    )

    assert all(
        isinstance(item, PluginResult)
        for item in batch.results
    )


def test_current_context_not_reprocessed():
    pipeline = BehaviorExecutionPipeline()
    pipeline.process_tracking(**frame_args())

    with pytest.raises(ValueError):
        pipeline.process_tracking(**frame_args())


def test_result_for_unknown_track_raises():
    batch = BehaviorExecutionPipeline().process_tracking(
        **frame_args()
    )

    with pytest.raises(KeyError):
        batch.get_observation(999, "phone_usage")