from datetime import UTC, datetime

import pytest

from company_vision.behaviors.feature_extractor import (
    FrameFeatures,
    PersonFeatures,
)
from company_vision.behaviors.plugin_registry import (
    BehaviorPluginContext,
    PluginResult,
)
from company_vision.behaviors.rule_engine import (
    RuleConfigurationError,
    RuleContractError,
    RuleDefinition,
    RuleEngine,
)


def make_person(
    *,
    track_id=7,
    eligible=True,
    observed=True,
    predicted=False,
):
    return PersonFeatures(
        camera_id="camera-1",
        session_id="session-1",
        frame_id=10,
        track_id=track_id,
        behavior_eligible=eligible,
        observed=observed,
        predicted=predicted,
        bbox_center_x_norm=0.5,
        bbox_center_y_norm=0.5,
        bbox_width_norm=0.2,
        bbox_height_norm=0.4,
        keypoints=(),
        head_center=None,
        shoulder_center=None,
        shoulder_width_norm=None,
        shoulder_line=None,
        head_to_shoulder=None,
        left_arm=None,
        right_arm=None,
        total_keypoints=9,
        available_keypoints=0,
        unavailable_keypoints=9,
        mean_keypoint_quality=0.0,
    )


def make_context(persons=None):
    frame = FrameFeatures(
        schema_version="1.0",
        camera_id="camera-1",
        session_id="session-1",
        frame_id=10,
        timestamp_utc=datetime.now(UTC),
        monotonic_timestamp=10.0,
        width=1920,
        height=1080,
        persons=tuple(persons or ()),
        objects=(),
        object_detection_status="unknown",
        source="tracking_output",
        source_status="ok",
    )
    return BehaviorPluginContext(frame=frame, histories=())


def make_rule(**kwargs):
    values = {
        "rule_id": "phone-score",
        "plugin_id": "phone_usage",
        "metric_key": "score",
        "operator": "gte",
        "threshold": 0.7,
    }
    values.update(kwargs)
    return RuleDefinition(**values)


def make_result(
    *,
    observations=None,
    status="ok",
    plugin_id="phone_usage",
    frame_id=10,
):
    payload = {}

    if observations is not None:
        payload["observations"] = observations

    return PluginResult(
        plugin_id=plugin_id,
        camera_id="camera-1",
        session_id="session-1",
        frame_id=frame_id,
        status=status,
        payload=payload,
    )


def make_observation(
    *,
    track_id=7,
    score=0.85,
    confidence=0.95,
    status="ok",
):
    return {
        "track_id": track_id,
        "status": status,
        "metrics": {"score": score},
        "confidence": confidence,
        "evidence": {"source": "test"},
    }


def test_rule_triggered():
    context = make_context([make_person()])
    engine = RuleEngine([make_rule()])

    result = engine.evaluate(
        context,
        [make_result(observations=[make_observation()])],
    )

    assert len(result) == 1
    assert result[0].outcome == "triggered"
    assert result[0].track_id == 7
    assert result[0].observed_value == 0.85


def test_rule_not_triggered():
    context = make_context([make_person()])
    engine = RuleEngine([make_rule()])

    result = engine.evaluate(
        context,
        [make_result(observations=[make_observation(score=0.2)])],
    )

    assert result[0].outcome == "not_triggered"


def test_missing_plugin_result_is_unknown():
    engine = RuleEngine([make_rule()])

    result = engine.evaluate(make_context(), [])

    assert result[0].outcome == "unknown"


def test_unknown_plugin_status_is_unknown():
    engine = RuleEngine([make_rule()])

    result = engine.evaluate(
        make_context(),
        [make_result(status="unknown")],
    )

    assert result[0].outcome == "unknown"


def test_not_applicable_plugin_status():
    engine = RuleEngine([make_rule()])

    result = engine.evaluate(
        make_context(),
        [make_result(status="not_applicable")],
    )

    assert result[0].outcome == "not_applicable"


def test_empty_observations_are_unknown():
    engine = RuleEngine([make_rule()])

    result = engine.evaluate(
        make_context(),
        [make_result(observations=[])],
    )

    assert result[0].outcome == "unknown"


def test_missing_metric_is_unknown():
    context = make_context([make_person()])
    engine = RuleEngine([make_rule()])

    observation = {
        "track_id": 7,
        "status": "ok",
        "metrics": {},
    }

    result = engine.evaluate(
        context,
        [make_result(observations=[observation])],
    )

    assert result[0].outcome == "unknown"


def test_unregistered_track_is_unknown():
    context = make_context([make_person(track_id=8)])
    engine = RuleEngine([make_rule()])

    result = engine.evaluate(
        context,
        [make_result(observations=[make_observation(track_id=7)])],
    )

    assert result[0].outcome == "unknown"


def test_ineligible_person_is_unknown():
    context = make_context([make_person(eligible=False)])
    engine = RuleEngine([make_rule()])

    result = engine.evaluate(
        context,
        [make_result(observations=[make_observation()])],
    )

    assert result[0].outcome == "unknown"


def test_predicted_person_is_unknown():
    context = make_context([
        make_person(eligible=False, observed=False, predicted=True)
    ])
    engine = RuleEngine([make_rule()])

    result = engine.evaluate(
        context,
        [make_result(observations=[make_observation()])],
    )

    assert result[0].outcome == "unknown"


def test_low_confidence_is_unknown():
    context = make_context([make_person()])
    engine = RuleEngine([
        make_rule(min_confidence=0.8)
    ])

    result = engine.evaluate(
        context,
        [make_result(observations=[
            make_observation(confidence=0.4)
        ])],
    )

    assert result[0].outcome == "unknown"


def test_missing_confidence_with_requirement_is_unknown():
    context = make_context([make_person()])
    engine = RuleEngine([
        make_rule(min_confidence=0.8)
    ])

    observation = make_observation()
    observation.pop("confidence")

    result = engine.evaluate(
        context,
        [make_result(observations=[observation])],
    )

    assert result[0].outcome == "unknown"


def test_observation_unknown_is_not_negative():
    context = make_context([make_person()])
    engine = RuleEngine([make_rule()])

    result = engine.evaluate(
        context,
        [make_result(observations=[
            make_observation(status="unknown")
        ])],
    )

    assert result[0].outcome == "unknown"


def test_disabled_rule_is_not_applicable():
    engine = RuleEngine([
        make_rule(enabled=False)
    ])

    result = engine.evaluate(make_context(), [])

    assert result[0].outcome == "not_applicable"


def test_duplicate_rule_ids_rejected():
    with pytest.raises(RuleConfigurationError):
        RuleEngine([
            make_rule(),
            make_rule(plugin_id="inactivity"),
        ])


def test_invalid_threshold_rejected():
    with pytest.raises(RuleConfigurationError):
        make_rule(threshold=float("nan"))


def test_duplicate_plugin_results_rejected():
    engine = RuleEngine([make_rule()])
    result = make_result(observations=[])

    with pytest.raises(RuleContractError):
        engine.evaluate(
            make_context(),
            [result, result],
        )


def test_frame_identity_mismatch_rejected():
    engine = RuleEngine([make_rule()])

    with pytest.raises(RuleContractError):
        engine.evaluate(
            make_context(),
            [make_result(frame_id=11)],
        )


def test_invalid_metric_type_rejected():
    context = make_context([make_person()])
    engine = RuleEngine([make_rule()])

    observation = make_observation()
    observation["metrics"]["score"] = True

    with pytest.raises(RuleContractError):
        engine.evaluate(
            context,
            [make_result(observations=[observation])],
        )


def test_invalid_observation_confidence_rejected():
    context = make_context([make_person()])
    engine = RuleEngine([make_rule()])

    observation = make_observation(confidence=2.0)

    with pytest.raises(RuleContractError):
        engine.evaluate(
            context,
            [make_result(observations=[observation])],
        )


def test_all_comparison_operators():
    comparisons = [
        ("gt", 0.8, False),
        ("gte", 0.8, True),
        ("lt", 0.8, False),
        ("lte", 0.8, True),
        ("eq", 0.8, True),
    ]

    for operator, threshold, expected in comparisons:
        engine = RuleEngine([
            make_rule(operator=operator, threshold=0.8)
        ])

        result = engine.evaluate(
            make_context([make_person()]),
            [make_result(observations=[
                make_observation(score=0.8)
            ])],
        )

        assert (result[0].outcome == "triggered") is expected


def test_multiple_observations_are_independent():
    engine = RuleEngine([make_rule()])
    context = make_context([
        make_person(track_id=7),
        make_person(track_id=8),
    ])

    result = engine.evaluate(
        context,
        [make_result(observations=[
            make_observation(track_id=7, score=0.9),
            make_observation(track_id=8, score=0.2),
        ])],
    )

    assert len(result) == 2
    assert result[0].outcome == "triggered"
    assert result[1].outcome == "not_triggered"


def test_group_rule_can_skip_person_eligibility():
    engine = RuleEngine([
        make_rule(requires_eligible_track=False)
    ])

    result = engine.evaluate(
        make_context(),
        [make_result(observations=[
            make_observation(track_id=None)
        ])],
    )

    assert result[0].outcome == "triggered"


def test_rule_order_is_deterministic():
    rules = [
        make_rule(rule_id="z-rule"),
        make_rule(rule_id="a-rule"),
    ]

    engine = RuleEngine(rules)

    assert [rule.rule_id for rule in engine.rules] == [
        "a-rule",
        "z-rule",
    ]


def test_result_evidence_is_preserved():
    engine = RuleEngine([make_rule()])

    result = engine.evaluate(
        make_context([make_person()]),
        [make_result(observations=[make_observation()])],
    )

    assert result[0].evidence["source"] == "test"