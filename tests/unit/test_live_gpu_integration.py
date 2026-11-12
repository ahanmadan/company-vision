"""Node 5.10.2 integration validation."""

import numpy as np

from company_vision.behaviors.live_gpu_integration import (
    LiveBehaviorIntegrator,
    draw_behavior_labels,
    overlay_lines,
)


def frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def track(track_id=1):
    points = np.array(
        [[65, 30 + i * 10, 0.90] for i in range(9)],
        dtype=np.float32,
    )
    return {
        "track_id": track_id,
        "bbox": np.array([40, 10, 100, 140]),
        "keypoints": points,
        "raw_keypoints": points.copy(),
        "keypoint_weights": np.ones(9) * 0.8,
        "confidence": 0.95,
        "observed": True,
        "predicted": False,
        "behavior_eligible": True,
        "missed": 0,
    }


def process(persons=None, session="test-session"):
    integration = LiveBehaviorIntegrator(session_id=session)

    output = integration.process(
        frame_id=1,
        monotonic_timestamp=1.0,
        frame=frame(),
        tracked_persons=[] if persons is None else persons,
    )

    return integration, output


def test_empty_frame():
    _, output = process()
    assert output.results == ()
    assert output.overlays == ()


def test_tracked_person():
    _, output = process([track()])
    assert output.results[0].track_id == 1
    assert output.results[0].status == "unknown"


def test_four_behavior_plugins():
    _, output = process([track()])
    assert len(output.results[0].rule_evaluations) == 4
    assert len(output.results[0].evidence) == 4


def test_multiple_persons():
    _, output = process([track(1), track(2), track(3)])
    assert len(output.results) == 3


def test_overlay_labels():
    _, output = process([track()])
    labels = overlay_lines(output)
    assert 1 in labels
    assert "ID 1" in labels[1][0]


def test_renderer_dimensions():
    image = frame()
    _, output = process()
    rendered = draw_behavior_labels(image, [], output)
    assert rendered.shape == image.shape


def test_clear():
    integration, _ = process()
    assert integration.frames_processed == 1
    integration.clear()
    assert integration.frames_processed == 0
    assert integration.results_produced == 0