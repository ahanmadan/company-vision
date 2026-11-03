

import numpy as np
import pytest

from company_vision.core.schemas import (
    BoundingBox,
    FramePacket,
    FrameResult,
    Keypoint,
    ObjectObservation,
    PersonObservation,
    current_utc,
)
from company_vision.rendering.overlay_renderer import (
    OverlayConfig,
    OverlayRenderer,
    OverlayRendererError,
)


def make_packet():
    return FramePacket(
        camera_id="cam_01",
        frame_id=1,
        timestamp=current_utc(),
        image=np.zeros((240, 320, 3), dtype=np.uint8),
    )


def make_person(track_id=7):
    names = [
        "nose",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
    ]

    points = [
        Keypoint(
            name=name,
            x=100.0 + index,
            y=50.0 + index,
            confidence=0.9,
        )
        for index, name in enumerate(names)
    ]

    return PersonObservation(
        track_id=track_id,
        bbox=BoundingBox(
            x1=50,
            y1=20,
            x2=180,
            y2=220,
        ),
        confidence=0.95,
        keypoints=points,
    )


def make_object():
    return ObjectObservation(
        class_id=67,
        class_name="cell phone",
        bbox=BoundingBox(
            x1=200,
            y1=80,
            x2=250,
            y2=150,
        ),
        confidence=0.88,
    )


def make_result(packet, persons=None, objects=None):
    return FrameResult.from_packet(
        packet,
        persons=persons or [],
        objects=objects or [],
        status="ok",
    )


def test_01_render_returns_numpy_image():
    packet = make_packet()
    result = make_result(packet, [make_person()], [make_object()])

    output = OverlayRenderer().render(packet, result)

    assert isinstance(output, np.ndarray)
    assert output.shape == packet.image.shape
    assert output.dtype == np.uint8


def test_02_original_frame_is_not_modified():
    packet = make_packet()
    original = packet.image.copy()

    result = make_result(packet, [make_person()])

    OverlayRenderer().render(packet, result)

    assert np.array_equal(packet.image, original)


def test_03_output_is_independent_copy():
    packet = make_packet()
    result = make_result(packet, [make_person()])

    output = OverlayRenderer().render(packet, result)

    assert output is not packet.image
    assert not np.shares_memory(output, packet.image)


def test_04_person_box_is_drawn():
    packet = make_packet()
    result = make_result(packet, [make_person()])

    output = OverlayRenderer().render(packet, result)

    assert np.any(output != 0)


def test_05_object_box_is_drawn():
    packet = make_packet()
    result = make_result(packet, objects=[make_object()])

    output = OverlayRenderer().render(packet, result)

    assert np.any(output != 0)


def test_06_empty_frame_is_valid():
    packet = make_packet()
    result = make_result(packet)

    output = OverlayRenderer().render(packet, result)

    assert np.array_equal(output, packet.image)


def test_07_track_id_can_be_missing():
    packet = make_packet()
    result = make_result(packet, [make_person(None)])

    output = OverlayRenderer().render(packet, result)

    assert output.shape == packet.image.shape


def test_08_low_confidence_keypoints_are_hidden():
    packet = make_packet()

    person = make_person()

    person.keypoints = [
        Keypoint(
            name="nose",
            x=100,
            y=100,
            confidence=0.1,
        )
    ]

    result = make_result(packet, [person])

    renderer = OverlayRenderer(
        OverlayConfig(
            show_person_boxes=False,
            show_track_ids=False,
            confidence_threshold=0.25,
        )
    )

    output = renderer.render(packet, result)

    assert np.array_equal(output, packet.image)


def test_09_camera_mismatch_rejected():
    packet = make_packet()

    result = make_result(packet)

    result.camera_id = "another_camera"

    with pytest.raises(OverlayRendererError):
        OverlayRenderer().render(packet, result)


def test_10_frame_id_mismatch_rejected():
    packet = make_packet()

    result = make_result(packet)

    result.frame_id = 99

    with pytest.raises(OverlayRendererError):
        OverlayRenderer().render(packet, result)


def test_11_invalid_image_rejected():
    packet = make_packet()
    result = make_result(packet)

    invalid_image = np.zeros(
        (240, 320),
        dtype=np.uint8,
    )

    object.__setattr__(
        packet,
        "image",
        invalid_image,
    )

    with pytest.raises(OverlayRendererError):
        OverlayRenderer().render(packet, result)


def test_12_invalid_configuration_rejected():
    with pytest.raises(ValueError):
        OverlayRenderer(
            OverlayConfig(confidence_threshold=2.0)
        )
