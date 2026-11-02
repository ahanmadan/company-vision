
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from pydantic import ValidationError

from company_vision.core.schemas import (
    BoundingBox,
    FramePacket,
    FrameResult,
    Keypoint,
    ObjectObservation,
    PersonObservation,
    current_utc,
)
from company_vision.core.settings import (
    Settings,
    load_settings,
)
from company_vision.utils.logging_config import (
    configure_logging,
    get_logger,
)


@pytest.fixture
def settings():
    return load_settings("configs/app.yaml")


@pytest.fixture
def packet():
    return FramePacket(
        camera_id="cam_01",
        frame_id=1,
        timestamp=current_utc(),
        image=np.zeros((120, 160, 3), dtype=np.uint8),
    )


def make_box():
    return BoundingBox(x1=10, y1=20, x2=80, y2=100)


def test_01_settings_load(settings):
    assert settings.app.name == "Company Vision"
    assert settings.camera.target_fps == 15


def test_02_invalid_settings_rejected(settings):
    data = settings.model_dump()
    data["models"]["pose_confidence"] = 1.5

    with pytest.raises(ValidationError):
        Settings.model_validate(data)


def test_03_unknown_settings_key_rejected(settings):
    data = settings.model_dump()
    data["camera"]["unexpected_key"] = 10

    with pytest.raises(ValidationError):
        Settings.model_validate(data)


def test_04_valid_bbox():
    box = make_box()
    assert box.width == 70
    assert box.height == 80


def test_05_invalid_bbox_rejected():
    with pytest.raises(ValidationError):
        BoundingBox(x1=50, y1=20, x2=10, y2=100)


def test_06_keypoint_confidence_rejected():
    with pytest.raises(ValidationError):
        Keypoint(
            name="nose",
            x=10,
            y=20,
            confidence=1.8,
        )


def test_07_frame_packet_valid(packet):
    assert packet.width == 160
    assert packet.height == 120
    assert packet.image.dtype == np.uint8


def test_08_invalid_frame_image_rejected():
    with pytest.raises(ValueError):
        FramePacket(
            camera_id="cam_01",
            frame_id=1,
            timestamp=current_utc(),
            image=np.zeros((120, 160), dtype=np.uint8),
        )


def test_09_naive_timestamp_rejected():
    with pytest.raises(ValueError):
        FramePacket(
            camera_id="cam_01",
            frame_id=1,
            timestamp=datetime(2026, 1, 1),
            image=np.zeros((120, 160, 3), dtype=np.uint8),
        )


def test_10_non_utc_timestamp_rejected():
    offset = timezone(timedelta(hours=3, minutes=30))

    with pytest.raises(ValueError):
        FramePacket(
            camera_id="cam_01",
            frame_id=1,
            timestamp=datetime.now(offset),
            image=np.zeros((120, 160, 3), dtype=np.uint8),
        )


def test_11_duplicate_track_id_rejected(packet):
    person_a = PersonObservation(
        track_id=3,
        bbox=make_box(),
        confidence=0.9,
    )
    person_b = PersonObservation(
        track_id=3,
        bbox=make_box(),
        confidence=0.8,
    )

    with pytest.raises(ValidationError):
        FrameResult.from_packet(
            packet,
            persons=[person_a, person_b],
        )


def test_12_json_roundtrip_and_bounds(packet):
    person = PersonObservation(
        track_id=5,
        bbox=make_box(),
        confidence=0.9,
        keypoints=[
            Keypoint(
                name="left_wrist",
                x=40,
                y=60,
                confidence=0.85,
            )
        ],
    )

    obj = ObjectObservation(
        class_id=67,
        class_name="cell phone",
        bbox=BoundingBox(x1=40, y1=50, x2=60, y2=80),
        confidence=0.91,
    )

    result = FrameResult.from_packet(
        packet,
        persons=[person],
        objects=[obj],
    )

    restored = FrameResult.model_validate_json(
        result.model_dump_json()
    )

    assert restored == result
    assert len(restored.persons) == 1
    assert len(restored.objects) == 1

    with pytest.raises(ValidationError):
        FrameResult.from_packet(
            packet,
            persons=[
                PersonObservation(
                    track_id=6,
                    bbox=BoundingBox(
                        x1=20, y1=20, x2=200, y2=110
                    ),
                    confidence=0.9,
                )
            ],
        )


def test_13_logging(settings, capsys):
    configure_logging(settings)
    logger = get_logger("company_vision.test")
    logger.info("TEST LOG MESSAGE")

    output = capsys.readouterr().out
    assert "TEST LOG MESSAGE" in output
