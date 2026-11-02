
import numpy as np

from company_vision.core.schemas import (
    BoundingBox,
    FramePacket,
    FrameResult,
    Keypoint,
    ObjectObservation,
    PersonObservation,
    current_utc,
)
from company_vision.core.settings import load_settings
from company_vision.utils.logging_config import (
    configure_logging,
    get_logger,
)


def main():
    settings = load_settings()
    configure_logging(settings)

    logger = get_logger("company_vision.node_1_4")
    logger.info("Starting node 1.4 smoke test")

    image = np.zeros(
        (120, 160, 3),
        dtype=np.uint8,
    )

    packet = FramePacket(
        camera_id=settings.camera.camera_id,
        frame_id=1,
        timestamp=current_utc(),
        image=image,
    )

    person = PersonObservation(
        track_id=7,
        bbox=BoundingBox(
            x1=20,
            y1=10,
            x2=100,
            y2=110,
        ),
        confidence=0.95,
        keypoints=[
            Keypoint(
                name="left_wrist",
                x=70,
                y=65,
                confidence=0.89,
            ),
        ],
    )

    phone = ObjectObservation(
        class_id=67,
        class_name="cell phone",
        bbox=BoundingBox(
            x1=65,
            y1=60,
            x2=85,
            y2=90,
        ),
        confidence=0.91,
    )

    result = FrameResult.from_packet(
        packet=packet,
        persons=[person],
        objects=[phone],
    )

    serialized = result.model_dump_json()
    restored = FrameResult.model_validate_json(serialized)

    assert restored == result
    assert restored.width == 160
    assert restored.height == 120
    assert restored.persons[0].track_id == 7
    assert restored.objects[0].class_name == "cell phone"

    logger.info("FramePacket created successfully")
    logger.info("FrameResult validated successfully")
    logger.info("JSON round-trip successful")

    print("NODE 1.4 SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
