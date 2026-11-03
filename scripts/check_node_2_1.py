
from pathlib import Path

from company_vision.camera.video_source import (
    VideoFileSource,
)
from company_vision.core.settings import (
    load_settings,
)
from company_vision.utils.logging_config import (
    configure_logging,
    get_logger,
)

VIDEO_PATH = Path(
    "data/samples/node_2_1_sample.avi"
)


def main() -> None:
    settings = load_settings()
    configure_logging(settings)

    logger = get_logger(
        "company_vision.node_2_1"
    )

    logger.info("Starting video file smoke test")

    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(
            f"Sample video missing: {VIDEO_PATH}"
        )

    received_ids: list[int] = []

    with VideoFileSource(
        path=VIDEO_PATH,
        camera_id="office_test_cam",
    ) as source:
        logger.info(
            "Video: %s | FPS: %.2f | Size: %dx%d",
            VIDEO_PATH,
            source.fps,
            source.width,
            source.height,
        )

        while True:
            packet = source.read()

            if packet is None:
                break

            assert packet.camera_id == (
                "office_test_cam"
            )

            assert packet.width == 320
            assert packet.height == 240

            received_ids.append(
                packet.frame_id
            )

        assert source.is_eof

        logger.info(
            "Read %d frames",
            source.frames_read,
        )

    assert received_ids == list(range(24))

    logger.info(
        "Frame IDs, dimensions and EOF verified"
    )

    print("NODE 2.1 SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
