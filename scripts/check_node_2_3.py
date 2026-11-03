
import argparse
import os
from datetime import timedelta

import cv2

from company_vision.camera.rtsp_source import (
    RTSPSource,
    RTSPSourceError,
)
from company_vision.core.settings import load_settings
from company_vision.utils.logging_config import (
    configure_logging,
    get_logger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Company Vision RTSP test"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--display",
        action="store_true",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Check configuration without opening a camera",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.frames <= 0:
        raise ValueError("--frames must be positive")

    settings = load_settings()
    configure_logging(settings)

    logger = get_logger("company_vision.node_2_3")

    url = os.environ.get("COMPANY_VISION_RTSP_URL", "")
    camera_id = os.environ.get(
        "COMPANY_VISION_RTSP_CAMERA_ID",
        "office_rtsp_01",
    )

    if args.check_config:
        logger.info("RTSP configuration check only")
        logger.info("URL configured: %s", bool(url))
        logger.info("Camera ID: %s", camera_id)
        print("NODE 2.3 CONFIG CHECK: PASS")
        print("REAL RTSP CONNECTION: NOT TESTED")
        return

    if not url:
        raise RTSPSourceError(
            "COMPANY_VISION_RTSP_URL is not configured"
        )

    source = RTSPSource(
        url=url,
        camera_id=camera_id,
        open_timeout_ms=5000,
        read_timeout_ms=5000,
    )

    count = 0
    first_timestamp = None
    last_timestamp = None

    try:
        with source:
            logger.info(
                "RTSP opened: camera=%s size=%dx%d fps=%.2f",
                source.camera_id,
                source.width,
                source.height,
                source.fps,
            )

            while count < args.frames:
                packet = source.read()

                assert packet.camera_id == camera_id
                assert packet.frame_id == count

                if last_timestamp is not None:
                    assert packet.timestamp >= last_timestamp

                if first_timestamp is None:
                    first_timestamp = packet.timestamp

                last_timestamp = packet.timestamp
                count += 1

                if args.display:
                    preview = packet.image.copy()

                    cv2.putText(
                        preview,
                        f"RTSP Frame: {packet.frame_id}",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

                    cv2.imshow(
                        "Company Vision - RTSP",
                        preview,
                    )

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

    finally:
        if args.display:
            cv2.destroyAllWindows()

    assert count > 0
    assert first_timestamp is not None
    assert last_timestamp is not None

    elapsed: timedelta = last_timestamp - first_timestamp

    logger.info("Frames received: %d", count)
    logger.info("Elapsed: %.3f sec", elapsed.total_seconds())

    print("NODE 2.3 RTSP SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
