
import argparse
from datetime import timedelta

import cv2

from company_vision.camera.webcam_source import WebcamSource
from company_vision.core.settings import load_settings
from company_vision.utils.logging_config import (
    configure_logging,
    get_logger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Company Vision webcam smoke test"
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Webcam index",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=50,
        help="Number of frames to capture",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display live frames",
    )
    parser.add_argument(
        "--backend",
        choices=["any", "dshow", "msmf"],
        default="any",
    )
    return parser.parse_args()


def resolve_backend(name: str) -> int:
    return {
        "any": cv2.CAP_ANY,
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
    }[name]


def main() -> None:
    args = parse_args()

    if args.frames <= 0:
        raise ValueError("--frames must be positive")

    settings = load_settings()
    configure_logging(settings)
    logger = get_logger("company_vision.node_2_2")

    received = 0
    first_timestamp = None
    previous_timestamp = None

    logger.info(
        "Opening webcam index=%d backend=%s",
        args.index,
        args.backend,
    )

    try:
        with WebcamSource(
            camera_index=args.index,
            camera_id="office_webcam_test",
            width=640,
            height=480,
            backend=resolve_backend(args.backend),
        ) as source:
            logger.info(
                "Camera metadata: %dx%d FPS=%.2f",
                source.width,
                source.height,
                source.fps,
            )

            while received < args.frames:
                packet = source.read()

                assert packet.frame_id == received
                assert packet.camera_id == "office_webcam_test"

                if previous_timestamp is not None:
                    assert packet.timestamp >= previous_timestamp

                if first_timestamp is None:
                    first_timestamp = packet.timestamp

                previous_timestamp = packet.timestamp
                received += 1

                if args.display:
                    frame = packet.image.copy()

                    label = (
                        f"Camera: {packet.camera_id} "
                        f"Frame: {packet.frame_id}"
                    )

                    cv2.putText(
                        frame,
                        label,
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                    cv2.imshow("Company Vision - Webcam", frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("User stopped preview")
                        break

    finally:
        if args.display:
            cv2.destroyAllWindows()

    assert received > 0
    assert first_timestamp is not None
    assert previous_timestamp is not None

    elapsed: timedelta = previous_timestamp - first_timestamp

    logger.info("Frames received: %d", received)
    logger.info("Elapsed time: %.3f s", elapsed.total_seconds())

    print("NODE 2.2 SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
