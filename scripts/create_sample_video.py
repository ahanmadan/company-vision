
from pathlib import Path

import cv2
import numpy as np

OUTPUT_PATH = Path(
    "data/samples/node_2_1_sample.avi"
)


def create_sample_video(
    output_path: Path = OUTPUT_PATH,
    frame_count: int = 24,
    fps: float = 12.0,
) -> Path:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    width = 320
    height = 240

    fourcc = cv2.VideoWriter_fourcc(
        *"MJPG"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Cannot create sample video"
        )

    try:
        for frame_id in range(frame_count):
            frame = np.zeros(
                (height, width, 3),
                dtype=np.uint8,
            )

            x = 20 + frame_id * 8

            cv2.rectangle(
                frame,
                (x, 90),
                (x + 40, 150),
                (255, 180, 50),
                -1,
            )

            cv2.putText(
                frame,
                f"FRAME {frame_id:02d}",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            writer.write(frame)

    finally:
        writer.release()

    if (
        not output_path.is_file()
        or output_path.stat().st_size == 0
    ):
        raise RuntimeError(
            "Sample video was not created"
        )

    return output_path


if __name__ == "__main__":
    path = create_sample_video()
    print(f"SAMPLE VIDEO CREATED: {path}")
