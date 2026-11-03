
import argparse
from collections import Counter
from pathlib import Path
from statistics import mean

import torch

from company_vision.camera.video_source import VideoFileSource
from company_vision.core.settings import load_settings
from company_vision.vision.object_detector import (
    OFFICE_CLASS_IDS,
    ObjectDetector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Company Vision Node 3.2 detection test"
    )

    parser.add_argument(
        "--video",
        type=Path,
        default=Path("data/samples/node_2_1_sample.avi"),
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
    )
    parser.add_argument(
        "--office-only",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.frames <= 0:
        raise ValueError("--frames must be positive")

    if not args.video.is_file():
        raise FileNotFoundError(
            f"Video file not found: {args.video}"
        )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable")

    settings = load_settings()

    model_path = settings.models.detection_model

    print("MODEL:", model_path)
    print("DEVICE:", args.device)
    print("TORCH:", torch.__version__)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    class_ids = (
        set(OFFICE_CLASS_IDS)
        if args.office_only
        else None
    )

    detector = ObjectDetector(
        model_path=model_path,
        device=args.device,
        confidence=0.35,
        image_size=640,
        class_ids=class_ids,
    )

    frame_count = 0
    object_count = 0
    durations = []
    class_counts: Counter[str] = Counter()

    with VideoFileSource(
        args.video,
        camera_id="detection_test_video",
    ) as source:
        while frame_count < args.frames:
            packet = source.read()

            if packet is None:
                break

            output = detector.predict(packet)

            assert output.packet.frame_id == packet.frame_id
            assert output.packet.camera_id == packet.camera_id

            for obj in output.objects:
                assert 0 <= obj.confidence <= 1
                assert len(obj.bbox) == 4
                class_counts[obj.class_name] += 1

            object_count += output.object_count
            durations.append(output.inference_ms)
            frame_count += 1

    if frame_count == 0:
        raise RuntimeError("No video frames processed")

    print("FRAMES:", frame_count)
    print("TOTAL OBJECT DETECTIONS:", object_count)
    print("MEAN INFERENCE MS:", round(mean(durations), 2))
    print("DETECTIONS BY CLASS:")

    for name, count in class_counts.most_common():
        print(f"  {name}: {count}")

    if not class_counts:
        print("  none")

    if args.device.startswith("cuda"):
        print(
            "CUDA MEMORY ALLOCATED MB:",
            round(torch.cuda.memory_allocated(0) / 1048576, 2),
        )

    print("NODE 3.2 REAL MODEL SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
