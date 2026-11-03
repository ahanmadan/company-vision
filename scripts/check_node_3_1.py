
import argparse
from pathlib import Path
from statistics import mean

import torch

from company_vision.camera.video_source import VideoFileSource
from company_vision.core.settings import load_settings
from company_vision.vision.pose_estimator import PoseEstimator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Company Vision Node 3.1 GPU smoke test"
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("data/samples/node_2_1_sample.avi"),
    )
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.frames <= 0:
        raise ValueError("--frames must be positive")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable")

    if not args.video.is_file():
        raise FileNotFoundError(
            f"Video not found: {args.video}"
        )

    settings = load_settings()

    print("MODEL:", settings.models.pose_model)
    print("DEVICE:", args.device)
    print("TORCH:", torch.__version__)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    estimator = PoseEstimator(
        model_path=settings.models.pose_model,
        device=args.device,
        confidence=0.35,
        image_size=640,
    )

    frame_count = 0
    person_detections = 0
    durations = []

    with VideoFileSource(
        args.video,
        camera_id="pose_test_video",
    ) as source:
        while frame_count < args.frames:
            packet = source.read()

            if packet is None:
                break

            result = estimator.predict(packet)

            assert result.packet.frame_id == packet.frame_id
            assert result.packet.camera_id == packet.camera_id

            for person in result.people:
                assert len(person.keypoints) == 17

            person_detections += result.person_count
            durations.append(result.inference_ms)
            frame_count += 1

    if frame_count == 0:
        raise RuntimeError("No video frames processed")

    print("FRAMES:", frame_count)
    print("PERSON DETECTIONS:", person_detections)
    print("MEAN INFERENCE MS:", round(mean(durations), 2))

    if args.device.startswith("cuda"):
        print(
            "CUDA MEMORY ALLOCATED MB:",
            round(torch.cuda.memory_allocated(0) / 1048576, 2),
        )

    print("NODE 3.1 REAL MODEL SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
