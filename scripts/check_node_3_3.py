
import argparse
from pathlib import Path
from statistics import mean

import torch

from company_vision.camera.video_source import VideoFileSource
from company_vision.core.settings import load_settings
from company_vision.vision.person_tracker import PersonTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Company Vision Node 3.3 ByteTrack smoke test"
    )

    parser.add_argument(
        "--video",
        type=Path,
        default=Path("data/samples/node_2_1_sample.avi"),
    )
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")

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
    tracker_config = "bytetrack.yaml"

    print("MODEL:", model_path)
    print("TRACKER:", tracker_config)
    print("DEVICE:", args.device)
    print("TORCH:", torch.__version__)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    tracker = PersonTracker(
        model_path=model_path,
        device=args.device,
        tracker_config=tracker_config,
        confidence=0.1,
        image_size=640,
    )

    frames = 0
    total_tracked_persons = 0
    unique_ids: set[int] = set()
    durations: list[float] = []

    with VideoFileSource(
        args.video,
        camera_id="tracking_test_video",
    ) as source:
        while frames < args.frames:
            packet = source.read()

            if packet is None:
                break

            output = tracker.track(packet)

            assert output.packet.frame_id == packet.frame_id

            for person in output.persons:
                assert person.track_id >= 0
                assert len(person.bbox) == 4
                assert 0 <= person.confidence <= 1
                unique_ids.add(person.track_id)

            total_tracked_persons += output.person_count
            durations.append(output.inference_ms)
            frames += 1

    if frames == 0:
        raise RuntimeError("No video frames processed")

    assert tracker.processed_frames == frames

    print("FRAMES:", frames)
    print("TRACKED PERSON OBSERVATIONS:", total_tracked_persons)
    print("UNIQUE TRACK IDS:", len(unique_ids))
    print("MEAN TRACKING MS:", round(mean(durations), 2))

    if args.device.startswith("cuda"):
        print(
            "CUDA MEMORY ALLOCATED MB:",
            round(torch.cuda.memory_allocated(0) / 1048576, 2),
        )

    print("NODE 3.3 REAL MODEL SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
