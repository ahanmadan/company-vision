
import argparse
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from company_vision.camera.video_source import VideoFileSource
from company_vision.core.settings import load_settings
from company_vision.vision.vision_pipeline import VisionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Company Vision Node 3.4 integration smoke test"
    )

    parser.add_argument(
        "--video",
        type=Path,
        default=Path("data/samples/node_2_1_sample.avi"),
    )
    parser.add_argument("--frames", type=int, default=5)
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

    print("POSE MODEL:", settings.models.pose_model)
    print("DETECTION MODEL:", settings.models.detection_model)
    print("TRACKER: bytetrack.yaml")
    print("DEVICE:", args.device)
    print("TORCH:", torch.__version__)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    pipeline = VisionPipeline.from_models(
        pose_model=settings.models.pose_model,
        detection_model=settings.models.detection_model,
        device=args.device,
    )

    frames = 0
    total_persons = 0
    total_objects = 0
    partial_frames = 0
    durations: list[float] = []

    with VideoFileSource(
        args.video,
        camera_id="pipeline_smoke_camera",
    ) as source:
        while frames < args.frames:
            packet = source.read()

            if packet is None:
                break

            started = perf_counter()
            result = pipeline.process(packet)
            durations.append(
                (perf_counter() - started) * 1000.0
            )

            assert result.camera_id == packet.camera_id
            assert result.frame_id == packet.frame_id
            assert result.timestamp == packet.timestamp
            assert result.width == packet.width
            assert result.height == packet.height
            assert result.status in ("ok", "partial")

            # The Pydantic result must be serializable.
            serialized = result.model_dump(mode="json")

            assert serialized["frame_id"] == packet.frame_id

            for person in result.persons:
                if person.track_id is not None:
                    assert person.track_id >= 0

            total_persons += len(result.persons)
            total_objects += len(result.objects)

            if result.status == "partial":
                partial_frames += 1

            frames += 1

    if frames == 0:
        raise RuntimeError("No frames were processed")

    stats = pipeline.stats()

    assert stats.processed_frames == frames
    assert stats.total_persons == total_persons
    assert stats.total_objects == total_objects
    assert stats.partial_frames == partial_frames

    print("FRAMES:", frames)
    print("PERSON OBSERVATIONS:", total_persons)
    print("OBJECT OBSERVATIONS:", total_objects)
    print("PARTIAL FRAMES:", partial_frames)
    print("MEAN PIPELINE MS:", round(mean(durations), 2))

    if args.device.startswith("cuda"):
        print(
            "CUDA MEMORY ALLOCATED MB:",
            round(torch.cuda.memory_allocated(0) / 1048576, 2),
        )

    print("NODE 3.4 INTEGRATION SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
