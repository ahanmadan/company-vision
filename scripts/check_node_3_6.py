
import argparse
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import cv2
import torch
import ultralytics

from company_vision.benchmarking.evaluator import (
    BenchmarkEvaluator,
    load_annotations,
)
from company_vision.camera.video_source import VideoFileSource
from company_vision.core.settings import load_settings
from company_vision.rendering.overlay_renderer import OverlayRenderer
from company_vision.vision.vision_pipeline import VisionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Company Vision Phase 3 Benchmark"
    )

    parser.add_argument(
        "--mode",
        choices=("smoke", "real"),
        default="smoke",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("data/samples/node_2_1_sample.avi"),
    )
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/outputs/node_3_6_benchmark.json"),
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=None,
        help="Optional annotated preview image",
    )

    return parser.parse_args()


def percentile_95(values: list[float]) -> float:
    ordered = sorted(values)
    index = max(0, (95 * len(ordered) + 99) // 100 - 1)
    return ordered[min(index, len(ordered) - 1)]


def main() -> None:
    args = parse_args()

    if args.frames <= 0:
        raise ValueError("--frames must be positive")

    if args.warmup < 0 or args.warmup >= args.frames:
        raise ValueError("warmup must be less than frames")

    if not args.video.is_file():
        raise FileNotFoundError(args.video)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    annotations = None

    if args.mode == "real":
        if args.annotations is None:
            raise ValueError(
                "Real mode requires --annotations"
            )

        if not args.annotations.is_file():
            raise FileNotFoundError(args.annotations)

        annotations = load_annotations(args.annotations)

    settings = load_settings()

    pipeline = VisionPipeline.from_models(
        pose_model=settings.models.pose_model,
        detection_model=settings.models.detection_model,
        device=args.device,
    )

    renderer = OverlayRenderer()
    evaluator = (
        BenchmarkEvaluator()
        if annotations is not None
        else None
    )

    durations = []
    processed = 0
    persons = 0
    objects = 0
    partial = 0
    annotated_persons = 0
    last_image = None

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    with VideoFileSource(
        args.video,
        camera_id="phase3_benchmark",
    ) as source:
        while processed < args.frames:
            packet = source.read()

            if packet is None:
                break

            if (
                annotations is not None
                and packet.frame_id not in annotations
            ):
                raise ValueError(
                    "Missing annotation for frame "
                    f"{packet.frame_id}"
                )

            if args.device.startswith("cuda"):
                torch.cuda.synchronize()

            started = perf_counter()

            result = pipeline.process(packet)
            image = renderer.render(packet, result)

            if args.device.startswith("cuda"):
                torch.cuda.synchronize()

            elapsed_ms = (perf_counter() - started) * 1000.0

            if image.shape != packet.image.shape:
                raise RuntimeError("Rendered frame shape mismatch")

            if image.dtype != packet.image.dtype:
                raise RuntimeError("Rendered frame dtype mismatch")

            if processed >= args.warmup:
                durations.append(elapsed_ms)

            if evaluator is not None:
                reference = annotations[packet.frame_id]
                annotated_persons += len(reference["persons"])
                evaluator.update(
                    packet.frame_id,
                    result,
                    reference,
                )

            persons += len(result.persons)
            objects += len(result.objects)

            if result.status == "partial":
                partial += 1

            processed += 1
            last_image = image

    if processed <= args.warmup:
        raise RuntimeError(
            "Not enough frames after warmup"
        )

    if args.mode == "real" and annotated_persons == 0:
        raise ValueError(
            "Real benchmark needs annotated persons"
        )

    report = {
        "benchmark_mode": args.mode,
        "validation_level": (
            "ANNOTATED_VIDEO_EVALUATION"
            if args.mode == "real"
            else "SYNTHETIC_SOFTWARE_SMOKE"
        ),
        "environment": {
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "device": args.device,
            "gpu": (
                torch.cuda.get_device_name(0)
                if args.device.startswith("cuda")
                else None
            ),
            "pose_model": settings.models.pose_model,
            "detection_model": settings.models.detection_model,
            "tracker": "bytetrack.yaml",
        },
        "processing": {
            "frames": processed,
            "warmup_frames": args.warmup,
            "measured_frames": len(durations),
            "person_observations": persons,
            "object_observations": objects,
            "partial_frames": partial,
            "mean_ms": round(mean(durations), 3),
            "median_ms": round(median(durations), 3),
            "p95_ms": round(percentile_95(durations), 3),
            "processing_only_fps": round(
                1000.0 / mean(durations), 3
            ),
            "measurement_scope": (
                "VisionPipeline.process plus "
                "OverlayRenderer.render; excludes file I/O"
            ),
        },
        "accuracy": (
            evaluator.report()
            if evaluator is not None
            else None
        ),
        "limitations": [
            "No real RTSP measured",
            "No sustained production throughput guarantee",
            "Pipeline currently performs repeated detection",
        ],
    }

    if args.device.startswith("cuda"):
        report["processing"]["peak_cuda_allocated_mb"] = round(
            torch.cuda.max_memory_allocated(0) / 1048576,
            2,
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    if args.preview is not None:
        if last_image is None:
            raise RuntimeError("No image available for preview")

        args.preview.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not cv2.imwrite(str(args.preview), last_image):
            raise RuntimeError("Preview could not be written")

    print("MODE:", args.mode)
    print("FRAMES:", processed)
    print("WARMUP:", args.warmup)
    print("MEAN MS:", report["processing"]["mean_ms"])
    print("P95 MS:", report["processing"]["p95_ms"])
    print("PROCESSING FPS:", report["processing"]["processing_only_fps"])
    print("PERSON OBSERVATIONS:", persons)
    print("OBJECT OBSERVATIONS:", objects)
    print("REPORT:", args.output)

    if evaluator is not None:
        print("ACCURACY:", json.dumps(evaluator.report()))

    print("NODE 3.6 BENCHMARK EXECUTION: PASS")


if __name__ == "__main__":
    main()
