import json
import time
from pathlib import Path

import torch
from ultralytics import YOLO

IMAGE = Path("data/outputs/dahua_channel_2.jpg")
OUTPUT = Path("data/outputs/pose_optimization")
MODELS = ["yolo26n-pose.pt", "yolo26m-pose.pt"]
RESOLUTIONS = [640, 960, 1280]
WARMUP = 3
ITERATIONS = 20


def main():
    if not IMAGE.exists():
        raise FileNotFoundError(IMAGE)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    image = str(IMAGE)
    reports = []

    print("GPU:", torch.cuda.get_device_name(0))
    print("IMAGE:", IMAGE)
    print("ITERATIONS:", ITERATIONS)

    for model_name in MODELS:
        print("\nLOADING:", model_name)
        model = YOLO(model_name)

        for size in RESOLUTIONS:
            print(f"\nMODEL={model_name} IMGSZ={size}")

            for _ in range(WARMUP):
                model.predict(
                    source=image,
                    imgsz=size,
                    device=0,
                    conf=0.15,
                    verbose=False,
                )

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            durations = []
            result = None

            for _ in range(ITERATIONS):
                torch.cuda.synchronize()
                start = time.perf_counter()

                result = model.predict(
                    source=image,
                    imgsz=size,
                    device=0,
                    conf=0.15,
                    verbose=False,
                )[0]

                torch.cuda.synchronize()
                durations.append(
                    (time.perf_counter() - start) * 1000
                )

            boxes = result.boxes
            keypoints = result.keypoints

            count = len(boxes)
            confidences = (
                boxes.conf.detach().cpu().tolist()
                if count else []
            )

            visibility = None
            if keypoints is not None and keypoints.data is not None:
                points = keypoints.data.detach().cpu()

                if points.shape[-1] >= 3 and points.numel():
                    visibility = float(
                        (points[..., 2] >= 0.25).float().mean()
                    )

            average_ms = sum(durations) / len(durations)
            sorted_ms = sorted(durations)
            p95_ms = sorted_ms[
                min(len(sorted_ms) - 1, int(len(sorted_ms) * 0.95))
            ]

            memory_mb = (
                torch.cuda.max_memory_allocated() / 1024**2
            )

            annotated = OUTPUT / (
                f"{model_name.replace('.pt', '')}_{size}.jpg"
            )

            if not result.save(filename=str(annotated)):
                raise RuntimeError("Could not save annotated image")

            report = {
                "model": model_name,
                "imgsz": size,
                "iterations": ITERATIONS,
                "persons_detected": count,
                "mean_confidence": (
                    sum(confidences) / len(confidences)
                    if confidences else None
                ),
                "keypoint_visibility_ratio": visibility,
                "mean_inference_ms": round(average_ms, 2),
                "p95_inference_ms": round(p95_ms, 2),
                "inference_fps": round(1000 / average_ms, 2),
                "peak_cuda_memory_mb": round(memory_mb, 2),
                "annotated_image": str(annotated),
            }

            reports.append(report)
            print(json.dumps(report, indent=2))

            del result
            torch.cuda.empty_cache()

        del model
        torch.cuda.empty_cache()

    report_path = OUTPUT / "benchmark_report.json"

    report_path.write_text(
        json.dumps(reports, indent=2),
        encoding="utf-8",
    )

    print("\nBENCHMARK COMPLETE")
    print("REPORT:", report_path)


if __name__ == "__main__":
    main()
