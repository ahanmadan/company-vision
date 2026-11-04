import argparse
import json
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from company_vision.benchmarking.side_profile_audit import (
    analyze_pose_result_with_confidence,
    calculate_overlap_pairs,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        default="data/outputs/dahua_channel_2.jpg",
    )
    parser.add_argument(
        "--model",
        default="yolo26n-pose.pt",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--keypoint-conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)

    parser.add_argument(
        "--output",
        default="data/outputs/node_4_1",
    )

    args = parser.parse_args()

    source = Path(args.image)
    output = Path(args.output)

    if not source.is_file():
        raise FileNotFoundError(source)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    image = cv2.imread(str(source))

    if image is None:
        raise RuntimeError("Cannot read source image")

    height, width = image.shape[:2]

    print("=== NODE 4.1 SIDE PROFILE BASELINE ===")
    print("IMAGE:", source)
    print("RESOLUTION:", f"{width}x{height}")
    print("MODEL:", args.model)
    print("DEVICE:", args.device)

    model = YOLO(args.model)

    start = time.perf_counter()

    results = model.predict(
        source=image,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        verbose=False,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    result = results[0]

    if result.boxes is None or result.keypoints is None:
        raise RuntimeError("Pose model did not return pose outputs")

    boxes = result.boxes.xyxy.cpu().numpy()
    box_conf = result.boxes.conf.cpu().numpy()

    keypoints = result.keypoints.xy.cpu().numpy()
    keypoint_conf = result.keypoints.conf

    if keypoint_conf is None:
        raise RuntimeError("Keypoint confidence unavailable")

    keypoint_conf = keypoint_conf.cpu().numpy()

    audit = analyze_pose_result_with_confidence(
        boxes=boxes,
        box_confidence=box_conf,
        keypoints=keypoints,
        keypoint_confidence=keypoint_conf,
        image_width=width,
        image_height=height,
        confidence_threshold=args.conf,
        keypoint_threshold=args.keypoint_conf,
    )

    audit["source_image"] = str(source)
    audit["model"] = args.model
    audit["device"] = args.device
    audit["inference_ms"] = round(elapsed_ms, 2)
    audit["inference_fps"] = round(
        1000 / elapsed_ms,
        2,
    ) if elapsed_ms > 0 else 0

    audit["overlap_pairs"] = calculate_overlap_pairs(
        audit["persons"]
    )

    audit["overlap_pair_count"] = len(
        audit["overlap_pairs"]
    )

    output.mkdir(parents=True, exist_ok=True)

    annotated = result.plot(
        boxes=True,
        labels=True,
        conf=True,
    )

    preview_path = output / "node_4_1_baseline.jpg"
    report_path = output / "node_4_1_report.json"

    if not cv2.imwrite(str(preview_path), annotated):
        raise RuntimeError("Failed to save preview")

    report_path.write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== BASELINE RESULTS ===")
    print("PERSON CANDIDATES:", audit["person_candidates"])
    print("ACCEPTED PERSONS:", audit["accepted_persons"])
    print("REJECTED PERSONS:", audit["rejected_persons"])
    print(
        "MEAN DETECTION CONFIDENCE:",
        audit["mean_detection_confidence"],
    )
    print(
        "MEAN KEYPOINT VISIBILITY:",
        audit["mean_keypoint_visibility"],
    )
    print("OVERLAP PAIRS:", audit["overlap_pair_count"])
    print("INFERENCE MS:", audit["inference_ms"])
    print("INFERENCE FPS:", audit["inference_fps"])
    print("PREVIEW:", preview_path)
    print("REPORT:", report_path)

    print("\nNODE 4.1 BASELINE AUDIT: PASS")


if __name__ == "__main__":
    main()