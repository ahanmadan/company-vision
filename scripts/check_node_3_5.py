
import argparse
from pathlib import Path

import cv2
import torch

from company_vision.camera.video_source import VideoFileSource
from company_vision.core.settings import load_settings
from company_vision.rendering.overlay_renderer import OverlayRenderer
from company_vision.vision.vision_pipeline import VisionPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Company Vision Node 3.5 Overlay Test"
    )

    parser.add_argument(
        "--video",
        type=Path,
        default=Path("data/samples/node_2_1_sample.avi"),
    )

    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/outputs/node_3_5_overlay.jpg"),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.frames <= 0:
        raise ValueError("--frames must be positive")

    if not args.video.is_file():
        raise FileNotFoundError(args.video)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    settings = load_settings()

    pipeline = VisionPipeline.from_models(
        pose_model=settings.models.pose_model,
        detection_model=settings.models.detection_model,
        device=args.device,
    )

    renderer = OverlayRenderer()

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed = 0
    total_persons = 0
    total_objects = 0
    last_output = None

    print("MODEL POSE:", settings.models.pose_model)
    print("MODEL DETECTION:", settings.models.detection_model)
    print("DEVICE:", args.device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    with VideoFileSource(
        args.video,
        camera_id="node_3_5_smoke_camera",
    ) as source:

        while processed < args.frames:
            packet = source.read()

            if packet is None:
                break

            result = pipeline.process(packet)

            annotated = renderer.render(
                packet,
                result,
            )

            if annotated.shape != packet.image.shape:
                raise RuntimeError("Rendered frame dimensions changed")

            if annotated.dtype != packet.image.dtype:
                raise RuntimeError("Rendered frame dtype changed")

            total_persons += len(result.persons)
            total_objects += len(result.objects)

            last_output = annotated
            processed += 1

    if processed == 0 or last_output is None:
        raise RuntimeError("No frame was rendered")

    if not cv2.imwrite(
        str(args.output),
        last_output,
    ):
        raise RuntimeError("Could not save rendered image")

    if not args.output.is_file():
        raise RuntimeError("Output image was not created")

    print("FRAMES PROCESSED:", processed)
    print("PERSON OBSERVATIONS:", total_persons)
    print("OBJECT OBSERVATIONS:", total_objects)
    print("OUTPUT IMAGE:", args.output)
    print("OUTPUT SIZE:", last_output.shape)
    print("NODE 3.5 OVERLAY SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
