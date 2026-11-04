from pathlib import Path
from time import perf_counter

import cv2
import torch
from ultralytics import YOLO

SOURCE = Path("data/outputs/dahua_channel_2.jpg")
OUTPUT_DIR = Path("data/outputs/dahua_yolo_test")
DEVICE = "cuda:0"

def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Image not found: {SOURCE}")

    image = cv2.imread(str(SOURCE))
    if image is None:
        raise RuntimeError("OpenCV could not read the image")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    height, width = image.shape[:2]

    print("=== COMPANY VISION / DAHUA YOLO TEST ===")
    print("SOURCE:", SOURCE)
    print("IMAGE SIZE:", f"{width}x{height}")
    print("GPU:", torch.cuda.get_device_name(0))

    models = [
        ("pose", "yolo26n-pose.pt"),
        ("detection", "yolo26n.pt"),
    ]

    for name, model_path in models:
        print(f"\n=== {name.upper()} ===")

        model = YOLO(model_path)

        start = perf_counter()
        results = model.predict(
            source=image,
            device=DEVICE,
            conf=0.25,
            verbose=False,
        )
        elapsed_ms = (perf_counter() - start) * 1000

        result = results[0]
        count = len(result.boxes) if result.boxes is not None else 0

        output_path = OUTPUT_DIR / f"dahua_{name}.jpg"
        annotated = result.plot()

        if not cv2.imwrite(str(output_path), annotated):
            raise RuntimeError(f"Failed saving {output_path}")

        print("PREDICTIONS:", count)
        print("END-TO-END TIME MS:", round(elapsed_ms, 2))
        print("OUTPUT:", output_path)
        print(f"{name.upper()} TEST: PASS")

    print("\nCOMPANY VISION DAHUA YOLO TEST: PASS")

if __name__ == "__main__":
    main()
