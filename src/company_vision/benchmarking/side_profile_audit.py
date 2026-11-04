from __future__ import annotations

import numpy as np


def analyze_pose_result(
    boxes,
    keypoints,
    image_width,
    image_height,
    confidence_threshold=0.35,
    keypoint_threshold=0.25,
):
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")

    if not 0 <= confidence_threshold <= 1:
        raise ValueError("Invalid confidence threshold")

    if not 0 <= keypoint_threshold <= 1:
        raise ValueError("Invalid keypoint threshold")

    boxes = np.asarray(boxes, dtype=np.float32)

    if boxes.size == 0:
        boxes = np.empty((0, 4), dtype=np.float32)

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4)")

    if not np.isfinite(boxes).all():
        raise ValueError("Invalid box values")

    if np.any(boxes[:, 2] <= boxes[:, 0]):
        raise ValueError("Invalid box width")

    if np.any(boxes[:, 3] <= boxes[:, 1]):
        raise ValueError("Invalid box height")

    if np.any(boxes < 0):
        raise ValueError("Negative box coordinates")

    if np.any(boxes[:, 2] > image_width):
        raise ValueError("Box exceeds image width")

    if np.any(boxes[:, 3] > image_height):
        raise ValueError("Box exceeds image height")

    keypoints = np.asarray(keypoints, dtype=np.float32)

    if keypoints.size == 0:
        keypoints = np.empty(
            (len(boxes), 17, 2),
            dtype=np.float32,
        )

    if keypoints.shape != (len(boxes), 17, 2):
        raise ValueError("Expected keypoints shape (N, 17, 2)")

    if not np.isfinite(keypoints).all():
        raise ValueError("Invalid keypoint coordinates")

    persons = []

    for index, (box, points) in enumerate(zip(boxes, keypoints)):
        x1, y1, x2, y2 = map(float, box)

        person_points = [
            {
                "x": round(float(point[0]), 2),
                "y": round(float(point[1]), 2),
                "confidence": None,
                "visible": None,
            }
            for point in points
        ]

        persons.append({
            "detection_index": index,
            "bbox": [
                round(x1, 2),
                round(y1, 2),
                round(x2, 2),
                round(y2, 2),
            ],
            "bbox_width": round(x2 - x1, 2),
            "bbox_height": round(y2 - y1, 2),
            "center": [
                round((x1 + x2) / 2, 2),
                round((y1 + y2) / 2, 2),
            ],
            "normalized_center": [
                round((x1 + x2) / (2 * image_width), 4),
                round((y1 + y2) / (2 * image_height), 4),
            ],
            "keypoints": person_points,
        })

    return {
        "image_width": image_width,
        "image_height": image_height,
        "person_candidates": len(boxes),
        "confidence_threshold": confidence_threshold,
        "keypoint_threshold": keypoint_threshold,
        "persons": persons,
    }


def analyze_pose_result_with_confidence(
    boxes,
    box_confidence,
    keypoints,
    keypoint_confidence,
    image_width,
    image_height,
    confidence_threshold=0.35,
    keypoint_threshold=0.25,
):
    boxes = np.asarray(boxes, dtype=np.float32)
    box_confidence = np.asarray(
        box_confidence,
        dtype=np.float32,
    ).reshape(-1)

    keypoints = np.asarray(keypoints, dtype=np.float32)
    keypoint_confidence = np.asarray(
        keypoint_confidence,
        dtype=np.float32,
    )

    if len(boxes) != len(box_confidence):
        raise ValueError("Box confidence count mismatch")

    if keypoint_confidence.shape != (len(boxes), 17):
        raise ValueError("Expected keypoint confidence shape (N, 17)")

    if not np.isfinite(box_confidence).all():
        raise ValueError("Invalid box confidence")

    if not np.isfinite(keypoint_confidence).all():
        raise ValueError("Invalid keypoint confidence")

    if np.any((box_confidence < 0) | (box_confidence > 1)):
        raise ValueError("Box confidence must be between 0 and 1")

    if np.any(
        (keypoint_confidence < 0)
        | (keypoint_confidence > 1)
    ):
        raise ValueError("Keypoint confidence must be between 0 and 1")

    result = analyze_pose_result(
        boxes,
        keypoints,
        image_width,
        image_height,
        confidence_threshold,
        keypoint_threshold,
    )

    records = []

    for index, (box, confidence) in enumerate(
        zip(boxes, box_confidence)
    ):
        x1, y1, x2, y2 = map(float, box)

        scores = keypoint_confidence[index]
        visible = scores >= keypoint_threshold

        points = []

        for point, score, is_visible in zip(
            keypoints[index],
            scores,
            visible,
        ):
            points.append({
                "x": round(float(point[0]), 2),
                "y": round(float(point[1]), 2),
                "confidence": round(float(score), 4),
                "visible": bool(is_visible),
            })

        records.append({
            "detection_index": index,
            "bbox": [
                round(x1, 2),
                round(y1, 2),
                round(x2, 2),
                round(y2, 2),
            ],
            "bbox_width": round(x2 - x1, 2),
            "bbox_height": round(y2 - y1, 2),
            "center": [
                round((x1 + x2) / 2, 2),
                round((y1 + y2) / 2, 2),
            ],
            "normalized_center": [
                round((x1 + x2) / (2 * image_width), 4),
                round((y1 + y2) / (2 * image_height), 4),
            ],
            "detection_confidence": round(float(confidence), 4),
            "accepted_by_confidence": bool(
                confidence >= confidence_threshold
            ),
            "keypoints": points,
            "visible_keypoints": int(visible.sum()),
            "low_confidence_keypoints": int((~visible).sum()),
            "keypoint_visibility_ratio": round(
                float(visible.mean()),
                4,
            ),
        })

    result["persons"] = records

    accepted = [
        person for person in records
        if person["accepted_by_confidence"]
    ]

    result["accepted_persons"] = len(accepted)
    result["rejected_persons"] = len(records) - len(accepted)

    result["mean_detection_confidence"] = (
        round(float(box_confidence.mean()), 4)
        if len(box_confidence)
        else 0.0
    )

    result["mean_keypoint_visibility"] = (
        round(
            float(np.mean([
                person["keypoint_visibility_ratio"]
                for person in records
            ])),
            4,
        )
        if records
        else 0.0
    )

    return result


def calculate_box_iou(box_a, box_b):
    a = np.asarray(box_a, dtype=np.float32)
    b = np.asarray(box_b, dtype=np.float32)

    if a.shape != (4,) or b.shape != (4,):
        raise ValueError("Each box must have four coordinates")

    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Invalid box coordinates")

    if a[2] <= a[0] or a[3] <= a[1]:
        raise ValueError("Invalid first box")

    if b[2] <= b[0] or b[3] <= b[1]:
        raise ValueError("Invalid second box")

    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    union = area_a + area_b - intersection

    return float(intersection / union) if union else 0.0


def calculate_overlap_pairs(persons):
    pairs = []

    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            iou = calculate_box_iou(
                persons[i]["bbox"],
                persons[j]["bbox"],
            )

            if iou > 0:
                pairs.append({
                    "person_a": persons[i]["detection_index"],
                    "person_b": persons[j]["detection_index"],
                    "iou": round(iou, 4),
                })

    return pairs

