"""Live object evidence, wrist matching and diagnostic overlay."""

from __future__ import annotations

import cv2
import numpy as np

CLASS_IDS = (63, 64, 66, 67)
LABELS = (
    ("phone_usage", "PHONE"),
    ("computer_usage", "COMPUTER"),
    ("inactivity", "INACTIVITY"),
    ("gathering", "GATHERING"),
)


def iou(a, b):
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(aa + bb - intersection, 1e-9)


def result_objects(result, offset=(0, 0), scale=1.0):
    objects = []
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return objects

    xyxy = boxes.xyxy.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy()
    scores = boxes.conf.detach().cpu().numpy()

    for index, box in enumerate(xyxy):
        cid = int(classes[index])
        if cid not in CLASS_IDS:
            continue

        converted = [
            float(box[0]) * scale + offset[0],
            float(box[1]) * scale + offset[1],
            float(box[2]) * scale + offset[0],
            float(box[3]) * scale + offset[1],
        ]

        objects.append({
            "object_id": index,
            "class_id": cid,
            "class_name": str(result.names[cid]),
            "confidence": float(scores[index]),
            "bbox": converted,
        })

    return objects


def detect_objects(model, frame, tracks, *, imgsz=1280):
    """Global detection + native-resolution person-region crops."""

    height, width = frame.shape[:2]

    def predict(image, conf):
        return model.predict(
            source=image,
            device="cuda:0",
            imgsz=imgsz,
            conf=conf,
            classes=list(CLASS_IDS),
            verbose=False,
        )[0]

    candidates = result_objects(predict(frame, 0.18))

    # Additional crops enlarge small phones relative to YOLO input.
    eligible = [
        person for person in tracks
        if person.get("observed") is True
    ]

    for person in eligible[:8]:
        x1, y1, x2, y2 = map(float, person["bbox"])

        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        left = max(0, int(x1 - bw * 0.20))
        top = max(0, int(y1 - bh * 0.20))
        right = min(width, int(x2 + bw * 0.20))
        bottom = min(height, int(y2 + bh * 0.20))

        if right - left < 48 or bottom - top < 48:
            continue

        crop = frame[top:bottom, left:right]

        candidates.extend(
            result_objects(
                predict(crop, 0.15),
                offset=(left, top),
            )
        )

    # Deduplicate full-frame and ROI detections.
    candidates.sort(
        key=lambda obj: obj["confidence"],
        reverse=True,
    )

    selected = []

    for item in candidates:
        box = item["bbox"]
        box[0] = max(0.0, min(width - 1.0, box[0]))
        box[1] = max(0.0, min(height - 1.0, box[1]))
        box[2] = max(0.0, min(float(width), box[2]))
        box[3] = max(0.0, min(float(height), box[3]))

        if box[2] <= box[0] or box[3] <= box[1]:
            continue

        duplicate = any(
            old["class_id"] == item["class_id"]
            and iou(old["bbox"], item["bbox"]) > 0.45
            for old in selected
        )

        if not duplicate:
            selected.append(item)

    for index, item in enumerate(selected):
        item["object_id"] = index

    return selected


def match_wrists(tracks, boxes, coco17, frame_shape):
    """Associate COCO wrists only to unambiguous observed Track IDs."""

    if coco17 is None:
        return {}

    boxes = np.asarray(boxes, dtype=np.float32)
    poses = np.asarray(coco17, dtype=np.float32)

    if boxes.ndim != 2 or boxes.shape[1:] != (4,):
        return {}

    if poses.shape != (len(boxes), 17, 3):
        return {}

    height, width = frame_shape[:2]
    candidates = []

    for person in tracks:
        if not person.get("observed", False):
            continue

        if not person.get("behavior_eligible", False):
            continue

        scores = [
            (iou(person["bbox"], box), index)
            for index, box in enumerate(boxes)
        ]

        scores.sort(reverse=True)

        if not scores or scores[0][0] < 0.35:
            continue

        if len(scores) > 1 and scores[0][0] - scores[1][0] < 0.12:
            continue

        candidates.append((
            scores[0][0],
            int(person["track_id"]),
            scores[0][1],
        ))

    # A detection can supplement only one stable Track ID.
    candidates.sort(reverse=True)
    assigned = set()
    wrists = {}

    for _, track_id, detection_index in candidates:
        if detection_index in assigned:
            continue

        assigned.add(detection_index)
        extras = {}

        for name, keypoint_index in (
            ("left_wrist", 9),
            ("right_wrist", 10),
        ):
            point = poses[detection_index, keypoint_index].copy()

            if not np.isfinite(point).all():
                continue

            if not (
                0 <= point[0] <= width
                and 0 <= point[1] <= height
                and 0 <= point[2] <= 1
            ):
                continue

            extras[name] = point.tolist()

        if extras:
            wrists[track_id] = extras

    return wrists


def diagnostic_rows(output):
    """Use actual plugin evidence rather than guessed behavior labels."""

    rows = {}

    for result in output.results:
        evidence = result.evidence
        events = result.events
        lines = []

        for plugin_id, title in LABELS:
            item = evidence.get(plugin_id, {})
            status = item.get("status", "unknown")
            metrics = item.get("metrics", {})
            reason = item.get("evidence", {}).get(
                "reason", "no_reason"
            )

            confirmed = any(
                event.get("plugin_id") == plugin_id
                and event.get("status") == "confirmed"
                for event in events
            )

            if confirmed:
                label = "CONFIRMED"
            elif status == "ok" and "score" in metrics:
                label = f'EVIDENCE {metrics["score"]:.2f}'
            else:
                label = status.upper()

            lines.append(f"{title}: {label} | {reason}")

        rows[int(result.track_id)] = lines

    return rows


def draw_diagnostics(image, output, objects):
    """Four independent behavior statuses for each visible person."""

    height, width = image.shape[:2]

    for obj in objects:
        x1, y1, x2, y2 = map(int, obj["bbox"])
        phone = obj["class_id"] == 67
        color = (0, 220, 255) if phone else (255, 180, 0)

        cv2.rectangle(
            image, (x1, y1), (x2, y2), color, 2
        )

        name = "PHONE" if phone else obj["class_name"]

        cv2.putText(
            image,
            f'{name} {obj["confidence"]:.2f}',
            (max(0, x1), max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52, color, 2,
        )

    rows = diagnostic_rows(output)

    for overlay in output.overlays:
        if not overlay.show:
            continue

        x1, y1, x2, _ = overlay.bbox_xyxy

        # Put diagnostics beside the person when space permits.
        x = x2 + 10 if x2 + 430 < width else max(0, x1 - 430)
        y = max(185, min(height - 95, y1 + 20))

        lines = rows.get(overlay.track_id, [])

        for line_index, label in enumerate(lines):
            cv2.putText(
                image,
                label[:65],
                (x, y + line_index * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (255, 255, 255),
                1,
            )

    count = sum(obj["class_id"] == 67 for obj in objects)

    cv2.putText(
        image,
        f"OBJECTS: {len(objects)} | PHONES: {count}",
        (25, 154),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70, (0, 220, 255), 2,
    )

    return image