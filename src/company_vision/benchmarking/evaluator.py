
import json
from dataclasses import dataclass
from math import hypot, isfinite
from pathlib import Path
from typing import Any


def box_tuple(value: Any) -> tuple[float, float, float, float]:
    if isinstance(value, (list, tuple)):
        values = value
    else:
        values = [
            getattr(value, name)
            for name in ("x1", "y1", "x2", "y2")
        ]

    if len(values) != 4:
        raise ValueError("Bounding box must contain four values")

    if any(
        isinstance(v, bool)
        or not isinstance(v, (int, float))
        or not isfinite(v)
        for v in values
    ):
        raise ValueError("Bounding box contains invalid values")

    x1, y1, x2, y2 = (float(v) for v in values)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid bounding box dimensions")

    return x1, y1, x2, y2


def iou(a: Any, b: Any) -> float:
    ax1, ay1, ax2, ay2 = box_tuple(a)
    bx1, by1, bx2, by2 = box_tuple(b)

    intersection = (
        max(0.0, min(ax2, bx2) - max(ax1, bx1))
        * max(0.0, min(ay2, by2) - max(ay1, by1))
    )

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def match_boxes(
    ground_truth: list[Any],
    predictions: list[Any],
    threshold: float = 0.5,
) -> list[tuple[int, int]]:
    candidates = []

    for gi, gt in enumerate(ground_truth):
        for pi, pred in enumerate(predictions):
            overlap = iou(gt, pred)
            if overlap >= threshold:
                candidates.append((overlap, gi, pi))

    candidates.sort(key=lambda item: -item[0])

    matched_gt = set()
    matched_pred = set()
    matches = []

    for _, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue

        matched_gt.add(gi)
        matched_pred.add(pi)
        matches.append((gi, pi))

    return matches


def safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(
        self,
        truth_count: int,
        prediction_count: int,
        match_count: int,
    ) -> None:
        self.tp += match_count
        self.fp += prediction_count - match_count
        self.fn += truth_count - match_count

    def report(self) -> dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": safe_ratio(
                self.tp, self.tp + self.fp
            ),
            "recall": safe_ratio(
                self.tp, self.tp + self.fn
            ),
        }


def load_annotations(path: Path) -> dict[int, dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(document, dict):
        raise TypeError("Annotations must be a JSON object")

    frames = document.get("frames")

    if not isinstance(frames, list) or not frames:
        raise ValueError("Annotations must contain frames")

    lookup = {}
    identities = set()

    for frame in frames:
        if not isinstance(frame, dict):
            raise TypeError("Invalid annotation frame")

        frame_id = frame.get("frame_id")

        if (
            type(frame_id) is not int
            or frame_id < 0
            or frame_id in lookup
        ):
            raise ValueError("Invalid or duplicate frame_id")

        persons = frame.get("persons")
        objects = frame.get("objects")

        if not isinstance(persons, list) or not isinstance(objects, list):
            raise TypeError("Each frame requires persons and objects")

        frame_ids = set()

        for person in persons:
            if not isinstance(person, dict):
                raise TypeError("Invalid person annotation")

            identity = person.get("id")

            if not isinstance(identity, str) or not identity.strip():
                raise ValueError("Person identity must be nonempty")

            if identity in frame_ids:
                raise ValueError("Duplicate person identity in frame")

            frame_ids.add(identity)
            identities.add(identity)
            box_tuple(person.get("bbox"))

            points = person.get("keypoints", {})

            if not isinstance(points, dict):
                raise TypeError("keypoints must be an object")

            for name, point in points.items():
                if (
                    not isinstance(name, str)
                    or not isinstance(point, list)
                    or len(point) != 2
                    or any(
                        isinstance(v, bool)
                        or not isinstance(v, (int, float))
                        or not isfinite(v)
                        for v in point
                    )
                ):
                    raise ValueError("Invalid keypoint annotation")

        for obj in objects:
            if not isinstance(obj, dict):
                raise TypeError("Invalid object annotation")

            if not isinstance(obj.get("class_name"), str):
                raise TypeError("Object class_name is required")

            box_tuple(obj.get("bbox"))

        lookup[frame_id] = frame

    return lookup


class BenchmarkEvaluator:
    """
    Ground-truth comparison for one camera/video.

    Does not infer identity or behavior.
    Requires human-reviewed reference annotations.
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        pck_threshold: float = 0.1,
    ) -> None:
        if not 0 < iou_threshold <= 1:
            raise ValueError("Invalid IoU threshold")

        if not 0 < pck_threshold <= 1:
            raise ValueError("Invalid PCK threshold")

        self.iou_threshold = iou_threshold
        self.pck_threshold = pck_threshold

        self.person_counts = Counts()
        self.object_counts = Counts()
        self.frames = 0
        self.matched_with_track_id = 0
        self.id_switches = 0
        self.keypoints_correct = 0
        self.keypoints_total = 0

        self.previous_tracks: dict[str, tuple[int, int]] = {}

    def update(
        self,
        frame_id: int,
        result: Any,
        ground_truth: dict[str, Any],
    ) -> None:
        gt_persons = ground_truth["persons"]
        gt_objects = ground_truth["objects"]
        persons = list(result.persons)

        matches = match_boxes(
            [p["bbox"] for p in gt_persons],
            [p.bbox for p in persons],
            self.iou_threshold,
        )

        self.person_counts.add(
            len(gt_persons),
            len(persons),
            len(matches),
        )

        for gi, pi in matches:
            gt = gt_persons[gi]
            pred = persons[pi]
            identity = gt["id"]
            track_id = pred.track_id

            if track_id is not None:
                self.matched_with_track_id += 1

                previous = self.previous_tracks.get(identity)

                if (
                    previous is not None
                    and previous[0] == frame_id - 1
                    and previous[1] != track_id
                ):
                    self.id_switches += 1

                self.previous_tracks[identity] = (
                    frame_id,
                    track_id,
                )

            keypoints = {
                point.name: point
                for point in pred.keypoints
            }

            gx1, gy1, gx2, gy2 = box_tuple(gt["bbox"])
            tolerance = max(
                gx2 - gx1,
                gy2 - gy1,
            ) * self.pck_threshold

            for name, expected in gt.get("keypoints", {}).items():
                self.keypoints_total += 1
                actual = keypoints.get(name)

                if actual is None:
                    continue

                if getattr(actual, "confidence", 0) < 0.25:
                    continue

                distance = hypot(
                    actual.x - expected[0],
                    actual.y - expected[1],
                )

                if distance <= tolerance:
                    self.keypoints_correct += 1

        predicted_objects = list(result.objects)

        class_names = {
            obj["class_name"] for obj in gt_objects
        } | {
            obj.class_name for obj in predicted_objects
        }

        for class_name in class_names:
            gt_boxes = [
                obj["bbox"]
                for obj in gt_objects
                if obj["class_name"] == class_name
            ]

            pred_boxes = [
                obj.bbox
                for obj in predicted_objects
                if obj.class_name == class_name
            ]

            matches_for_class = match_boxes(
                gt_boxes,
                pred_boxes,
                self.iou_threshold,
            )

            self.object_counts.add(
                len(gt_boxes),
                len(pred_boxes),
                len(matches_for_class),
            )

        self.frames += 1

    def report(self) -> dict[str, Any]:
        return {
            "annotated_frames": self.frames,
            "persons": self.person_counts.report(),
            "objects": self.object_counts.report(),
            "track_id_coverage": safe_ratio(
                self.matched_with_track_id,
                self.person_counts.tp,
            ),
            "id_switches_consecutive_frames": self.id_switches,
            "pose_pck_at_0_1": safe_ratio(
                self.keypoints_correct,
                self.keypoints_total,
            ),
            "pose_labeled_keypoints": self.keypoints_total,
            "metric_definitions": {
                "matching_iou": self.iou_threshold,
                "pck_normalization": "max_gt_box_side",
                "pck_threshold": self.pck_threshold,
                "tracking": (
                    "ID changes for matched persons in "
                    "consecutive annotated frames only"
                ),
            },
        }

