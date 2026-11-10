from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class ROI:
    x1: int
    y1: int
    x2: int
    y2: int
    source: str
    track_id: int | None = None


@dataclass
class DetectionBatch:
    boxes: np.ndarray
    poses: np.ndarray
    scores: np.ndarray
    sources: dict[str, int]


def empty_batch():
    return DetectionBatch(
        np.empty((0, 4), dtype=np.float32),
        np.empty((0, 17, 3), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        {},
    )


class MultiStageROIPipeline:
    """
    Multi-stage YOLO Pose inference.

    Global detection runs every frame.
    Track ROIs recover temporarily weak detections.
    Desk ROIs are configurable and independent of employee count.
    Tiled recovery periodically scans the full image.
    """

    def __init__(
        self,
        model: Any,
        config_path: str | Path,
        *,
        device: str = "cuda:0",
        image_size: int = 1280,
        confidence: float = 0.25,
    ):
        self.model = model
        self.device = device
        self.image_size = image_size
        self.confidence = confidence

        with Path(config_path).open(
            "r", encoding="utf-8"
        ) as file:
            self.config = yaml.safe_load(file)

        self.frame_index = 0

        self.roi_scale = float(
            self.config["roi"]["upscale"]
        )
        self.padding = float(
            self.config["roi"]["track_padding"]
        )
        self.tile_every = int(
            self.config["tiles"]["every_n_frames"]
        )
        self.tile_rows = int(
            self.config["tiles"]["rows"]
        )
        self.tile_columns = int(
            self.config["tiles"]["columns"]
        )
        self.tile_overlap = float(
            self.config["tiles"]["overlap"]
        )
        self.nms_threshold = float(
            self.config["fusion"]["nms_threshold"]
        )
        self.max_rois = int(
            self.config["roi"]["max_track_rois"]
        )
        self.min_roi_size = int(
            self.config["roi"]["min_size"]
        )

        if self.roi_scale < 1:
            raise ValueError("ROI upscale must be >= 1")

        if self.tile_every < 1:
            raise ValueError("Tile interval must be >= 1")

        if self.tile_rows < 1 or self.tile_columns < 1:
            raise ValueError("Invalid tile grid")

        if not 0 <= self.tile_overlap < 0.8:
            raise ValueError("Invalid tile overlap")

        self.desk_rois = self._load_desk_rois()

    def _load_desk_rois(self):
        result = []

        for item in self.config.get("desk_rois", []):
            coords = np.asarray(
                item["xyxy"],
                dtype=np.float32,
            )

            if coords.shape != (4,):
                raise ValueError("Desk ROI requires four coordinates")

            if (
                not np.isfinite(coords).all()
                or np.any(coords < 0)
                or np.any(coords > 1)
                or coords[2] <= coords[0]
                or coords[3] <= coords[1]
            ):
                raise ValueError("Invalid normalized desk ROI")

            result.append(
                (str(item["name"]), coords)
            )

        return result

    @staticmethod
    def _clip_box(box, width, height):
        x1, y1, x2, y2 = map(float, box)

        x1 = max(0, min(width - 1, int(x1)))
        y1 = max(0, min(height - 1, int(y1)))
        x2 = max(0, min(width, int(x2)))
        y2 = max(0, min(height, int(y2)))

        if x2 <= x1 or y2 <= y1:
            return None

        return x1, y1, x2, y2

    def _make_track_rois(self, manager, width, height):
        rois = []

        for track_id, track in manager.tracks.items():
            if track.missed > manager.max_missed:
                continue

            state = track.state

            cx, cy, box_width, box_height = state[:4]

            # Increase search area with missed-frame count.
            expansion = self.padding * (
                1.0 + min(track.missed, 8) * 0.12
            )

            box_width = max(
                self.min_roi_size,
                float(box_width) * expansion,
            )
            box_height = max(
                self.min_roi_size,
                float(box_height) * expansion,
            )

            box = self._clip_box(
                (
                    cx - box_width / 2,
                    cy - box_height / 2,
                    cx + box_width / 2,
                    cy + box_height / 2,
                ),
                width,
                height,
            )

            if box is not None:
                rois.append(
                    ROI(*box, "track", int(track_id))
                )

        # Prioritize tracks with larger missed counts.
        rois.sort(
            key=lambda roi: (
                manager.tracks[roi.track_id].missed,
                manager.tracks[roi.track_id].age,
            ),
            reverse=True,
        )

        return rois[:self.max_rois]

    def _make_desk_rois(self, width, height):
        rois = []

        for name, coords in self.desk_rois:
            box = self._clip_box(
                (
                    coords[0] * width,
                    coords[1] * height,
                    coords[2] * width,
                    coords[3] * height,
                ),
                width,
                height,
            )

            if box is not None:
                rois.append(ROI(*box, name))

        return rois

    def _make_tiles(self, width, height):
        rois = []

        tile_width = int(
            width / (
                1 + (self.tile_columns - 1)
                * (1 - self.tile_overlap)
            )
        )

        tile_height = int(
            height / (
                1 + (self.tile_rows - 1)
                * (1 - self.tile_overlap)
            )
        )

        step_x = max(
            1,
            int(tile_width * (1 - self.tile_overlap)),
        )
        step_y = max(
            1,
            int(tile_height * (1 - self.tile_overlap)),
        )

        for row in range(self.tile_rows):
            for column in range(self.tile_columns):
                x1 = column * step_x
                y1 = row * step_y

                x2 = min(width, x1 + tile_width)
                y2 = min(height, y1 + tile_height)

                x1 = min(x1, max(0, width - 1))
                y1 = min(y1, max(0, height - 1))

                if x2 > x1 and y2 > y1:
                    rois.append(
                        ROI(x1, y1, x2, y2, "tile")
                    )

        return rois

    def _infer(self, image, roi=None):
        if roi is None:
            crop = image
            offset_x = 0
            offset_y = 0
        else:
            crop = image[
                roi.y1:roi.y2,
                roi.x1:roi.x2,
            ]

            offset_x = roi.x1
            offset_y = roi.y1

        if crop.size == 0:
            return empty_batch()

        original_height, original_width = crop.shape[:2]

        scale = 1.0

        if roi is not None:
            scale = min(
                self.roi_scale,
                2048 / max(original_width, original_height),
            )

        if scale > 1.0:
            crop = cv2.resize(
                crop,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )

        results = self.model.predict(
            source=crop,
            device=self.device,
            imgsz=self.image_size,
            conf=self.confidence,
            classes=[0],
            verbose=False,
        )

        result = results[0]
        boxes_object = result.boxes
        pose_object = result.keypoints

        if (
            boxes_object is None
            or pose_object is None
            or pose_object.data is None
            or len(boxes_object) == 0
        ):
            return empty_batch()

        boxes = boxes_object.xyxy.detach().cpu().numpy()
        scores = boxes_object.conf.detach().cpu().numpy()
        poses = pose_object.data.detach().cpu().numpy()

        if not (len(boxes) == len(scores) == len(poses)):
            raise RuntimeError("Detection and pose lengths differ")

        boxes = boxes.astype(np.float32)
        poses = poses.astype(np.float32)

        boxes /= scale
        poses[:, :, :2] /= scale

        boxes[:, [0, 2]] += offset_x
        boxes[:, [1, 3]] += offset_y

        poses[:, :, 0] += offset_x
        poses[:, :, 1] += offset_y

        return DetectionBatch(
            boxes,
            poses,
            scores.astype(np.float32),
            {},
        )

    def _fuse(self, batches):
        valid_boxes = []
        valid_poses = []
        valid_scores = []
        sources = {}

        for batch in batches:
            sources.update(batch.sources)

            for box, pose, score in zip(
                batch.boxes,
                batch.poses,
                batch.scores,
                strict=True,
            ):
                if (
                    not np.isfinite(box).all()
                    or not np.isfinite(pose).all()
                    or not np.isfinite(score)
                    or box[2] <= box[0]
                    or box[3] <= box[1]
                ):
                    continue

                valid_boxes.append(box)
                valid_poses.append(pose)
                valid_scores.append(float(score))

        if not valid_boxes:
            result = empty_batch()
            result.sources = sources
            return result

        boxes = np.asarray(valid_boxes, dtype=np.float32)
        poses = np.asarray(valid_poses, dtype=np.float32)
        scores = np.asarray(valid_scores, dtype=np.float32)

        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.confidence,
            self.nms_threshold,
        )

        if len(indices) == 0:
            return empty_batch()

        indices = np.asarray(indices).reshape(-1)

        result = DetectionBatch(
            boxes[indices],
            poses[indices],
            scores[indices],
            sources,
        )

        return result

    def process(self, frame, manager):
        if frame is None or frame.ndim != 3:
            raise ValueError("Expected a BGR image")

        height, width = frame.shape[:2]

        if width < 2 or height < 2:
            raise ValueError("Invalid frame dimensions")

        self.frame_index += 1

        batches = []

        # Stage 1: global detection on every frame.
        global_batch = self._infer(frame)
        global_batch.sources = {"global": len(global_batch.boxes)}
        batches.append(global_batch)

        # Stage 2: predicted track ROIs.
        track_rois = self._make_track_rois(
            manager, width, height
        )

        for roi in track_rois:
            batch = self._infer(frame, roi)
            batch.sources = {"track_roi": len(batch.boxes)}
            batches.append(batch)

        # Stage 3: configured desk ROIs.
        desk_rois = self._make_desk_rois(width, height)

        for roi in desk_rois:
            batch = self._infer(frame, roi)
            batch.sources = {"desk_roi": len(batch.boxes)}
            batches.append(batch)

        # Stage 4: periodic full-frame tiled recovery.
        if self.frame_index % self.tile_every == 0:
            for roi in self._make_tiles(width, height):
                batch = self._infer(frame, roi)
                batch.sources = {"tile": len(batch.boxes)}
                batches.append(batch)

        return self._fuse(batches)
