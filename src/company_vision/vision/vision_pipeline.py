
from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from company_vision.core.schemas import (
    BoundingBox,
    FramePacket,
    FrameResult,
    Keypoint,
    ObjectObservation,
    PersonObservation,
)
from company_vision.vision.object_detector import (
    DetectionFrameResult,
    ObjectDetector,
)
from company_vision.vision.person_tracker import (
    PersonTracker,
    TrackingFrameResult,
)
from company_vision.vision.pose_estimator import (
    PoseEstimator,
    PoseFrameResult,
    PosePerson,
)


class VisionPipelineError(RuntimeError):
    """Vision pipeline integration failed."""


class PoseEngine(Protocol):
    def predict(self, packet: FramePacket) -> PoseFrameResult: ...


class DetectionEngine(Protocol):
    def predict(self, packet: FramePacket) -> DetectionFrameResult: ...


class TrackingEngine(Protocol):
    def track(self, packet: FramePacket) -> TrackingFrameResult: ...


@dataclass(frozen=True)
class PipelineStats:
    processed_frames: int
    total_persons: int
    total_objects: int
    partial_frames: int


def calculate_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Calculate intersection over union for XYXY boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def bounded_box(
    coords: tuple[float, float, float, float],
    width: int,
    height: int,
) -> BoundingBox:
    x1, y1, x2, y2 = coords

    if not all(isfinite(value) for value in coords):
        raise VisionPipelineError("Non-finite bounding box")

    x1 = max(0.0, min(float(width), float(x1)))
    x2 = max(0.0, min(float(width), float(x2)))
    y1 = max(0.0, min(float(height), float(y1)))
    y2 = max(0.0, min(float(height), float(y2)))

    if x2 <= x1 or y2 <= y1:
        raise VisionPipelineError("Invalid bounding box")

    return BoundingBox(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )


def convert_keypoints(
    pose: PosePerson | None,
    width: int,
    height: int,
) -> list[Keypoint]:
    if pose is None:
        return []

    converted = []

    for point in pose.keypoints:
        if not isfinite(point.x) or not isfinite(point.y):
            raise VisionPipelineError("Non-finite keypoint")

        confidence = (
            point.confidence
            if point.confidence is not None
            else 0.0
        )

        if not isfinite(confidence) or not 0 <= confidence <= 1:
            raise VisionPipelineError(
                "Invalid keypoint confidence"
            )

        converted.append(
            Keypoint(
                name=point.name,
                x=max(0.0, min(float(width), point.x)),
                y=max(0.0, min(float(height), point.y)),
                confidence=confidence,
            )
        )

    return converted


class VisionPipeline:
    """
    Combine Pose, Detection and Tracking into FrameResult.

    The tracker determines temporary person IDs.
    Pose-to-track association uses one-to-one IoU matching.
    """

    def __init__(
        self,
        pose_engine: PoseEngine,
        detection_engine: DetectionEngine,
        tracking_engine: TrackingEngine,
        *,
        match_iou: float = 0.3,
    ) -> None:
        if (
            isinstance(match_iou, bool)
            or not isinstance(match_iou, (float, int))
            or not isfinite(match_iou)
            or not 0 < match_iou <= 1
        ):
            raise ValueError("match_iou must be in (0, 1]")

        self.pose_engine = pose_engine
        self.detection_engine = detection_engine
        self.tracking_engine = tracking_engine
        self.match_iou = float(match_iou)

        self._camera_id: str | None = None
        self._last_frame_id: int | None = None

        self._processed_frames = 0
        self._total_persons = 0
        self._total_objects = 0
        self._partial_frames = 0

    @classmethod
    def from_models(
        cls,
        *,
        pose_model: str = "yolo26x-pose.pt",
        detection_model: str = "yolo26x.pt",
        device: str = "cuda:0",
        match_iou: float = 0.3,
    ) -> "VisionPipeline":
        return cls(
            pose_engine=PoseEstimator(
                model_path=pose_model,
                device=device,
            ),
            detection_engine=ObjectDetector(
                model_path=detection_model,
                device=device,
            ),
            tracking_engine=PersonTracker(
                model_path=detection_model,
                device=device,
                tracker_config="bytetrack.yaml",
            ),
            match_iou=match_iou,
        )

    def _validate_packet(self, packet: FramePacket) -> None:
        if not isinstance(packet, FramePacket):
            raise TypeError("packet must be a FramePacket")

        if (
            self._camera_id is not None
            and packet.camera_id != self._camera_id
        ):
            raise VisionPipelineError(
                "Pipeline received another camera"
            )

        if (
            self._last_frame_id is not None
            and packet.frame_id <= self._last_frame_id
        ):
            raise VisionPipelineError(
                "Frame IDs must increase strictly"
            )

    @staticmethod
    def _validate_identity(
        packet: FramePacket,
        *outputs: object,
    ) -> None:
        for output in outputs:
            if getattr(output, "packet", None) is not packet:
                raise VisionPipelineError(
                    "Engine results belong to different frames"
                )

    def _associate(
        self,
        poses: PoseFrameResult,
        tracks: TrackingFrameResult,
    ) -> tuple[list[PersonObservation], bool]:
        width = poses.packet.width
        height = poses.packet.height

        candidates: list[tuple[float, int, int]] = []

        for track_index, tracked in enumerate(tracks.persons):
            for pose_index, pose in enumerate(poses.people):
                overlap = calculate_iou(tracked.bbox, pose.bbox)

                if overlap >= self.match_iou:
                    candidates.append(
                        (overlap, track_index, pose_index)
                    )

        # Highest-overlap matches first, one-to-one.
        candidates.sort(key=lambda item: -item[0])

        matches: dict[int, int] = {}
        used_poses: set[int] = set()

        for _overlap, track_index, pose_index in candidates:
            if track_index in matches or pose_index in used_poses:
                continue

            matches[track_index] = pose_index
            used_poses.add(pose_index)

        persons: list[PersonObservation] = []

        for track_index, tracked in enumerate(tracks.persons):
            pose = (
                poses.people[matches[track_index]]
                if track_index in matches
                else None
            )

            persons.append(
                PersonObservation(
                    track_id=tracked.track_id,
                    bbox=bounded_box(
                        tracked.bbox,
                        width,
                        height,
                    ),
                    confidence=tracked.confidence,
                    keypoints=convert_keypoints(
                        pose,
                        width,
                        height,
                    ),
                )
            )

        # Pose detections without a matching Track ID
        # are retained as untracked persons.
        for pose_index, pose in enumerate(poses.people):
            if pose_index in used_poses:
                continue

            persons.append(
                PersonObservation(
                    track_id=None,
                    bbox=bounded_box(
                        pose.bbox,
                        width,
                        height,
                    ),
                    confidence=pose.confidence,
                    keypoints=convert_keypoints(
                        pose,
                        width,
                        height,
                    ),
                )
            )

        incomplete = (
            len(matches) != len(tracks.persons)
            or len(used_poses) != len(poses.people)
        )

        return persons, incomplete

    @staticmethod
    def _convert_objects(
        detection: DetectionFrameResult,
    ) -> list[ObjectObservation]:
        width = detection.packet.width
        height = detection.packet.height

        objects: list[ObjectObservation] = []

        for obj in detection.objects:
            # Human detections are represented in persons.
            if obj.class_id == 0:
                continue

            objects.append(
                ObjectObservation(
                    class_id=obj.class_id,
                    class_name=obj.class_name,
                    bbox=bounded_box(
                        obj.bbox,
                        width,
                        height,
                    ),
                    confidence=obj.confidence,
                )
            )

        return objects

    def process(self, packet: FramePacket) -> FrameResult:
        self._validate_packet(packet)

        # Run independent inference before stateful tracking.
        pose = self.pose_engine.predict(packet)
        detection = self.detection_engine.predict(packet)

        self._validate_identity(packet, pose, detection)

        # Stateful ByteTrack is invoked last.
        tracking = self.tracking_engine.track(packet)

        self._validate_identity(packet, tracking)

        persons, incomplete = self._associate(pose, tracking)
        objects = self._convert_objects(detection)

        status = "partial" if incomplete else "ok"

        result = FrameResult.from_packet(
            packet,
            persons=persons,
            objects=objects,
            status=status,
        )

        # Commit pipeline counters only after validation succeeds.
        self._camera_id = packet.camera_id
        self._last_frame_id = packet.frame_id

        self._processed_frames += 1
        self._total_persons += len(persons)
        self._total_objects += len(objects)

        if incomplete:
            self._partial_frames += 1

        return result

    def stats(self) -> PipelineStats:
        return PipelineStats(
            processed_frames=self._processed_frames,
            total_persons=self._total_persons,
            total_objects=self._total_objects,
            partial_frames=self._partial_frames,
        )
