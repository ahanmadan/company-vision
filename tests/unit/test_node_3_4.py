
from types import SimpleNamespace

import numpy as np
import pytest

from company_vision.core.schemas import FramePacket, current_utc
from company_vision.vision.object_detector import (
    DetectedObject,
    DetectionFrameResult,
)
from company_vision.vision.person_tracker import (
    TrackedPerson,
    TrackingFrameResult,
)
from company_vision.vision.pose_estimator import (
    COCO_KEYPOINT_NAMES,
    PoseFrameResult,
    PoseKeypoint,
    PosePerson,
)
from company_vision.vision.vision_pipeline import (
    VisionPipeline,
    VisionPipelineError,
    calculate_iou,
)


def make_packet(frame_id=0, camera_id="cam_01"):
    return FramePacket(
        camera_id=camera_id,
        frame_id=frame_id,
        timestamp=current_utc(),
        image=np.zeros((240, 320, 3), dtype=np.uint8),
    )


def make_pose(bbox=(10, 20, 100, 200)):
    return PosePerson(
        bbox=bbox,
        confidence=0.9,
        keypoints=tuple(
            PoseKeypoint(
                name=name,
                x=30.0,
                y=40.0,
                confidence=0.85,
            )
            for name in COCO_KEYPOINT_NAMES
        ),
    )


def make_track(
    track_id=7,
    bbox=(10, 20, 100, 200),
):
    return TrackedPerson(
        track_id=track_id,
        bbox=bbox,
        confidence=0.92,
    )


def make_object(
    class_id=67,
    class_name="cell phone",
):
    return DetectedObject(
        class_id=class_id,
        class_name=class_name,
        bbox=(120, 40, 160, 100),
        confidence=0.8,
    )


class FakePoseEngine:
    def __init__(self, people):
        self.people = tuple(people)

    def predict(self, packet):
        return PoseFrameResult(
            packet=packet,
            people=self.people,
            inference_ms=1.0,
        )


class FakeDetectionEngine:
    def __init__(self, objects):
        self.objects = tuple(objects)

    def predict(self, packet):
        return DetectionFrameResult(
            packet=packet,
            objects=self.objects,
            inference_ms=1.0,
        )


class FakeTrackingEngine:
    def __init__(self, persons):
        self.persons = tuple(persons)

    def track(self, packet):
        return TrackingFrameResult(
            packet=packet,
            persons=self.persons,
            inference_ms=1.0,
        )


def make_pipeline(
    poses=None,
    tracks=None,
    objects=None,
    **kwargs,
):
    if poses is None:
        poses = [make_pose()]

    if tracks is None:
        tracks = [make_track()]

    if objects is None:
        objects = [make_object()]

    return VisionPipeline(
        pose_engine=FakePoseEngine(poses),
        detection_engine=FakeDetectionEngine(objects),
        tracking_engine=FakeTrackingEngine(tracks),
        **kwargs,
    )


def test_01_iou_identical():
    box = (10, 20, 100, 200)
    assert calculate_iou(box, box) == pytest.approx(1)


def test_02_iou_disjoint():
    assert calculate_iou(
        (0, 0, 10, 10),
        (20, 20, 30, 30),
    ) == 0


def test_03_matched_person():
    output = make_pipeline().process(make_packet())

    assert len(output.persons) == 1
    assert output.persons[0].track_id == 7
    assert len(output.persons[0].keypoints) == 17
    assert output.status == "ok"


def test_04_object_conversion():
    output = make_pipeline().process(make_packet())

    assert len(output.objects) == 1
    assert output.objects[0].class_id == 67
    assert output.objects[0].class_name == "cell phone"


def test_05_remove_duplicate_person_class():
    pipeline = make_pipeline(
        objects=[
            make_object(0, "person"),
            make_object(67, "cell phone"),
        ]
    )

    output = pipeline.process(make_packet())

    assert len(output.objects) == 1
    assert output.objects[0].class_id == 67


def test_06_empty_frame():
    pipeline = make_pipeline(
        poses=[],
        tracks=[],
        objects=[],
    )

    output = pipeline.process(make_packet())

    assert output.status == "ok"
    assert output.persons == []
    assert output.objects == []


def test_07_pose_without_track():
    pipeline = make_pipeline(
        poses=[make_pose()],
        tracks=[],
    )

    output = pipeline.process(make_packet())

    assert output.status == "partial"
    assert len(output.persons) == 1
    assert output.persons[0].track_id is None


def test_08_track_without_pose():
    pipeline = make_pipeline(
        poses=[],
        tracks=[make_track()],
    )

    output = pipeline.process(make_packet())

    assert output.status == "partial"
    assert output.persons[0].track_id == 7
    assert output.persons[0].keypoints == []


def test_09_two_persons():
    pipeline = make_pipeline(
        poses=[
            make_pose((10, 20, 100, 200)),
            make_pose((150, 20, 240, 200)),
        ],
        tracks=[
            make_track(1, (10, 20, 100, 200)),
            make_track(2, (150, 20, 240, 200)),
        ],
    )

    output = pipeline.process(make_packet())

    assert len(output.persons) == 2
    assert [p.track_id for p in output.persons] == [1, 2]
    assert output.status == "ok"


def test_10_no_matching_overlap():
    pipeline = make_pipeline(
        poses=[make_pose((10, 20, 100, 200))],
        tracks=[
            make_track(9, (180, 20, 260, 200))
        ],
    )

    output = pipeline.process(make_packet())

    assert output.status == "partial"
    assert len(output.persons) == 2
    assert output.persons[0].track_id == 9
    assert output.persons[1].track_id is None


def test_11_frame_metadata():
    packet = make_packet(frame_id=15)
    output = make_pipeline().process(packet)

    assert output.camera_id == packet.camera_id
    assert output.frame_id == 15
    assert output.width == 320
    assert output.height == 240
    assert output.timestamp == packet.timestamp


def test_12_frame_order_rejected():
    pipeline = make_pipeline()
    pipeline.process(make_packet(5))

    with pytest.raises(VisionPipelineError):
        pipeline.process(make_packet(4))


def test_13_camera_change_rejected():
    pipeline = make_pipeline()
    pipeline.process(make_packet(0, "cam_a"))

    with pytest.raises(VisionPipelineError):
        pipeline.process(make_packet(1, "cam_b"))


def test_14_invalid_match_threshold():
    with pytest.raises(ValueError):
        make_pipeline(match_iou=0)


def test_15_pipeline_stats():
    pipeline = make_pipeline()

    pipeline.process(make_packet(1))
    pipeline.process(make_packet(2))

    stats = pipeline.stats()

    assert stats.processed_frames == 2
    assert stats.total_persons == 2
    assert stats.total_objects == 2
    assert stats.partial_frames == 0


def test_16_mismatched_frame_identity():
    other = make_packet(99)

    class WrongFramePose:
        def predict(self, packet):
            return SimpleNamespace(
                packet=other,
                people=(),
            )

    pipeline = VisionPipeline(
        pose_engine=WrongFramePose(),
        detection_engine=FakeDetectionEngine([]),
        tracking_engine=FakeTrackingEngine([]),
    )

    with pytest.raises(VisionPipelineError):
        pipeline.process(make_packet())
