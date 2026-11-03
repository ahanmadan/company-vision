
import json
from types import SimpleNamespace

import pytest

from company_vision.benchmarking.evaluator import (
    BenchmarkEvaluator,
    Counts,
    box_tuple,
    iou,
    load_annotations,
    match_boxes,
    safe_ratio,
)


def person(
    track_id=1,
    bbox=(10, 10, 100, 100),
    keypoints=(),
):
    return SimpleNamespace(
        track_id=track_id,
        bbox=bbox,
        keypoints=keypoints,
    )


def obj(class_name="cell phone", bbox=(10, 10, 30, 30)):
    return SimpleNamespace(
        class_name=class_name,
        bbox=bbox,
    )


def gt_person(
    identity="A",
    bbox=(10, 10, 100, 100),
    keypoints=None,
):
    return {
        "id": identity,
        "bbox": list(bbox),
        "keypoints": keypoints or {},
    }


def truth(persons=None, objects=None):
    return {
        "persons": persons if persons is not None else [],
        "objects": objects if objects is not None else [],
    }


def result(persons=None, objects=None):
    return SimpleNamespace(
        persons=persons if persons is not None else [],
        objects=objects if objects is not None else [],
    )


def test_01_iou_identical():
    box = (0, 0, 10, 10)
    assert iou(box, box) == pytest.approx(1.0)


def test_02_iou_disjoint():
    assert iou(
        (0, 0, 10, 10),
        (20, 20, 30, 30),
    ) == 0


def test_03_invalid_box():
    with pytest.raises(ValueError):
        box_tuple((10, 10, 5, 5))


def test_04_one_to_one_matching():
    matches = match_boxes(
        [(0, 0, 10, 10)],
        [(0, 0, 10, 10), (0, 0, 10, 10)],
    )

    assert len(matches) == 1


def test_05_precision_recall():
    counts = Counts()
    counts.add(2, 3, 2)

    report = counts.report()

    assert report["tp"] == 2
    assert report["fp"] == 1
    assert report["fn"] == 0
    assert report["precision"] == pytest.approx(2 / 3, abs=0.0001)
    assert report["recall"] == 1


def test_06_unknown_denominator():
    assert safe_ratio(0, 0) is None


def test_07_person_true_positive():
    evaluator = BenchmarkEvaluator()

    evaluator.update(
        0,
        result(persons=[person()]),
        truth(persons=[gt_person()]),
    )

    report = evaluator.report()

    assert report["persons"]["tp"] == 1
    assert report["persons"]["precision"] == 1
    assert report["track_id_coverage"] == 1


def test_08_missed_person():
    evaluator = BenchmarkEvaluator()

    evaluator.update(
        0,
        result(),
        truth(persons=[gt_person()]),
    )

    report = evaluator.report()

    assert report["persons"]["fn"] == 1
    assert report["persons"]["recall"] == 0


def test_09_track_id_switch():
    evaluator = BenchmarkEvaluator()

    for frame_id, track_id in ((0, 1), (1, 1), (2, 7)):
        evaluator.update(
            frame_id,
            result(persons=[person(track_id=track_id)]),
            truth(persons=[gt_person()]),
        )

    assert evaluator.report()["id_switches_consecutive_frames"] == 1


def test_10_object_class_matching():
    evaluator = BenchmarkEvaluator()

    evaluator.update(
        0,
        result(objects=[obj("cell phone")]),
        truth(objects=[
            {
                "class_name": "cell phone",
                "bbox": [10, 10, 30, 30],
            }
        ]),
    )

    assert evaluator.report()["objects"]["tp"] == 1


def test_11_pose_pck():
    evaluator = BenchmarkEvaluator()

    point = SimpleNamespace(
        name="nose",
        x=20.0,
        y=20.0,
        confidence=0.9,
    )

    evaluator.update(
        0,
        result(persons=[person(keypoints=[point])]),
        truth(persons=[
            gt_person(keypoints={"nose": [20, 20]})
        ]),
    )

    assert evaluator.report()["pose_pck_at_0_1"] == 1


def test_12_annotations_validation(tmp_path):
    path = tmp_path / "labels.json"

    path.write_text(
        json.dumps({
            "frames": [
                {
                    "frame_id": 0,
                    "persons": [
                        gt_person()
                    ],
                    "objects": [],
                }
            ]
        }),
        encoding="utf-8",
    )

    annotations = load_annotations(path)

    assert 0 in annotations
    assert annotations[0]["persons"][0]["id"] == "A"
