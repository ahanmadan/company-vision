
# Company Vision — Node 3.6 Benchmark Report

## Node Information

- Phase: 3
- Node: 3.6
- Title: Vision Core Accuracy and Performance Benchmark
- Branch: feature/phase3-node-3-6
- Status: IN_PROGRESS

## Environment

- Operating System: Windows
- Python: 3.12.10
- GPU: NVIDIA GeForce RTX 4070 SUPER
- CUDA: 13.0
- Torch: 2.11.0+cu130
- Pose Model: yolo26n-pose.pt
- Detection Model: yolo26n.pt
- Tracker: bytetrack.yaml
- Dependency lock: windows-py312-cu130.lock.txt

## Implemented Benchmark Features

- End-to-end VisionPipeline execution
- OverlayRenderer execution
- CUDA-synchronized latency measurement
- Warm-up exclusion from timing
- Mean and median latency
- P95 latency
- Processing-only FPS
- Peak allocated CUDA memory
- Person bounding-box matching
- Object class and bounding-box matching
- Person and object Precision/Recall
- Track ID coverage
- Consecutive-frame ID switch counting
- Optional pose PCK evaluation
- Machine-readable JSON reports

## Software Validation

| Test | Expected | Actual |
|---|---|---|
| pip check | PASS | PENDING |
| Node 3.6 unit tests | 12 passed | PENDING |
| Full project regression | 180 passed | PENDING |
| Ruff | PASS | PENDING |
| Node 3.4 GPU | PASS | PENDING |
| Node 3.5 GPU | PASS | PENDING |
| Synthetic benchmark | PASS | PENDING |
| Dependency lock | Verified | PENDING |
| GitHub synchronization | PASS | PENDING |

## Real-Video Evaluation

- Video: NOT PROVIDED
- Reference annotations: NOT PROVIDED
- Person Precision: NOT EVALUATED
- Person Recall: NOT EVALUATED
- Object Precision: NOT EVALUATED
- Object Recall: NOT EVALUATED
- Track ID Coverage: NOT EVALUATED
- ID Switches: NOT EVALUATED
- Pose PCK: NOT EVALUATED
- Real-video latency: NOT EVALUATED

## Known Limitations

- Synthetic input does not establish accuracy
- Benchmark FPS excludes video capture/network I/O
- Current pipeline runs detection more than once per frame
- ID switches are measured only on consecutive annotated frames
- No full HOTA/MOTA evaluation
- PCK requires manually labeled keypoints
- Real webcam and RTSP tests remain deferred

## Acceptance

- Software implementation: PENDING
- GPU benchmark: PENDING
- Real-video accuracy: PENDING
- Operational camera validation: DEFERRED
- Phase 3 overall: IN_PROGRESS

## Approval

- Last Approved Node: 3.5
- Current Node: 3.6
- Next Phase: 4
- Approval: PENDING



Company Vision v1.0

Phase 1: PASS
Phase 2: SOFTWARE COMPLETE / HARDWARE DEFERRED

Phase 3:
  Node 3.1: PASS
  Node 3.2: PASS
  Node 3.3: PASS
  Node 3.4: PASS
  Node 3.5: PASS
  Node 3.6: IN_PROGRESS

Last Approved Node: 3.5
Current Node: 3.6

Deferred:
  - Real webcam integration
  - Office RTSP integration
  - Actual camera recovery test

Benchmark Pending:
  - Real video quality evaluation
  - Precision and Recall
  - Track ID switches
  - Pose keypoint accuracy
  - Real-world performance baseline
