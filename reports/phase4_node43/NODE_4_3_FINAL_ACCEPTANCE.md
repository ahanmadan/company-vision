# Node 4.3 - Final Acceptance Report

## Environment
- GPU: NVIDIA GeForce RTX 4070 SUPER
- CUDA: 13.0
- PyTorch: 2.11.0+cu130
- Ultralytics: 8.4.202

## FP16 validation
- Backend FP16: True
- Mean latency: 18.395 ms
- P95 latency: 20.342 ms
- Peak GPU memory: 301.51 MB
- Repeated runs: 5

## FP32 validation
- Backend FP16: False
- Mean latency: 32.313 ms
- P95 latency: 34.185 ms
- Peak GPU memory: 680.84 MB
- Repeated runs: 5

## Comparison
- Latency improvement: 43.07%
- Memory reduction: 55.71%

## Runner audit
- scripts/run_dahua_live_pose.py: EXISTS
  - quantize references: 1
  - half references: 0
- scripts/run_dahua_live_pose_node45.py: EXISTS
  - quantize references: 1
  - half references: 0
- scripts/benchmark_node45_live.py: EXISTS
  - quantize references: 1
  - half references: 0
- src/company_vision/vision/async_pose_pipeline.py: EXISTS
  - quantize references: 1
  - half references: 0

## Automated validation
- Python compileall: previously PASS
- Pytest: previously 245 PASS
- pip check: previously PASS

## Scope and limitations
- FP16 and FP32 backend execution were verified.
- Precision measurements used a synthetic black frame.
- Pose accuracy equivalence on real camera footage is NOT established by this test.
- Multi-person scalability is not evaluated here.
- No changes to approved tracking or keypoint filters are authorized by this report.

## Acceptance decision
GPU precision execution: PASS.
Full real-camera pose accuracy acceptance: PENDING.
Node 4.3 performance evidence: RECORDED.
