# COMPANY VISION
# PHASE 4 - FINAL VALIDATION AND RELEASE

Release date: 2026-11-11 13:16:05
Branch: main
Baseline commit: 3095d31

## 1. Release scope

This release consolidates the implemented and validated
Phase 4 nodes 4.1 through 4.6.

The SYNC processing pipeline is the official reference path.

The project has no hard-coded maximum number of people.
One-person live acceptance is the current acceptance baseline.
Multi-person stress validation remains a future test.

## 2. Node evidence

The following evidence files are preserved:

- phase4_primary_evidence_20261111_131006.csv
- phase4_evidence_20261111_130916.csv
- node45_live_benchmark_20261111_130010.csv

Detailed historical node reports and backup files are preserved
in the external release archive.

No unsupported measurements are substituted for missing data.

## 3. GPU validation

Hardware:
- NVIDIA RTX 4070 SUPER
- PyTorch 2.11.0+cu130
- CUDA 13.0
- Ultralytics 8.4.202

FP16:
- Inference: 16.63 ms
- Peak GPU memory: 301.08 MB
- Backend FP16: True

FP32:
- Inference: 31.16 ms
- Peak GPU memory: 680.99 MB
- Backend FP16: False

These are the recorded precision test measurements.
Equivalent pose accuracy between FP16 and FP32 has not
been established by this benchmark alone.

## 4. SYNC / ASYNC benchmark

SYNC:
- Repetition 1: 24.953 FPS
- Repetition 4: 24.967 FPS

ASYNC:
- Repetition 2: 24.633 FPS
- Repetition 3: 24.567 FPS

SYNC is retained as the production reference.

ASYNC is experimental and is not selected as the reference.

## 5. Live camera validation

The Dahua live camera acceptance test was performed.

The test included a primary visible person and a seated
guest detected in the camera view.

The detection of the seated guest is not a pipeline failure.

The test does not establish a maximum supported person count.

## 6. Automated regression

Recorded regression results:

- Python compile: PASS
- Ruff: PASS
- Pytest: 245 passed
- Pip check: PASS
- CUDA availability: PASS
- GPU computation: PASS
- Git diff check: PASS

Regression log:
D:\Projects\company-vision\reports\phase4_final_audit\final_regression_20261111_131300.log

## 7. Second system documentation

The clone command now uses main.

The documented live runner uses the SYNC reference path.

## 8. Outstanding scope

This release does not claim completion of future dataset
training or real-camera fine-tuning unless separately
validated and documented.

The behavior recognition engine is not included in this release.

## 9. Archive

Archive: D:\Projects\company-vision-archive\final_release_20261111_131604.zip

Archived untracked files: 28

SHA256 manifest:
SHA256_MANIFEST.csv

## 10. Git release

Target branch: main
Remote: origin
All other remote branches are expected to be absent.

This document records the validated release baseline.
