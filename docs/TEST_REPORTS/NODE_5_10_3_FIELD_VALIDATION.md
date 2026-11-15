# Node 5.10.3 — Field Validation Checkpoint

Status: FAIL / SIDE-PROFILE FIX REQUIRED

## Environment
- Camera: Dahua, physical channel 2
- GPU: NVIDIA RTX 4070 SUPER
- Pose model: YOLO26x-Pose
- Inference mode: SYNC
- Diagnostic object detection: DISABLED
- Stillness image threshold: 20 seconds

## Implemented
- Stable person tracking and pose visualization
- Per-person stillness timer
- Clean bounding-box crop at 20 seconds
- One capture per continuous stillness episode
- Local image output: project-root/img/

## Field validation
1. Complete inactivity: PASS
2. Motion after 10 seconds: PASS
3. Head movement: PASS
4. Hand and shoulder movement: PASS
5. Duplicate image prevention: PASS
6. Two inactivity episodes: PASS
7. Multiple persons: PASS
8. Person leaving camera view: PASS
9. Noise and small movements: FAIL

## Open defects
- Third person in side profile loses reliable detection
  after approximately 5 seconds.
- First person may retain a stillness timer near
  90 seconds despite subtle body movement.

## Acceptance
- Eight of nine scenarios passed in manual field testing.
- This is not a statistical accuracy measurement.
- Node 5.10.3 is NOT accepted.
- Existing successful behavior must remain unchanged.
- Side-profile and subtle-motion accuracy require correction.

## Privacy
- Do not commit camera recordings, cropped employee
  photographs, camera credentials, or private artifacts.
