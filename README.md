# Company Vision

Modular Python computer-vision platform for
multi-person tracking and observable activity analysis.

## Technology

- Python 3.12
- OpenCV
- YOLO Detection
- YOLO Pose
- ByteTrack
- FastAPI (later phase)
- PostgreSQL (later phase)

## Architecture

Camera -> Frame Reader -> YOLO Pose + Detection
-> ByteTrack -> FrameResult -> Behavior Plugins

## Documentation

- docs/PROJECT_SCOPE.md
- docs/ARCHITECTURE.md
- docs/WORKFLOW.md
- docs/DECISIONS.md
- docs/PROGRESS.md
- docs/ENVIRONMENT_REPORT.md

## Current status

Phase 1 - Project preparation.
No models or runtime dependencies installed yet.

## Privacy

Do not commit surveillance video, credentials,
private recordings, or trained model files.
