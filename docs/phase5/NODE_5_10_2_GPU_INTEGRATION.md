# Node 5.10.2 - Dahua GPU Behavior Integration

## Camera
Dahua physical Channel 2 = SDK index 1.

## Existing components reused
- NetSDK decoder from Phase 4
- YOLO26x-Pose on CUDA
- StablePoseManager
- OneEuro pose filtering
- TrackLifecycleGuard
- PoseTrackingBridge
- Four behavioral plugins
- RuleEngine and EventManager
- OperationalFrameOutput

## Execution
Live video runs until Q is pressed.
Actual GPU inference and tracking data feed the
behavior pipeline.

## Interpretation
The object detector is not run in this node.
Unknown evidence remains unknown.
Behavior labels are not productivity judgments.

## Acceptance
- Dedicated tests
- Full project regression
- Ruff and dependency checks
- Real Dahua GPU execution
- At least one integrated processed frame
- Explicit success marker
- GitHub synchronization

## Next
5.10.3: Behavioral accuracy validation.