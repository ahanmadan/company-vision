# Node 5.9.1 - Pose / Tracking Behavior Bridge

Status: Implementation pending automated acceptance.

## Existing upstream
- YOLO Pose generates COCO 17 points.
- StablePoseManager uses the existing filtered nine-point contract.
- TrackLifecycleGuard identifies behavior-eligible observed tracks.

## Implemented bridge
- Converts track dictionaries into BehaviorFrame.
- Preserves stable IDs, nine filtered points, raw points and weights.
- Allows independently matched supplemental wrist observations.
- Converts already computed object detector results.
- Preserves object detection status: not_run vs available.
- Produces FrameFeatures using the existing FeatureExtractor.

## Safety and accuracy boundaries
- Does not run a new YOLO inference.
- Does not associate raw COCO wrists with arbitrary track IDs.
- Does not use predicted tracks as behavior evidence.
- Does not fabricate phones, monitors or keyboards.
- Does not classify behaviors or produce events.
- Doesn't change prior Tracking/Pose modules.

## Integration contract
Use bridge.build(...) to get BehaviorFrame.
Use bridge.extract(...) to get FrameFeatures.

The caller must supply:
- observed tracked output from existing tracker/guard,
- reliable camera and session identity,
- UTC timestamp and monotonic timestamp,
- frame dimensions,
- optional independently detected objects.

Node 5.9.2 will connect temporal histories.
Node 5.9.3 will invoke all behavior plugins.
Node 5.9.4 will connect rules and event lifecycle.
Node 5.9.5 will format operational display results.

Real-camera acceptance is deferred to Node 5.10.