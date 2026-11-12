# Node 5.8.4 - GatheringPlugin

Status: IMPLEMENTED / PENDING ACCEPTANCE

## Purpose
Detect sustained observable proximity of a configurable
minimum number of tracked persons.

## Input
BehaviorPluginContext containing FrameFeatures
and TemporalTrackHistory.

## Output
PluginResult with plugin_id="gathering".

Each track observation contains:
- status
- score if evidence is sufficient
- confidence
- group member track IDs
- duration and reason

## Method
- Use bottom-center bbox anchors.
- Normalize pair distances by apparent person height.
- Reject excessive apparent-height disparities.
- Require complete-link proximity among group members.
- Verify temporal history continuity.
- Evaluate synchronized group snapshots.
- Avoid positive claims for temporarily passing people.

## Important limitations
- Image geometry is not calibrated ground-plane distance.
- Perspective, occlusion and ID switches can affect results.
- Nearness is not proof of conversation or misconduct.
- Timing thresholds are initial engineering assumptions.
- Behavior accuracy must be validated on real office videos.

## Boundaries
No changes to YOLO, camera input, tracking, or
previously approved behavior modules.

Node 5.9: behavior pipeline integration.
Node 5.10: real-data validation and calibration.