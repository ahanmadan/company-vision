# Node 5.8.1 - PhoneUsagePlugin

Status: IMPLEMENTED / AWAITING ACCEPTANCE

## Responsibility

Infer phone-use evidence per tracked person from:
- Explicit phone object detections.
- Person bbox.
- Valid elbow or wrist keypoints.
- Spatial proximity.
- Object detection availability and observation quality.

## Input

BehaviorPluginContext (Node 5.4).

## Output

PluginResult with plugin_id=phone_usage.

Per-track observations:
- track_id
- status: ok / unknown
- metrics.score: 0..1 when supported
- confidence when supported
- evidence and reason

## Rules

- Missing object detection => unknown.
- Ineligible / predicted track => unknown.
- No reliable person-phone association => no positive claim.
- One phone may not be positively assigned to multiple people.
- Missing or ambiguous arm geometry => unknown where relevant.
- Positive evidence is not a confirmed behavior event.

## Constraints

This plugin does not:
- Run YOLO or Object Detection.
- Read video or Dahua streams.
- Change upstream 9-point tracking.
- Implement temporal persistence.
- Confirm or terminate behavior events.
- Guarantee real-world phone-use accuracy.

## Next

Node 5.9 will connect upstream object detection, tracking,
temporal state, plugins, rules, events and output formatting.

Calibration and real-video acceptance remain pending.