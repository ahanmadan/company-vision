# Node 5.8.3 — InactivityPlugin

## Status
IMPLEMENTED / PENDING TEST VALIDATION

## Purpose
Identify sustained, visually observable upper-body
stillness using per-track temporal Pose features.

This is not a sleep diagnosis or a productivity assessment.

## Inputs
BehaviorPluginContext
FrameFeatures
TemporalTrackHistory

## Outputs
PluginResult / plugin_id=inactivity

Per person:
- track_id
- status
- metrics.score
- confidence
- evidence

## Rules
- Observed and eligible tracks only.
- Time-based duration, not fixed frame counts.
- Minimum three temporally valid observations.
- Require visible shoulders and an arm landmark.
- Check upper-body geometry and bounding-box movement.
- Reject broken histories and inadequate evidence.
- Describe lowered-head posture without claiming sleep.

## Thresholds
Thresholds in InactivityConfig are initial engineering
assumptions, not field-calibrated values.

## Limitations
- Pose cannot establish whether someone is asleep.
- Stillness does not imply poor performance.
- Camera motion and occlusion may affect measurements.
- Missing evidence must remain unknown.
- Frame-level stillness is not a confirmed event.

## Next steps
Node 5.9: integrate detection into behavior pipeline.
Node 5.10: evaluate real camera evidence and accuracy.