# Node 5.9.5 - Operational Behavior Output

## Purpose
Convert RuleEventBatch into canonical BehaviorResult objects
and renderer-independent overlay instructions.

## Output types
- BehaviorResult (existing versioned project contract)
- PersonOverlay
- OperationalFrameOutput

## Status semantics
- confirmed: a previously verified event is confirmed
- active/pending: a positive event awaits confirmation
- unknown: evidence is insufficient or track is ineligible
- no_event: explicit rule evaluations have no positive event
- ended: an event ended during this frame

No status is a finding of misconduct, productivity or intent.

## Rendering
The output supplies:
- camera_id and session_id
- frame_id and track_id
- pixel-coordinate bbox
- display label
- confirmed behavior evidence labels
- visibility flag

Predicted tracks are never rendered as observed people.
This node does not itself draw any frames.

## Event and evidence integrity
- Four plugin observations are retained.
- Rule evaluations and event transitions are serialized.
- Old-session transitions are frame-level records.
- No arbitrary person count limit.
- Results are JSON-compatible.
- No database dependency.

## Next
Node 5.10: Real-video validation, GPU performance
measurement, accuracy analysis and calibration.

The final phase acceptance must include tests on real
Dahua video and must not rely on unit tests alone.