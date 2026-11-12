# Node 5.9.2 - Temporal Behavior Context Pipeline

## Status
IMPLEMENTED - AUTOMATED ACCEPTANCE PENDING

## Responsibility
Connect the existing Node 5.9.1 PoseTrackingBridge
to TemporalStateManager and BehaviorPluginContext.

## Inputs
- BehaviorFrame, or
- PoseTrackingBridge.build-compatible arguments.

## Outputs
BehaviorPluginContext containing:
- current FrameFeatures,
- bounded current-track temporal histories.

## Lifecycle
- Each identity is scoped by camera_id/session_id/track_id.
- Session changes reset only the relevant camera histories.
- Explicit ended/lost/removed/expired signals remove histories.
- Predicted and unconfirmed observations break eligible histories.
- Temporarily absent tracks are not assumed to be terminated.
- Current plugin context excludes absent track histories.

## Validation
- Duplicate/out-of-order frame IDs rejected.
- UTC and monotonic timestamp regressions rejected.
- Terminal/current-track conflicts rejected.
- No partial commits on invalid temporal updates.
- Reset permits a new timeline.
- No fixed person count.

## Limitations
- This module does not run YOLO, Dahua or GPU inference.
- It does not classify behaviors.
- It does not create rule evaluations or events.
- Source must provide reliable session identities and lifecycle.
- Missing terminal signals may retain dormant history.
- Real-camera acceptance belongs to Node 5.10.

## Next
5.9.3: execute the four behavior plugins.
5.9.4: rules and event lifecycle.
5.9.5: operational output formatting.