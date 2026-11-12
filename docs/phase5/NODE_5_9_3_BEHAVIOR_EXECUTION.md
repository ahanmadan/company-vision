# Node 5.9.3 - Unified Behavior Plugin Execution

## Responsibility

Execute four independently implemented behavior detectors
on one shared BehaviorPluginContext.

## Required plugins

1. phone_usage
2. computer_usage
3. inactivity
4. gathering

## Inputs

- BehaviorFrame
- Existing Node 5.9.1 tracking bridge parameters
- BehaviorPluginContext (already prepared)

## Output

BehaviorExecutionBatch:

- context
- results: tuple[PluginResult, ...]
- per_track: track_id -> plugin_id -> observation

## Guarantees

- Existing behavior detector logic is unchanged.
- Every current tracked person receives four observations.
- Duplicate or missing track results are rejected.
- Unknown results are retained and not promoted to positive.
- No hard-coded number of workers.
- Observed, predicted and ineligible states are preserved.
- One shared frame and history context is passed to all plugins.
- Plug-in execution order is deterministic.
- Missing/disabled plugins fail explicitly.

## Scope

This node does not perform:
- live Dahua capture,
- YOLO inference,
- conflict resolution between behaviors,
- employee productivity assessment,
- rule evaluation,
- event recording,
- dashboard or video overlay.

Plugin failure after temporal processing raises an error.
The temporal input may already be committed. Do not replay
the same frame without an explicit pipeline reset or a new
frame sequence.

## Subsequent nodes

5.9.4: rule evaluation and event lifecycle management.
5.9.5: operational status and overlay-ready output.
5.10: real video evaluation and calibration.