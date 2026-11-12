# Node 5.8.2 - ComputerUsagePlugin

Status: implemented, pending execution and validation on user's project.

## Semantic distinction
- `default_work_assumed`: operational baseline for a staff member. This is a **policy assumption**, not verified computer use; represented by `status=unknown` without a numeric score.
- `computer_activity_observed`: input device (keyboard, mouse, laptop) close to a valid wrist **and** meaningful wrist motion from the same tracked person in a configured time window. This is evidence of interaction, not proof of productivity.
- Mere seating, visible monitor, an absent object detection result, a predicted track or stationary wrist cannot assert computer use.

## Contract
Input: `BehaviorPluginContext(FrameFeatures, TemporalTrackHistory...)`.
Output: `PluginResult(plugin_id='computer_usage', payload={'observations': [...]})` compatible with `BehaviorPluginRegistry` and `RuleEngine`.

## Out of scope
No modification of upstream Pose, Tracking, or other plugins. No video/SDK, persistence, database, global work-state arbitration, or real camera accuracy claims. Temporal proof requires an upstream history. Thresholds are starting hypotheses, not field calibration.

## Acceptance
Compilation, Ruff, 16 focused tests, complete regression, pip check, unchanged dependency lock, Git commit, non-fast-forward merge and remote synchronization. Real-world calibration pending for Nodes 5.9/5.10.