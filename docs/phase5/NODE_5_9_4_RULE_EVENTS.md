# Node 5.9.4 - Rule and Event Integration

Status: IMPLEMENTED / AUTOMATED ACCEPTANCE PENDING

## Purpose
Connect Node 5.9.3 behavior evidence to the existing
RuleEngine and EventManager contracts.

## Behavior rules
- phone_usage
- computer_usage
- inactivity
- gathering

Rule thresholds and minimum confidence are configurable.

## Event lifecycle
- started: first validated positive observation
- confirmed: positive observations reach policy requirement
- updated: existing event changes
- ended: sufficient negative evidence or explicit lifecycle end

## Unknown evidence
Unknown is not negative.
Unknown does not start a new event or prove a violation.
Missing observations are not manufactured.

## Identity boundaries
Camera ID + session ID + rule ID + track ID.
Session changes and explicit track endings close matching events.

## Priority
The priority order determines only how evidence can be
presented to downstream consumers. It does not silently
discard other evidence or infer productivity or misconduct.

## Output
RuleEventBatch:
- source BehaviorExecutionBatch
- RuleEvaluation records
- EventTransition records
- active BehaviorEvent snapshots

## Limits
No event database persistence.
No operator notifications.
No live camera/GPU test.
No overlay or final person-status arbitration.
A failed plugin or rule step can leave the upstream
temporal frame consumed; the caller must not blindly replay it.

## Next
Node 5.9.5: operational output and overlay-ready formatting.
Node 5.10: real-video validation and calibration.