# Node 5.2 — FeatureExtractor

## Purpose

Convert a validated BehaviorFrame into immutable geometric
features without modifying the source observations.

## Input

company_vision.behaviors.schemas.BehaviorFrame

## Output

company_vision.behaviors.feature_extractor.FrameFeatures

## Extracted information

- Frame and camera identity.
- Person and track identity.
- Bounding box normalized to frame dimensions.
- Keypoint coordinates normalized to person bounding box.
- Keypoint confidence and explicit weight.
- Weighted head center.
- Weighted shoulder center.
- Shoulder width and shoulder-line vector.
- Head-to-shoulder vector.
- Left and right shoulder-to-elbow vectors.
- Keypoint availability and quality statistics.
- Generic object bounding box features.

## Weight policy

When keypoint_weights is non-empty, the explicit weight
is authoritative.

An explicit zero weight makes the point unavailable for
geometric calculations, even when valid is true.

When the complete weight map is empty, confidence is used
as the quality fallback.

Missing coordinates and invalid points are preserved as
unavailable features rather than fabricated values.

## Coordinate policy

Keypoint coordinates are normalized relative to the
person bounding box.

Person and object bounding boxes are normalized relative
to frame dimensions.

Coordinates are not clipped or silently corrected.

## Geometry policy

Angles are measured using atan2 in image coordinates.
The positive Y axis points downward.

Arm vectors represent shoulder-to-elbow direction.
They are not anatomical elbow angles because wrist
observations are not part of the current nine-point contract.

## Architectural boundaries

This node does not:

- classify behavior;
- maintain temporal history;
- access camera hardware;
- execute tracking;
- associate objects with individual people;
- execute behavior rules;
- create events.

## Compatibility

- BehaviorFrame remains unchanged.
- BehaviorInputAdapter remains unchanged.
- Phases 1–4 remain unchanged.
- SYNC remains the reference pipeline.
- No fixed person-count limit is introduced.
- No model weights are modified.

## Acceptance

- Dedicated unit tests.
- Full project regression.
- Ruff.
- Python compilation.
- Git diff validation.
- Preservation of the existing input contract.