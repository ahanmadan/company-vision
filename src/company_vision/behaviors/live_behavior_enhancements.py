"""Precision overlay and noise-aware stillness for real Channel-2 observations.

This module never concludes that a person is asleep or unproductive.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite

import cv2
import numpy as np

OBJECT_IDS = (63, 64, 66, 67)
JOINT_PRIORITY = np.array([.85,.45,.45,.4,.4,1.,1.,.95,.95], dtype=np.float64)


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0., x2-x1) * max(0., y2-y1)
    aa = max(0., a[2]-a[0])*max(0., a[3]-a[1])
    bb = max(0., b[2]-b[0])*max(0., b[3]-b[1])
    return intersection/max(aa+bb-intersection, 1.e-6)


def _parse(result, xoff=0, yoff=0):
    items = []
    if result.boxes is None or len(result.boxes) == 0:
        return items
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    for coords, cls, confidence in zip(xyxy,classes,scores,strict=True):
        cid = int(cls)
        if cid not in OBJECT_IDS:
            continue
        x1,y1,x2,y2 = map(float,coords)
        items.append({
            "object_id": 0,
            "class_id": cid,
            "class_name": str(result.names[cid]),
            "confidence": float(confidence),
            "bbox": [x1+xoff,y1+yoff,x2+xoff,y2+yoff],
        })
    return items


def _valid_crop(frame, box):
    height, width = frame.shape[:2]
    x1,y1,x2,y2 = map(float, box)
    cx,cy = (x1+x2)/2., (y1+y2)/2.
    span_x = max(80., (x2-x1)*.70)
    span_y = max(90., (y2-y1)*.65)
    xa = max(0,int(cx-span_x))
    xb = min(width,int(cx+span_x))
    ya = max(0,int(cy-span_y))
    yb = min(height,int(cy+span_y))
    if xb-xa < 48 or yb-ya < 48:
        return None
    return xa,ya,xb,yb


def detect_precise_objects(model, frame, tracks, *, imgsz=1280):
    """Full image + up to 3 track ROIs, without claiming guaranteed detection."""
    height,width = frame.shape[:2]
    def infer(image, image_size, confidence):
        return model.predict(
            source=image, device="cuda:0", imgsz=image_size,
            conf=confidence, classes=list(OBJECT_IDS),
            verbose=False,
        )[0]
    candidates = _parse(infer(frame,imgsz,.17))
    observed = [p for p in tracks if p.get('observed') is True]
    observed.sort(key=lambda p: float(p.get('confidence',0)),reverse=True)
    for person in observed[:3]:
        coords = _valid_crop(frame,person['bbox'])
        if coords is None:
            continue
        xa,ya,xb,yb = coords
        crop = frame[ya:yb,xa:xb]
        # The crop increases relative phone size rather than synthesizing pixels.
        candidates.extend(_parse(infer(crop,960,.12),xa,ya))
    candidates.sort(key=lambda v:v['confidence'],reverse=True)
    selected = []
    for item in candidates:
        box = item['bbox']
        box[0]=max(0.,min(float(width-1),box[0]))
        box[1]=max(0.,min(float(height-1),box[1]))
        box[2]=max(0.,min(float(width),box[2]))
        box[3]=max(0.,min(float(height),box[3]))
        if box[2]<=box[0] or box[3]<=box[1]:
            continue
        if any(old['class_id']==item['class_id'] and
               box_iou(old['bbox'],box)>.45 for old in selected):
            continue
        item['object_id']=len(selected)
        selected.append(item)
    return selected


@dataclass
class _TrackState:
    start: float
    last: float
    anchor: np.ndarray
    anchor_center: np.ndarray
    previous: np.ndarray
    bbox_diag: float
    move_hits: int = 0
    available: bool = True


# NODE_5102_STILLNESS_ROBUST_V3
class WeightedStillness:
    """Conservative clock-time stillness from quality-weighted pose evidence.

    Distinct from sleep, attention, or productivity classification.
    Bounding-box movement never independently resets this clock.
    """

    def __init__(
        self, *, jitter_pixels=3.0, relative_jitter=0.012,
        motion_fraction=0.025, max_gap=1.5, move_confirm=3,
    ):
        self.jitter_pixels = float(jitter_pixels)
        self.relative_jitter = float(relative_jitter)
        self.motion_fraction = float(motion_fraction)
        self.max_gap = float(max_gap)
        self.move_confirm = int(move_confirm)
        if (self.jitter_pixels < 0 or self.relative_jitter < 0
                or self.motion_fraction <= 0 or self.max_gap <= 0
                or self.move_confirm < 2):
            raise ValueError("Invalid stillness parameters")
        self.states = {}

    @staticmethod
    def _measurement(person):
        pts = np.asarray(person.get("keypoints"), dtype=np.float64)
        box = np.asarray(person.get("bbox"), dtype=np.float64)
        if (pts.shape != (9, 3) or box.shape != (4,)
                or not np.isfinite(pts).all() or not np.isfinite(box).all()
                or box[2] <= box[0] or box[3] <= box[1]):
            return None
        raw = np.asarray(
            person.get("keypoint_weights", np.ones(9)), dtype=np.float64
        )
        if raw.shape != (9,) or not np.isfinite(raw).all():
            raw = np.ones(9, dtype=np.float64)
        weights = np.clip(raw, 0, 1) * np.clip(pts[:, 2], 0, 1) * JOINT_PRIORITY
        weights[pts[:, 2] < 0.35] = 0.0
        # At least 2 dependable torso landmarks and one additional landmark.
        if (np.count_nonzero(weights[5:7] > 0.15) < 2
                or np.count_nonzero(weights > 0.12) < 3):
            return None
        diag = max(float(hypot(box[2] - box[0], box[3] - box[1])), 1.0)
        return pts.copy(), weights.copy(), diag

    @staticmethod
    def _label(seconds):
        if seconds >= 60.0:
            return "PROLONGED STILLNESS"
        if seconds >= 30.0:
            return "STILLNESS 30s+"
        if seconds >= 10.0:
            return "STILLNESS OBSERVED"
        return "OBSERVING"

    def update(self, tracks, now):
        now = float(now)
        if not isfinite(now):
            raise ValueError("Timestamp must be finite")
        output = {}
        seen = set()
        for person in tracks:
            track_id = person.get("track_id")
            if track_id is None:
                continue
            tid = int(track_id)
            if tid in seen:
                # Never combine ambiguous identities in a clock.
                self.states.pop(tid, None)
                output[tid] = {
                    "state": "UNKNOWN", "seconds": 0.0,
                    "reason": "duplicate_track_id",
                }
                continue
            seen.add(tid)
            if (person.get("observed") is not True
                    or person.get("behavior_eligible") is not True):
                self.states.pop(tid, None)
                output[tid] = {
                    "state": "UNKNOWN", "seconds": 0.0,
                    "reason": "track_not_observed_or_ineligible",
                }
                continue
            item = self._measurement(person)
            if item is None:
                self.states.pop(tid, None)
                output[tid] = {
                    "state": "UNKNOWN", "seconds": 0.0,
                    "reason": "reliable_keypoints_unavailable",
                }
                continue
            points, weights, diag = item
            state = self.states.get(tid)
            if (state is None or now <= state["last"]
                    or now - state["last"] > self.max_gap):
                state = {
                    "start": now, "last": now, "anchor": points,
                    "anchor_weights": weights,
                    "diag": diag, "move_hits": 0,
                    "last_motion": 0.0,
                }
                self.states[tid] = state
                output[tid] = {
                    "state": "OBSERVING", "seconds": 0.0,
                    "reason": "new_valid_baseline", "normalized_motion": 0.0,
                    "valid_joints": int(np.count_nonzero(weights > 0.12)),
                }
                continue

            anchor = state["anchor"]
            confidence = np.minimum(weights, state["anchor_weights"])
            mask = confidence > 0.12
            if (np.count_nonzero(mask[5:7]) < 2
                    or np.count_nonzero(mask) < 3):
                self.states.pop(tid, None)
                output[tid] = {
                    "state": "UNKNOWN", "seconds": 0.0,
                    "reason": "stable_joint_overlap_missing",
                }
                continue

            # Compare body landmarks to the stillness baseline. Unlike
            # consecutive-frame differences this catches gradual motion.
            # Use a stable scale, not the detector's current bounding box.
            stable_diag = float(state["diag"])
            tolerance = max(
                self.jitter_pixels, self.relative_jitter * stable_diag
            )
            residual = np.linalg.norm(
                points[:, :2] - anchor[:, :2], axis=1
            )
            normalized = np.maximum(0.0, residual[mask] - tolerance)
            conf = confidence[mask]
            # Weighted RMS preserves genuine coherent movement while
            # keeping low-confidence one-joint spikes from dominating.
            score = float(np.sqrt(
                np.average(normalized ** 2, weights=conf)
            ) / stable_diag)
            moving = score > self.motion_fraction
            state["move_hits"] = (
                state["move_hits"] + 1 if moving else 0
            )
            state["last_motion"] = score
            state["last"] = now
            reset = state["move_hits"] >= self.move_confirm
            if reset:
                # Motion must persist; one bbox/pose spike is insufficient.
                state.update({
                    "start": now,
                    "anchor": points.copy(),
                    "anchor_weights": weights.copy(),
                    "diag": diag,
                    "move_hits": 0,
                })
            seconds = max(0.0, now - state["start"])
            output[tid] = {
                "state": self._label(seconds),
                "seconds": round(seconds, 2),
                "reason": "confirmed_motion_reset" if reset else
                          "weighted_pose_motion",
                "normalized_motion": round(score, 6),
                "valid_joints": int(np.count_nonzero(mask)),
                "motion_hits": state["move_hits"],
                "jitter_tolerance_px": round(tolerance, 3),
                "body_scale_px": round(stable_diag, 3),
            }
        # Missing frames are not treated as stillness. Avoid stale states.
        for tid in list(self.states):
            if tid not in seen:
                del self.states[tid]
        return output

def _plugin_state(item):
    if not isinstance(item,dict):
        return 'UNKNOWN'
    metrics=item.get('metrics',{})
    if item.get('status')!='ok' or 'score' not in metrics:
        return 'UNKNOWN'
    score=float(metrics['score'])
    if score>=.75:
        return 'EVIDENCE'
    return 'NO EVIDENCE'


def render_head_status(image, output, objects, tracks, stillness):
    """Draw actual per-plugin evidence directly above each observed bbox."""
    height,width=image.shape[:2]
    by_id={int(item.track_id):item for item in output.results}
    for obj in objects:
        a,b,c,d=map(int,obj['bbox'])
        is_phone=obj['class_id']==67
        color=(0,230,255) if is_phone else (250,165,50)
        cv2.rectangle(image,(a,b),(c,d),color,2)
        cv2.putText(image,f"{'PHONE' if is_phone else obj['class_name']} {obj['confidence']:.2f}",
                    (max(0,a),max(20,b-6)),cv2.FONT_HERSHEY_SIMPLEX,.56,color,2)
    for person in tracks:
        if person.get('observed') is not True:
            continue
        tid=int(person['track_id'])
        result=by_id.get(tid)
        if result is None:
            continue
        state=stillness.get(tid,{'state':'UNKNOWN','seconds':0.})
        evidence=result.evidence
        phone=_plugin_state(evidence.get('phone_usage'))
        computer=_plugin_state(evidence.get('computer_usage'))
        group=_plugin_state(evidence.get('gathering'))
        # This is an independent visual stillness estimate, not a sleep diagnosis.
        lines=[f'ID {tid} | PHONE: {phone} | PC: {computer}',
               f"STILL: {state['state']} {state['seconds']:.0f}s | GROUP: {group}"]
        x1,y1,_,_=map(int,person['bbox'])
        x=max(0,min(x1,width-480))
        # Prefer just above the head; if close to top, move into upper box.
        y=y1-47 if y1>=56 else max(15,y1+18)
        # Opaque background avoids white text disappearing on bright scenes.
        cv2.rectangle(image,(x,max(0,y-18)),(min(width-1,x+480),min(height-1,y+27)),(14,18,24),-1)
        for i,line in enumerate(lines):
            cv2.putText(image,line,(x+5,y+i*19),cv2.FONT_HERSHEY_SIMPLEX,
                        .46,(255,255,255),1,cv2.LINE_AA)
    return image