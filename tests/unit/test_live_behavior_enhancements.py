"""Precision upgrades: synthetic tests that do not require live camera."""
import numpy as np

from company_vision.behaviors.live_behavior_enhancements import (
    WeightedStillness,
    _plugin_state,
    box_iou,
)


def track(dx=0.,tid=7):
    pts=np.zeros((9,3),dtype=np.float32)
    for i in range(9):
        pts[i]=[100+i*3+dx,120+i*4,0.95]
    return {'track_id':tid,'bbox':[90+dx,90,190+dx,280],
            'keypoints':pts,'keypoint_weights':np.ones(9),
            'observed':True,'behavior_eligible':True}


def test_overlap():
    assert box_iou([0,0,10,10],[0,0,10,10]) == 1.


def test_new_track_observing():
    item=WeightedStillness().update([track()],1.0)[7]
    assert item['state']=='OBSERVING'


def test_pixel_jitter_does_not_reset():
    model=WeightedStillness()
    for i in range(70):
        result=model.update([track(dx=(i%3)-1)],float(i))[7]
    assert result['state']=='PROLONGED STILLNESS'
    assert result['seconds']>=60


def test_real_motion_resets_clock():
    model=WeightedStillness()
    for i in range(20):
        model.update([track()],float(i))
    for i in range(20,23):
        result=model.update([track(dx=45.)],float(i))[7]
    assert result['seconds']<10


def test_unknown_when_bad_keypoints():
    model=WeightedStillness()
    item=track()
    item['keypoints'][:,2]=0
    assert model.update([item],1.)[7]['state']=='UNKNOWN'


def test_drop_missing_track():
    model=WeightedStillness()
    model.update([track()],1.)
    model.update([],2.)
    assert model.states=={}


def test_unknown_plugin_not_false_negative():
    assert _plugin_state({'status':'unknown','metrics':{}})=='UNKNOWN'