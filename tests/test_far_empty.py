import math
import runpy
import numpy as np
import pytest
from classify import _far_empty_candidate,_mask,CellView
from report import accumulate_empty_candidates
from car_pose import CarPose,camera_frame
from field_map import DISPLAY
S=runpy.run_path('tests/test_robust_observation.py')


def scene():
    p,c,d,_,rays=S['boxes_scene'](0,4.2,pitch=-10)
    origin=camera_frame(p,S['OFFSET'],-10)[0]
    d=((S['CELL'].base_z-origin[2])/rays[:,:,2]).astype(np.float32)
    c[:]=(70,100,130)
    mask=_mask(S['CELL'],p,S['OFFSET'],-10,S['INTR'])
    return p,c,d,mask

@pytest.mark.parametrize('mode',['floor','rgb','missing','foreground','background','white','dark'])
def test_far_empty_requires_surface_and_no_counterevidence(mode):
    p,c,d,mask=scene()
    if mode=='rgb':d=None
    elif mode=='missing':d[:]=np.nan
    elif mode=='foreground':d[:]=1.
    elif mode=='background':d[:]=8.
    elif mode=='white':c[:]=220
    elif mode=='dark':c[:]=0;d=None
    info=_far_empty_candidate(S['CELL'],p,S['OFFSET'],-10,S['INTR'],c,d,mask)
    assert info['candidate']==(mode in ('floor','rgb','missing')),info


def view():
    return CellView(None,True,False,False,False,0,0,0,100,
        measurements={'far_empty':{'candidate':True}})

def test_far_empty_accumulates_without_lock_and_resets_on_move():
    tracks={};name=DISPLAY[5];p=CarPose(0,0,0)
    for t in [0,.4,.8,1.2,1.6,2.1]:
        v=view();accumulate_empty_candidates({name:v},tracks,p,0,0,t,True)
        assert v.conclusion is None and v.skip_vote
    assert v.needs_parallax and '换位' in v.reason
    v=view();accumulate_empty_candidates({name:v},tracks,CarPose(.1,0,0),0,0,2.2,True)
    assert not v.needs_parallax and tracks[name]['count']==1
    accumulate_empty_candidates({name:v},tracks,p,0,0,2.3,False)
    assert not tracks


def test_duplicate_time_does_not_accumulate_and_gap_resets():
    tracks={};name=DISPLAY[5];p=CarPose(0,0,0)
    for _ in range(10):accumulate_empty_candidates({name:view()},tracks,p,0,0,1.,True)
    assert tracks[name]['count']==1
    v=view();accumulate_empty_candidates({name:v},tracks,p,0,0,3.,True)
    assert tracks[name]['count']==1 and not v.needs_parallax


def test_full_classifier_marks_far_empty_pending_not_confirmed():
    from classify import inspect_frame
    p,c,d,_=scene()
    v=inspect_frame(p,S['OFFSET'],-10,S['INTR'],c,d)[S['CELL'].name]
    assert v.conclusion is None and v.skip_vote
    assert v.measurements['far_empty']['candidate']
    assert '疑似0层' in v.reason
