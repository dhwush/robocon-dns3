"""近处3cm层高优先；弱顶边不否决，实体归属和上层反证不放宽。"""
import runpy
import numpy as np
import pytest
import classify
from car_pose import camera_position

S=runpy.run_path('tests/test_robust_observation.py')
C,O,I=S['CELL'],S['OFFSET'],S['INTR']

@pytest.mark.parametrize('layers',[1,2])
@pytest.mark.parametrize('delta',[-.015,0.,.015,.05])
def test_close_height_overrides_weak_top_only_within_three_cm(monkeypatch,layers,delta):
    p,c,d,render,_=S['boxes_scene'](0,1.8)
    render(C.x-.175,C.x+.175,C.y-.175,C.y+.175,0,layers*.35+delta,(220,220,220))
    c[:]=160
    monkeypatch.setattr(classify,'_near_depth_top',lambda *a,**kw:False)
    monkeypatch.setattr(classify,'_near_top_observed',lambda *a,**kw:False)
    v=classify.inspect_frame(p,O,0,I,c,d)[C.name]
    if abs(delta)<.03:
        assert v.conclusion==str(layers),(v.reason,v.measurements)
        assert v.method=='深度'
        assert v.measurements['precise_depth']['accepted']
        assert v.measurements['precise_depth']['error_cm']<=3.
    else:
        assert v.conclusion is None,(v.conclusion,v.reason)
        assert not v.measurements.get('precise_depth',{}).get('accepted')


def test_upper_missing_near_layer_cannot_downgrade_true_two(monkeypatch):
    p,c,d,_,rays=S['boxes_scene'](2,1.8)
    heights=camera_position(p,O,0)[2]+d*rays[...,2]
    d[(d<6)&(heights>.35)]=np.nan
    c[:]=160
    v=classify.inspect_frame(p,O,0,I,c,d)[C.name]
    assert v.conclusion is None
    assert not v.measurements.get('precise_depth',{}).get('accepted')


def test_higher_missing_background_does_not_override_close_layer(monkeypatch):
    p,c,d,_,rays=S['boxes_scene'](1,1.8)
    heights=camera_position(p,O,0)[2]+d*rays[...,2]
    d[(d>6)&(heights>.8)]=np.nan
    c[:]=160
    monkeypatch.setattr(classify,'_near_depth_top',lambda *a,**kw:False)
    monkeypatch.setattr(classify,'_near_top_observed',lambda *a,**kw:False)
    v=classify.inspect_frame(p,O,0,I,c,d)[C.name]
    assert v.conclusion=='1',(v.reason,v.measurements)
    assert v.measurements['precise_depth']['accepted']
    assert v.measurements['precise_depth']['upper_space']['state']=='unknown'


def test_rgb_higher_line_cannot_override_precise_one_with_clear_depth(monkeypatch):
    p,c,d,_,_=S['boxes_scene'](1,1.8)
    monkeypatch.setattr(classify,'_silhouette_top',lambda *a,**kw:C.two_z)
    v=classify.inspect_frame(p,O,0,I,c,d)[C.name]
    assert v.conclusion=='1',(v.reason,v.measurements)
    assert v.measurements['precise_depth']['accepted']


@pytest.mark.parametrize('layers',[1,2])
def test_precise_layer_survives_local_top_depth_interference(layers):
    p,c,d,_,_=S['boxes_scene'](layers,1.8)
    body=d<6
    # 盒顶附近4像素混合近回波，真实盒身高度与更高处背景保持有效。
    for u in np.nonzero(body.any(axis=0))[0]:
        v=int(np.nonzero(body[:,u])[0][0])
        d[max(0,v-4):v,u]=.5
    c[:]=160
    view=classify.inspect_frame(p,O,0,I,c,d)[C.name]
    assert view.conclusion==str(layers),(view.reason,view.measurements)
    assert view.measurements['precise_depth']['accepted']
    assert not view.measurements['depth_top_check']['state']=='supported'
