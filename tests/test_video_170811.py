"""17:08录像真实RGB与显式合成深度，校验归属、层数及真盒位置。"""
import runpy
from pathlib import Path
import cv2
import numpy as np
import pytest
from classify import inspect_frame
from car_pose import OPTICAL_FROM_CORNER,camera_frame
from field_map import DISPLAY

R=runpy.run_path(str(Path(__file__).resolve().parents[1]/'debug_captures/video_170811/replay.py'))

@pytest.mark.parametrize('case',R['CASES'],ids=lambda c:str(c['second']))
@pytest.mark.parametrize('mode',['missing','background','foreground'])
def test_video_without_owned_depth_stays_unknown(case,mode):
    p,pitch,c,d=R['scene'](case,mode)
    v=inspect_frame(p,OPTICAL_FROM_CORNER,pitch,R['INTR'],c,d)[DISPLAY[case['tower']-1]]
    assert v.conclusion is None,(v.conclusion,v.reason)

@pytest.mark.parametrize('second,expected,roi',[(60,'1',(200,560,710,720)),(163,'2',(440,430,580,640))])
def test_video_real_rgb_with_controlled_owned_depth(second,expected,roi):
    case=next(c for c in R['CASES'] if c['second']==second)
    p,pitch,c,d=R['scene'](case,'owned')
    v=inspect_frame(p,OPTICAL_FROM_CORNER,pitch,R['INTR'],c,d)[DISPLAY[case['tower']-1]]
    assert v.conclusion==expected,(v.reason,v.measurements)
    x,y,w,h=cv2.boundingRect(v.piece)
    a,b,e,f=roi
    overlap=max(0,min(x+w,e)-max(x,a))*max(0,min(y+h,f)-max(y,b))
    assert overlap/(w*h)>.8
    assert v.method in ('深度','上部盒面','远处白盒')

@pytest.mark.parametrize('second',[3,241])
def test_hidden_tower_rgb_does_not_override_visibility(second):
    case=next(c for c in R['CASES'] if c['second']==second)
    p,pitch,c,d=R['scene'](case,'rgb')
    v=inspect_frame(p,OPTICAL_FROM_CORNER,pitch,R['INTR'],c,d)[DISPLAY[case['tower']-1]]
    assert v.conclusion is None

S=runpy.run_path('tests/test_robust_observation.py')

@pytest.mark.parametrize('layers',[1,2])
def test_near_partial_measured_body_recovers_without_rgb_outline(layers):
    p,c,d,_,rays=S['boxes_scene'](layers,1.8)
    origin,*_=camera_frame(p,S['OFFSET'],0)
    world=origin+d[...,None]*rays
    # 保留13.5cm宽的实测盒面；其余盒面深度缺测，背景深度仍真实可用。
    body=d<6
    d[body&(world[...,0]<S['CELL'].x+.035)]=np.nan
    c[:]=160
    v=inspect_frame(p,S['OFFSET'],0,S['INTR'],c,d)[S['CELL'].name]
    assert v.conclusion==str(layers),(v.reason,v.measurements)
    assert v.method in ('深度','上部盒面')
    if v.method=='上部盒面':
        assert v.measurements['upper_white']['depth_primary']
    else:
        assert v.measurements['depth_primary']

@pytest.mark.parametrize('distance',[1.8,2.5,4.2])
def test_staggered_shadowed_two_layers_keep_height(distance):
    cell=S['CELL']
    p,c,d,render,_=S['boxes_scene'](0,distance)
    render(cell.x-.175,cell.x+.175,cell.y-.175,cell.y+.175,0,.35,(190,190,190))
    render(cell.x-.075,cell.x+.25,cell.y-.09,cell.y+.25,.35,.70,(225,225,225))
    v=inspect_frame(p,S['OFFSET'],0,S['INTR'],c,d)[cell.name]
    assert v.conclusion=='2',(v.reason,v.measurements)
