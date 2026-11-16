"""真实射线渲染40cm前景台，以及左右视野裁切；不把裁切深度伪造为盒顶。"""
import runpy
from dataclasses import replace
import numpy as np
import pytest
from classify import inspect_frame
from car_pose import camera_position, project

S=runpy.run_path('tests/test_robust_observation.py')
CELL,OFFSET,INTR=S['CELL'],S['OFFSET'],S['INTR']

@pytest.mark.parametrize('pitch',[-8,0,8])
@pytest.mark.parametrize('exposed',[.05,.10,.15,.20,.25])
def test_one_layer_above_40cm_foreground(exposed,pitch):
    p,c,d,render,_=S['boxes_scene'](1,1.5,pitch=pitch)
    origin=camera_position(p,OFFSET,pitch)
    cut=CELL.one_z-exposed
    ratio=(.4-origin[2])/(cut-origin[2])
    front=origin[1]+ratio*(CELL.y-.175-origin[1])
    render(CELL.x-.6,CELL.x+.6,front,front+.01,0,.4,(30,70,30))
    view=inspect_frame(p,OFFSET,pitch,INTR,c,d)[CELL.name]
    if pitch==8:
        # 此相机/距离抬头8度时，前侧真实顶边已落到画面下沿之外。
        uv=project(np.array([CELL.x,CELL.y-.175,CELL.one_z]),p,OFFSET,pitch,INTR)
        assert uv[1]>=INTR.height-2
        assert view.conclusion is None
        return
    assert view.conclusion=='1', (exposed,view.reason,view.measurements)
    assert view.method in ('上部盒面','深度')

@pytest.mark.parametrize('layers',[2])
def test_foreground_cannot_turn_two_layers_into_one(layers):
    p,c,d,render,_=S['boxes_scene'](layers,1.5)
    render(CELL.x-.6,CELL.x+.6,CELL.y-.6,CELL.y-.58,0,.4,(30,70,30))
    assert inspect_frame(p,OFFSET,0,INTR,c,d)[CELL.name].conclusion!='1'

@pytest.mark.parametrize('use_depth',[False,True])
@pytest.mark.parametrize('cap',[False,True])
@pytest.mark.parametrize('remaining',[.80,.90])
@pytest.mark.parametrize('side',['left','right'])
def test_side_cut_80_to_90_percent(cap,remaining,side,use_depth):
    p,c,d,_,_=S['boxes_scene'](2 if cap else 1,2.,cap=cap)
    target=(np.all(c==(0,0,220),axis=2)|np.all(c==(220,0,0),axis=2)) if cap else np.all(c==220,axis=2)
    yy,xx=np.nonzero(target)
    if side=='left':
        start=int(xx.min()+(1-remaining)*(xx.max()-xx.min()+1))
        c,d=c[:,start:].copy(),d[:,start:].copy()
        intr=replace(INTR,width=c.shape[1],cx=INTR.cx-start)
    else:
        stop=int(xx.max()+1-(1-remaining)*(xx.max()-xx.min()+1))
        c,d=c[:,:stop].copy(),d[:,:stop].copy()
        intr=replace(INTR,width=c.shape[1])
    view=inspect_frame(p,OFFSET,0,intr,c,d if use_depth else None)[CELL.name]
    assert view.conclusion==('红' if cap else '1'),(view.reason,view.measurements)
    assert view.in_view and not view.needs_yaw and not view.needs_pitch
    assert not view.skip_vote


def test_missing_upper_depth_is_not_a_one_layer_top():
    p,c,d,_,_=S['boxes_scene'](2,1.5)
    # 真两层塔缺掉上半层深度：图像仍连续，不能据深度缺失边当盒顶。
    origin,right,down,ahead=S['camera_frame'](p,OFFSET,0,0)
    yy,xx=np.indices(d.shape)
    worldz=origin[2]+d*(ahead[2]+(xx-INTR.cx)/INTR.fx*right[2]+(yy-INTR.cy)/INTR.fy*down[2])
    d[worldz>.35]=0
    assert inspect_frame(p,OFFSET,0,INTR,c,d)[CELL.name].conclusion!='1'
