"""层数竞争时才检查接缝；单层、独立背景及明确深度反证不改判。"""
import runpy
import cv2
import numpy as np
import pytest
from car_pose import project,to_camera
from classify import _mask,_white_mask,_rgb_white_top,_resolve_layer_seams

S=runpy.run_path('tests/test_robust_observation.py')
CELL,OFFSET,INTR=S['CELL'],S['OFFSET'],S['INTR']


def two_tone_scene(dx=0.):
    p,c,d,render,_=S['boxes_scene'](0,4.2)
    render(CELL.x-.175,CELL.x+.175,CELL.y-.175,CELL.y+.175,0,.35,(225,225,225))
    render(CELL.x-.175+dx,CELL.x+.175+dx,CELL.y-.175,CELL.y+.175,.35,.70,(185,185,185))
    # 色差使中间横线通过原来的邻域亮度过滤；加暗缝模拟真实阴影。
    _,v=project(CELL.center(CELL.one_z),p,OFFSET,0,INTR)
    yy,xx=np.nonzero(np.any(c!=160,axis=2))
    c[int(v)-1:int(v)+2,xx.min():xx.max()+1]=65
    return p,c,d


@pytest.mark.parametrize('dx',[0.,.05,-.05])
@pytest.mark.parametrize('with_depth',[False,True])
def test_conflicting_one_and_two_use_surface_above_shadow_seam(dx,with_depth):
    p,c,d=two_tone_scene(dx)
    info={}
    r=_rgb_white_top(CELL,c,_white_mask(c),_mask(CELL,p,OFFSET,0,INTR),p,
        OFFSET,0,INTR,float(to_camera(CELL.center(CELL.one_z),p,OFFSET,0)[2]),
        diagnostic=info,depth=d if with_depth else None)
    assert r is not None and r[0]=='2',info
    # 深度有时已直接排除一层，只有两种有效候选并存时才启动新仲裁。
    if not with_depth:
        conflict=info['rgb_white_top']['seam_conflict']
        assert conflict['triggered'] and conflict['state']=='seam_rejected',info


def contenders(p,c):
    yy,xx=np.nonzero(np.any(c!=160,axis=2))
    left,right=int(xx.min()),int(xx.max())
    result=[]
    for state,height in [('1',CELL.one_z),('2',CELL.two_z)]:
        _,v=project(CELL.center(height),p,OFFSET,0,INTR)
        row=int(round(v))
        piece=np.array([[[left,row]],[[right,row]],[[right,row+10]],[[left,row+10]]],np.int32)
        result.append((state,piece,height,False))
    return result


def test_single_level_does_not_read_image_or_start_seam_check():
    item=('1',None,CELL.one_z,False)
    diagnostic={}
    # 不竞争时，无需图像和深度，也不应启动接缝算法。
    assert _resolve_layer_seams([item],None,None,None,CELL,None,None,
                               0,INTR,4.2,diagnostic=diagnostic)==[item]
    assert diagnostic=={}


def test_uniform_white_background_without_continuing_sides_is_not_a_seam():
    p,c,d=two_tone_scene()
    items=contenders(p,c)
    c[:]=220
    info={}
    result=_resolve_layer_seams(items,c,_white_mask(c),_mask(CELL,p,OFFSET,0,INTR),
        CELL,p,OFFSET,0,INTR,4.2,diagnostic=info)
    assert len(result)==2 and info['state']=='unresolved'


def test_background_depth_cannot_link_two_white_surfaces():
    p,c,d=two_tone_scene()
    items=contenders(p,c)
    _,v=project(CELL.center(CELL.one_z),p,OFFSET,0,INTR)
    d[:int(v)-2]=8.
    info={}
    result=_resolve_layer_seams(items,c,_white_mask(c),_mask(CELL,p,OFFSET,0,INTR),
        CELL,p,OFFSET,0,INTR,4.2,depth=d,diagnostic=info)
    assert len(result)==2 and info['state']=='unresolved'
    assert any('depth_ownership' in q for q in info['checks'])


@pytest.mark.parametrize('layers',[1,2])
def test_noncompeting_frame_does_not_call_seam_arbitration(monkeypatch,layers):
    import classify
    p,c,_,_,_=S['boxes_scene'](layers,4.2)
    def unexpected(*args,**kwargs):
        raise AssertionError('没有层数竞争时不应启动接缝仲裁')
    monkeypatch.setattr(classify,'_resolve_layer_seams',unexpected)
    info={}
    r=classify._rgb_white_top(CELL,c,_white_mask(c),_mask(CELL,p,OFFSET,0,INTR),p,
        OFFSET,0,INTR,float(to_camera(CELL.center(CELL.one_z),p,OFFSET,0)[2]),diagnostic=info)
    assert r is not None and r[0]==str(layers)
    assert 'seam_conflict' not in info['rgb_white_top']


def test_video_1906_shadow_seam_keeps_tower3_two_layers():
    from pathlib import Path
    import math
    from car_pose import CarPose,OPTICAL_FROM_CORNER,load_intrinsics
    from field_map import DISPLAY
    from classify import inspect_frame
    root=Path(__file__).resolve().parents[1]
    image=cv2.imread(str(root/'debug_captures/rgb_validation_20261116_1906/rgb_43.png'))
    assert image is not None
    # 来自录屏HUD；俯仰取显示值，未录得的roll按0回放，不模拟原始深度。
    pose=CarPose(-.044,.049,math.radians(5.6))
    view=inspect_frame(pose,OPTICAL_FROM_CORNER,-1,
                       load_intrinsics(str(root/'config/intrinsics.yaml')),image,None,0)[DISPLAY[2]]
    assert view.conclusion=='2',(view.reason,view.measurements)
    info=view.measurements['rgb_white_top']['seam_conflict']
    assert info['state']=='seam_rejected' and info['removed_indices']
