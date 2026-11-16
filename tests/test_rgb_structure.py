"""独立三维渲染验证RGB结构；背景也通过白色HSV时仍须有真实边界。"""
import runpy
import numpy as np
import pytest
from classify import inspect_frame, _rgb_white_edges, _white_mask, _mask, _cap_width_limit
from car_pose import to_camera

S=runpy.run_path('tests/test_robust_observation.py')

@pytest.mark.parametrize('layers',[1,2])
def test_far_rgb_separates_box_from_white_hsv_background(layers):
    p,c,_,_,_=S['boxes_scene'](layers,4.2)
    intr,offset,cell=S['INTR'],S['OFFSET'],S['CELL']
    view=inspect_frame(p,offset,0,intr,c,None)[cell.name]
    assert view.conclusion==str(layers)
    assert view.method=='RGB盒体边界'

@pytest.mark.parametrize('scene',['wall','no_top_edge'])
def test_rgb_structure_rejects_missing_real_boundaries(scene):
    p,c,_,_,_=S['boxes_scene'](1,4.2)
    intr,offset,cell=S['INTR'],S['OFFSET'],S['CELL']
    if scene=='wall':
        c[:]=220
    else:
        yy,xx=np.nonzero(np.all(c==220,axis=2))
        if scene=='occluded':
            c[yy.min():yy.max()+1,xx.min():(xx.min()+xx.max())//2]=40
        else:
            c[:yy.min(),xx.min():xx.max()+1]=220
    z=to_camera(cell.center(cell.one_z),p,offset,0)[2]
    assert _rgb_white_edges(cell,c,_white_mask(c),_mask(cell,p,offset,0,intr),p,offset,0,intr,z) is None


def test_cap_projection_limit_is_bounded_and_retains_both_colors_requirement():
    p,c,_,_,_=S['boxes_scene'](2,2.5,45,cap=True)
    intr,offset,cell=S['INTR'],S['OFFSET'],S['CELL']
    z=to_camera(cell.center(cell.one_z),p,offset,0)[2]
    assert .34<=_cap_width_limit(cell,p,offset,0,intr,z)<=.40
    c[np.all(c==(0,0,220),axis=2)]=(220,0,0)
    assert inspect_frame(p,offset,0,intr,c,None)[cell.name].conclusion not in ('红','蓝')


def test_two_visible_cap_faces_do_not_fail_fixed_34cm_width():
    p,c,_,render,_=S['boxes_scene'](2,2.5,45)
    cell,offset,intr=S['CELL'],S['OFFSET'],S['INTR']
    # 独立射线渲染有尺寸/分割容差的旋转顶块，不手工构造检测器输出。
    for low,high,ink in [(.70,.80,(220,0,0)),(.80,.90,(0,0,220))]:
        render(cell.x-.13,cell.x+.13,cell.y-.13,cell.y+.13,low,high,ink)
    view=inspect_frame(p,offset,0,intr,c,None)[cell.name]
    checks=view.measurements['rgb_cap']['checks']
    assert any(q['width_cm']>34 and not q['failures'] for q in checks)
    assert view.conclusion=='红'


@pytest.mark.parametrize('remaining',[.45,.65])
def test_block_pair_survives_horizontal_crop_and_height_mismatch(remaining):
    from dataclasses import replace
    from classify import _rgb_cap_reading
    p,c,_,render,_=S['boxes_scene'](0,2.5)
    cell,offset,intr=S['CELL'],S['OFFSET'],S['INTR']
    for low,high,ink in [(.30,.40,(220,0,0)),(.40,.50,(0,0,220))]:
        render(cell.x-.10,cell.x+.10,cell.y-.10,cell.y+.10,low,high,ink)
    yy,xx=np.nonzero(c[:,:,2]!=c[:,:,0])
    start=int(xx.min()+(1-remaining)*(xx.max()-xx.min()+1))
    c=c[:,start:].copy()
    intr=replace(intr,width=c.shape[1],cx=intr.cx-start)
    z=to_camera(cell.center(cell.one_z),p,offset,0)[2]
    result=_rgb_cap_reading(cell,c,_mask(cell,p,offset,0,intr),p,offset,0,intr,z)
    assert result is not None and result[0]=='红'


def test_depth_background_cannot_validate_rgb_box_edges():
    p,c,d,_,_=S['boxes_scene'](1,4.2)
    cell,offset,intr=S['CELL'],S['OFFSET'],S['INTR']
    z=to_camera(cell.center(cell.one_z),p,offset,0)[2]
    args=(cell,c,_white_mask(c),_mask(cell,p,offset,0,intr),p,offset,0,intr,z)
    assert _rgb_white_edges(*args,depth=d) is not None
    assert _rgb_white_edges(*args,depth=np.full_like(d,8.)) is None
