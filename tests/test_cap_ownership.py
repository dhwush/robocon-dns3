"""同一顶块只能归属一个塔；可靠深度优先，缺测时用RGB几何。"""
from dataclasses import replace
import math
import runpy
import numpy as np
import pytest
from car_pose import camera_frame, to_camera
from classify import _assign_caps, _mask, _rgb_cap_reading, inspect_frame
from field_map import TOWER_NO, tower_cells

M=runpy.run_path('tests/test_task3.py')
CELLS={TOWER_NO[c.name]:c for c in tower_cells()}
INTR,OFFSET=M['INTR'],M['OFFSET']


def paint_cap(color,depth,pose,cell,blue_top=False,shift=0.):
    inks=[(220,0,0),(0,0,220)]
    if blue_top:inks.reverse()
    for low,high,ink in [(cell.two_z,cell.two_z+.1,inks[0]),(cell.two_z+.1,cell.cap_z,inks[1])]:
        previous_color,previous_depth=color.copy(),depth.copy()
        M['_fill_plane'](color,depth,pose,OFFSET,0.,cell.y-.1,low,high,ink,cell.x+shift,.085)
        changed=np.any(color!=previous_color,axis=2)
        depth[~changed]=previous_depth[~changed]
        yy,xx=np.nonzero(changed)
        origin,right,down,ahead=camera_frame(pose,OFFSET,0.)
        direction=((xx-INTR.cx)/INTR.fx)*right[1]+((yy-INTR.cy)/INTR.fy)*down[1]+ahead[1]
        values=(cell.y-.1-origin[1])/direction
        depth[yy,xx]=values.astype(np.float32)


def overlapping_scene(front_number,both=False):
    front,back=CELLS[front_number],CELLS['6']
    dx,dy=back.x-front.x,back.y-front.y
    pose=M['_pose_for_camera'](front.x-.6*dx,front.y-.6*dy,-math.atan2(dx,dy),OFFSET)
    color=np.zeros((INTR.height,INTR.width,3),np.uint8)
    depth=np.full(color.shape[:2],8.,np.float32)
    paint_cap(color,depth,pose,back,blue_top=True)
    if both:paint_cap(color,depth,pose,front)
    return pose,color,depth


def local_reading(cell,pose,color):
    info={};z=float(to_camera(cell.center(cell.one_z),pose,OFFSET,0.)[2])
    r=_rgb_cap_reading(cell,color,_mask(cell,pose,OFFSET,0.,INTR),pose,OFFSET,0.,INTR,z,diagnostic=info)
    return r,info['rgb_cap']


@pytest.mark.parametrize('with_depth',[False,True])
def test_six_cap_cannot_be_claimed_by_four(with_depth):
    p,c,d=overlapping_scene('4')
    assert local_reading(CELLS['4'],p,c)[0][0]=='蓝'
    assert local_reading(CELLS['6'],p,c)[0][0]=='蓝'
    views=inspect_frame(p,OFFSET,0.,INTR,c,d if with_depth else None)
    assert views[CELLS['6'].name].conclusion=='蓝'
    assert views[CELLS['4'].name].conclusion not in ('红','蓝')
    assert any(r['owner']==CELLS['6'].name for r in views[CELLS['4'].name].measurements['cap_ownership'])
    assert all('_cap_allowed' not in v.measurements for v in views.values())


@pytest.mark.parametrize('with_depth',[False,True])
def test_other_tower_removed_before_counting_two_own_candidates(with_depth):
    p,c,d=overlapping_scene('2',both=True)
    assert local_reading(CELLS['2'],p,c)[1]['candidates']==2
    views=inspect_frame(p,OFFSET,0.,INTR,c,d if with_depth else None)
    assert views[CELLS['2'].name].conclusion=='红'
    assert views[CELLS['6'].name].conclusion=='蓝'
    assert views[CELLS['2'].name].measurements['rgb_cap']['candidates']==1


def test_equal_evidence_does_not_depend_on_tower_iteration_order():
    p,c,d=overlapping_scene('4');cell=CELLS['6'];twin=replace(cell,name='重叠塔位')
    for cells in ([cell,twin],[twin,cell]):
        for depth in (None,d):
            assignments=_assign_caps(cells,c,p,OFFSET,0.,INTR,depth=depth)
            for owned,possible,reports in assignments.values():
                assert not owned.any()
                assert possible.any()
                assert reports[0]['owner'] is None


def test_two_caps_really_at_same_tower_remain_unknown():
    cell=CELLS['6'];p=M['_pose_for_camera'](cell.x,cell.y-4.2,0.,OFFSET)
    c=np.zeros((INTR.height,INTR.width,3),np.uint8);d=np.full(c.shape[:2],8.,np.float32)
    for shift in (-.14,.14):paint_cap(c,d,p,cell,shift=shift)
    v=inspect_frame(p,OFFSET,0.,INTR,c,None)[cell.name]
    assert v.measurements['rgb_cap']['candidates']==2 and v.conclusion is None


def test_depth_separates_similar_rgb_projections_and_nan_falls_back():
    cell=CELLS['6'];decoy=replace(cell,name='投影相近的后方塔',y=cell.y+.6)
    p=M['_pose_for_camera'](cell.x,cell.y-4.2,0.,OFFSET)
    c=np.zeros((INTR.height,INTR.width,3),np.uint8);d=np.full(c.shape[:2],8.,np.float32)
    paint_cap(c,d,p,cell)
    cells=[cell,decoy]
    for depth in (None,np.full_like(d,np.nan)):
        assignments=_assign_caps(cells,c,p,OFFSET,0.,INTR,depth=depth)
        assert not assignments[cell.name][0].any()
    assignments=_assign_caps(cells,c,p,OFFSET,0.,INTR,depth=d)
    assert assignments[cell.name][0].any()
    assert not assignments[decoy.name][1].any()
    assert assignments[cell.name][2][0]['source']=='depth'
    assert assignments[decoy.name][2][0]['depth_coverage']<.6


def test_sparse_wrong_depth_does_not_override_rgb_owner():
    p,c,d=overlapping_scene('4');d[:]=np.nan
    d[::20,::20]=.5
    assignments=_assign_caps(list(CELLS.values()),c,p,OFFSET,0.,INTR,depth=d)
    assert assignments[CELLS['6'].name][0].any()
    assert all(r['source']=='rgb' for r in assignments[CELLS['6'].name][2])


def test_background_depth_outside_all_tower_volumes_falls_back_to_rgb():
    p,c,d=overlapping_scene('4')
    d[:]=.5
    assignments=_assign_caps(list(CELLS.values()),c,p,OFFSET,0.,INTR,depth=d)
    assert assignments[CELLS['6'].name][0].any()
    assert not assignments[CELLS['4'].name][0].any()
    assert all(r['source']=='rgb' and r['depth_status']=='outside_all_tower_volumes'
               for r in assignments[CELLS['6'].name][2])


def test_clipped_but_paired_cap_keeps_unknown_ownership_evidence():
    cell=CELLS['6'];twin=replace(cell,name='重叠塔位')
    p=M['_pose_for_camera'](cell.x,cell.y-.5,0.,OFFSET)
    c=np.zeros((INTR.height,INTR.width,3),np.uint8);d=np.full(c.shape[:2],8.,np.float32)
    paint_cap(c,d,p,cell)
    assert np.any(c[0])  # 顶块上沿出画，红蓝仍可见。
    owned,possible,_=_assign_caps([cell,twin],c,p,OFFSET,0.,INTR)[cell.name]
    diagnostic={'_cap_allowed':owned,'_cap_suspected_allowed':possible}
    z=float(to_camera(cell.center(cell.one_z),p,OFFSET,0.)[2])
    assert _rgb_cap_reading(cell,c,_mask(cell,p,OFFSET,0.,INTR),p,OFFSET,0.,INTR,z,diagnostic=diagnostic) is None
    assert diagnostic['rgb_cap']['state']=='suspected'
