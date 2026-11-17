"""远处RGB与深度分工：未知深度不否决，真实前景和上层占用仍拒绝。"""
import runpy

import numpy as np
import pytest

from classify import inspect_frame,_rgb_white_top,_mask,_white_mask
from car_pose import to_camera

SCENE=runpy.run_path('tests/test_robust_observation.py')
CELL,OFFSET,INTR=SCENE['CELL'],SCENE['OFFSET'],SCENE['INTR']


def local(p,color,depth,info=None):
    return _rgb_white_top(CELL,color,_white_mask(color),_mask(CELL,p,OFFSET,0,INTR),
        p,OFFSET,0,INTR,float(to_camera(CELL.center(CELL.one_z),p,OFFSET,0)[2]),
        depth=depth,diagnostic=info)


@pytest.mark.parametrize('layers',[1,2])
@pytest.mark.parametrize('distance',[3.2,4.2])
@pytest.mark.parametrize('value',[np.nan,8.])
def test_far_visible_top_survives_missing_or_background_depth(layers,distance,value):
    p,color,depth,_,_=SCENE['boxes_scene'](layers,distance)
    depth[:]=value
    result=local(p,color,depth)
    assert result is not None and result[0]==str(layers)
    view=inspect_frame(p,OFFSET,0,INTR,color,depth)[CELL.name]
    assert view.conclusion==str(layers),(view.reason,view.measurements)
    assert view.method in ('RGB局部盒面','RGB盒体边界')


@pytest.mark.parametrize('pattern',['all','alternating_rows'])
def test_coherent_foreground_on_candidate_cannot_be_ignored(pattern):
    p,color,depth,_,_=SCENE['boxes_scene'](2,4.2)
    if pattern=='all':
        depth[:]=2.
    else:
        rows=np.indices(depth.shape)[0]
        depth[(depth<6)&(rows%2==0)]=2.
    assert local(p,color,depth) is None
    view=inspect_frame(p,OFFSET,0,INTR,color,depth)[CELL.name]
    assert view.conclusion is None


def test_sparse_close_bad_points_do_not_invalidate_whole_top():
    p,color,depth,_,_=SCENE['boxes_scene'](2,4.2)
    rows,cols=np.nonzero(depth<6)
    depth[rows[::40],cols[::40]]=1.5
    result=local(p,color,depth)
    assert result is not None and result[0]=='2'


def test_near_white_face_still_requires_consistent_measured_depth():
    p,color,depth,_,_=SCENE['boxes_scene'](2,1.8)
    depth[:]=8.
    assert local(p,color,depth) is None


def test_foreground_below_visible_candidate_does_not_block_tower():
    p,color,depth,render,_=SCENE['boxes_scene'](2,4.2)
    render(CELL.x-.45,CELL.x+.45,CELL.y-1.1,CELL.y-.9,0,.24,(80,80,80))
    view=inspect_frame(p,OFFSET,0,INTR,color,depth)[CELL.name]
    assert view.conclusion=='2',(view.reason,view.measurements)
    info=view.measurements['foreground_check']
    assert info['full_ratio']>.1
    assert info['scope']=='candidate' and not info['occludes']


def test_real_foreground_hiding_the_top_remains_unknown():
    p,color,depth,render,_=SCENE['boxes_scene'](2,4.2)
    render(CELL.x-.45,CELL.x+.45,CELL.y-1.1,CELL.y-.9,0,.85,(80,80,80))
    view=inspect_frame(p,OFFSET,0,INTR,color,depth)[CELL.name]
    assert view.conclusion is None
    assert view.measurements['foreground_check']['occludes']


def test_gray_cap_with_measured_upper_occupation_does_not_become_plain_two():
    p,color,depth,_,_=SCENE['boxes_scene'](2,4.2,cap=True)
    red_blue=np.max(color,axis=2)-np.min(color,axis=2)>100
    color[red_blue]=120
    view=inspect_frame(p,OFFSET,0,INTR,color,depth)[CELL.name]
    assert view.conclusion is None,(view.conclusion,view.measurements)
