"""近处可靠实测深度优先；RGB不清与顶部深度缺失分开处理。"""
import runpy
import numpy as np
import pytest
from classify import inspect_frame, _near_depth_top, _white_cloud
from car_pose import camera_position

S=runpy.run_path('tests/test_robust_observation.py')
CELL,OFFSET,INTR=S['CELL'],S['OFFSET'],S['INTR']


@pytest.mark.parametrize('layers',[1,2])
def test_valid_near_depth_ignores_merged_rgb_outline(layers):
    p,c,d,_,_=S['boxes_scene'](layers,1.8)
    c[:]=160  # RGB中盒子和背景完全同色，轮廓不存在。
    v=inspect_frame(p,OFFSET,0,INTR,c,d)[CELL.name]
    assert v.conclusion==str(layers),(v.reason,v.measurements)
    assert v.method=='深度'
    assert v.measurements['depth_primary']


@pytest.mark.parametrize('layers',[1,2])
def test_partial_unknown_top_columns_do_not_veto_remaining_depth(layers):
    p,c,d,_,_=S['boxes_scene'](layers,1.8)
    c[:]=160
    body=d<6
    yy,xx=np.nonzero(body)
    cols=np.unique(xx)
    # 约40%的盒宽上方深度缺失；剩余60%为分布连续的真实顶边支撑。
    for u in cols[:int(.4*len(cols))]:
        first=int(np.nonzero(body[:,u])[0][0])
        d[max(0,first-5):first,u-1:u+2]=np.nan
    v=inspect_frame(p,OFFSET,0,INTR,c,d)[CELL.name]
    assert v.conclusion==str(layers),(v.reason,v.measurements)
    assert v.measurements['depth_primary']
    assert v.measurements['depth_top_check']['unknown_columns']>0


def test_rgb_cannot_cover_missing_top_of_true_two_layer_depth():
    p,c,d,_,rays=S['boxes_scene'](2,1.8)
    c[:]=160
    heights=camera_position(p,OFFSET,0)[2]+d*rays[...,2]
    d[(d<6)&(heights>.35)]=np.nan
    v=inspect_frame(p,OFFSET,0,INTR,c,d)[CELL.name]
    assert v.conclusion is None
    assert not v.measurements.get('depth_primary')


def test_bottom_foreground_does_not_override_reliable_upper_tower_depth():
    p,c,d,render,_=S['boxes_scene'](2,1.8)
    render(CELL.x-.45,CELL.x+.45,CELL.y-.8,CELL.y-.6,.05,.25,(65,65,65))
    v=inspect_frame(p,OFFSET,0,INTR,c,d)[CELL.name]
    assert v.conclusion=='2',(v.reason,v.measurements)
    assert v.method=='深度'
