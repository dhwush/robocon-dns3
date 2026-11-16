"""真实盒体与后方白标签不能合并成一个最高深度切片。"""
import runpy
import numpy as np
import pytest
from classify import inspect_frame
from car_pose import camera_position

S=runpy.run_path('tests/test_robust_observation.py')
CELL,OFFSET,INTR=S['CELL'],S['OFFSET'],S['INTR']


@pytest.mark.parametrize('layers',[1,2])
@pytest.mark.parametrize('background',['label','large_board'])
def test_background_high_white_points_do_not_steal_body_top(layers,background):
    p,c,d,render,_=S['boxes_scene'](layers,2.5)
    if background=='label':
        render(CELL.x-.16,CELL.x+.16,CELL.y+.28,CELL.y+.30,.88,.98,(220,220,220))
    else:
        render(CELL.x-.24,CELL.x+.24,CELL.y+.28,CELL.y+.30,.12,.98,(220,220,220))
    v=inspect_frame(p,OFFSET,0,INTR,c,d)[CELL.name]
    assert v.conclusion==str(layers),(v.reason,v.measurements)
    assert v.measurements['near_body']['excluded_points']>0
    expected=CELL.one_z if layers==1 else CELL.two_z
    assert abs(v.measurements['depth_height_cm']/100-expected)<.04


def test_missing_true_top_cannot_be_replaced_by_background_label():
    p,c,d,render,rays=S['boxes_scene'](2,2.5)
    render(CELL.x-.16,CELL.x+.16,CELL.y+.28,CELL.y+.30,.88,.98,(220,220,220))
    heights=camera_position(p,OFFSET,0)[2]+d*rays[...,2]
    # 只删掉前方盒体上半部；后方高标签保留。
    d[(d<2.6)&(heights>.35)]=np.nan
    v=inspect_frame(p,OFFSET,0,INTR,c,d)[CELL.name]
    assert v.conclusion is None,(v.reason,v.measurements)
    assert v.measurements['near_body']['excluded_points']>0
    assert v.measurements['depth_height_cm']<40
