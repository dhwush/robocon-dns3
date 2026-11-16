"""近背景、缺测和底部遮挡不能等同于真实上层占用。"""
import runpy
import numpy as np
import pytest
from classify import inspect_frame, _upper_white_reading, _white_mask, _mask, _upper_space_clear
from car_pose import camera_position

S=runpy.run_path('tests/test_robust_observation.py')
CELL,OFFSET,INTR=S['CELL'],S['OFFSET'],S['INTR']


@pytest.mark.parametrize('ink',[(65,65,65),(220,220,220)])
@pytest.mark.parametrize('layers',[1,2])
def test_back_margin_clutter_is_not_attached_upper_object(ink,layers):
    p,c,d,render,_=S['boxes_scene'](layers,4.2)
    # 后方杂物在定位搜索余量里，但与白盒正面没有空间连接。
    render(CELL.x-.3,CELL.x+.3,CELL.y+.29,CELL.y+.31,.40,1.0,ink)
    view=inspect_frame(p,OFFSET,0,INTR,c,d)[CELL.name]
    assert not view.measurements.get('upper_object')
    assert view.conclusion==str(layers)


@pytest.mark.parametrize('layers',[1,2])
def test_close_background_and_partial_missing_depth_allow_free_space(layers):
    p,c,d,render,_=S['boxes_scene'](layers,4.2)
    # 背景只在盒后约20cm，无须原先强求的40cm间隙。
    render(CELL.x-1,CELL.x+1,CELL.y+.29,CELL.y+.31,0,1.2,(150,150,150))
    assert _upper_space_clear(CELL,CELL.one_z if layers==1 else CELL.two_z,p,OFFSET,0,INTR,d)
    d[:,::4]=np.nan
    assert _upper_space_clear(CELL,CELL.one_z if layers==1 else CELL.two_z,p,OFFSET,0,INTR,d)
    assert not _upper_space_clear(CELL,CELL.one_z,p,OFFSET,0,INTR,np.full_like(d,np.nan))


@pytest.mark.parametrize('layers',[1,2])
def test_upper_body_can_classify_both_layer_counts_with_hidden_bottom(layers):
    p,c,d,_,rays=S['boxes_scene'](layers,4.2)
    origin=camera_position(p,OFFSET,0)
    heights=origin[2]+d*rays[...,2]
    top=CELL.one_z if layers==1 else CELL.two_z
    d[(d<6)&(heights<top-.12)]=1.0
    diag={}
    result=_upper_white_reading(CELL,c,_white_mask(c),d,_mask(CELL,p,OFFSET,0,INTR),p,OFFSET,0,INTR,diagnostic=diag)
    assert result is not None,diag
    assert result[0]==str(layers)
