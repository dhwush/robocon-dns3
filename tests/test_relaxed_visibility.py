"""部分可见可判读，但缺测不伪称空域，背景底边不改变层数。"""
import runpy
import numpy as np
import pytest
from classify import (_rgb_cap_reading, _rgb_white_edges, _mask, _white_mask,
                      _rgb_upper_consistent, _upper_space_clear)
from car_pose import to_camera
S=runpy.run_path('tests/test_robust_observation.py')
CELL,OFFSET,INTR=S['CELL'],S['OFFSET'],S['INTR']

@pytest.mark.parametrize('layers',[1,2])
def test_half_face_retains_layer_count(layers):
    p,c,_,_,_=S['boxes_scene'](layers,4.2)
    yy,xx=np.nonzero(np.all(c==220,axis=2))
    c[yy.min():yy.max()+1,xx.min():(xx.min()+xx.max())//2]=40
    z=float(to_camera(CELL.center(CELL.one_z),p,OFFSET,0)[2])
    r=_rgb_white_edges(CELL,c,_white_mask(c),_mask(CELL,p,OFFSET,0,INTR),p,OFFSET,0,INTR,z)
    assert r is not None and r[0]==str(layers)


def test_partial_roi_keeps_visible_red_blue_pair():
    p,c,_,_,_=S['boxes_scene'](2,2.5,cap=True)
    colored=(c[:,:,0]!=c[:,:,2])
    yy,xx=np.nonzero(colored)
    mask=np.zeros(c.shape[:2],np.uint8)
    mask[:,int(np.median(xx)):]=255
    z=float(to_camera(CELL.center(CELL.one_z),p,OFFSET,0)[2])
    result=_rgb_cap_reading(CELL,c,mask,p,OFFSET,0,INTR,z)
    assert result is not None and result[0]=='红'


def test_unknown_upper_depth_is_not_false_occupation_for_rgb():
    p,_,d,_,_=S['boxes_scene'](1,4.2)
    d[:]=np.nan
    info={}
    assert not _upper_space_clear(CELL,CELL.one_z,p,OFFSET,0,INTR,d,diagnostic=info)
    assert info['state']=='unknown'
    assert _rgb_upper_consistent(CELL,CELL.one_z,p,OFFSET,0,INTR,d,0,info)
    assert info['state']=='unknown'


def test_actual_second_layer_still_refutes_single_layer_rgb():
    p,_,d,_,_=S['boxes_scene'](2,4.2)
    info={}
    assert not _rgb_upper_consistent(CELL,CELL.one_z,p,OFFSET,0,INTR,d,0,info)
    assert info['state']=='occupied'
