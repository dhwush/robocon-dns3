"""窗帘碎片不得破坏顶块；背景深度不冒充本塔或前景。"""
import cv2
import numpy as np
import pytest
from classify import _cap_pair_masks, _cap_colors, _top_depth_support


def test_curtain_fragments_do_not_expand_blue_core():
    image=np.zeros((160,160,3),np.uint8)
    image[70:100,50:120]=(0,0,220)
    image[100:130,50:120]=(220,0,0)
    # 每段之间不相连，但红蓝合并后的7像素闭运算将它们连到红块。
    for y in range(30,70,6):
        image[y:y+3,50:54]=(220,0,0)
    raw_red,raw_blue=_cap_colors(image)
    red,blue=_cap_pair_masks(image)
    assert red.sum()==raw_red.sum()
    assert blue[100:130,50:120].all()
    assert not blue[:70].any()
    assert blue.sum()<raw_blue.sum()


def test_two_substantial_blue_blocks_are_not_silently_reduced_to_one():
    image=np.zeros((100,140,3),np.uint8)
    image[30:60,20:120]=(0,0,220)
    image[60:90,20:65]=(220,0,0)
    image[60:90,70:120]=(220,0,0)
    raw=_cap_colors(image)[1]
    assert np.array_equal(_cap_pair_masks(image)[1],raw)


@pytest.mark.parametrize('other,expected',[(8.,True),(3.,False),(4.08,False)])
def test_top_depth_excludes_only_separated_background(other,expected):
    xs=np.tile(np.arange(40),4)
    own=np.repeat([True,False,True,False],40)
    values=np.where(own,4.,other)
    info={}
    assert _top_depth_support(values,own,xs,np.arange(40),info)==expected
    assert info['depth_support_raw']==.5


def test_small_patch_at_correct_distance_cannot_rescue_background():
    xs=np.tile(np.arange(40),4)
    own=xs<10
    info={}
    assert not _top_depth_support(np.where(own,4.,8.),own,xs,np.arange(40),info)
    assert info['depth_column_coverage']==.25


@pytest.mark.parametrize('replacement,expected',[(8.,True),(2.,False)])
def test_local_top_uses_measured_face_with_background_holes_but_not_foreground(replacement,expected):
    import runpy
    from classify import _rgb_white_top,_white_mask,_mask
    from car_pose import to_camera
    scene=runpy.run_path('tests/test_robust_observation.py')
    cell,offset,intr=scene['CELL'],scene['OFFSET'],scene['INTR']
    p,c,d,_,_=scene['boxes_scene'](2,4.2)
    rows=np.indices(d.shape)[0]
    d[(d<6)&(rows%2==0)]=replacement
    info={}
    result=_rgb_white_top(cell,c,_white_mask(c),_mask(cell,p,offset,0,intr),p,offset,0,intr,
                          float(to_camera(cell.center(cell.one_z),p,offset,0)[2]),diagnostic=info,depth=d)
    assert (result is not None)==expected
    if expected:
        assert result[0]=='2'
        assert any(q.get('depth_background_points',0)>0 for q in info['rgb_white_top']['checks'])
