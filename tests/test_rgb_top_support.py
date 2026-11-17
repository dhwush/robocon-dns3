"""局部盒面以顶边/侧边识别，不把接缝、横线或深度背景当塔。"""
import runpy
import numpy as np
import pytest
from classify import _rgb_white_top,_white_mask,_mask
from car_pose import to_camera

S=runpy.run_path('tests/test_robust_observation.py')
CELL,OFFSET,INTR=S['CELL'],S['OFFSET'],S['INTR']


def detect(p,c,d=None):
    return _rgb_white_top(CELL,c,_white_mask(c),_mask(CELL,p,OFFSET,0,INTR),
        p,OFFSET,0,INTR,float(to_camera(CELL.center(CELL.one_z),p,OFFSET,0)[2]),depth=d)


@pytest.mark.parametrize('layers',[1,2])
def test_top_and_sides_allow_bottom_occlusion(layers):
    p,c,_,_,_=S['boxes_scene'](layers,4.2)
    yy,xx=np.nonzero(np.all(c==220,axis=2))
    top,bottom=int(yy.min()),int(yy.max())
    c[top+(bottom-top)//2:bottom+1,xx.min()-4:xx.max()+5]=(50,50,50)
    result=detect(p,c)
    assert result is not None
    assert result[0]==str(layers)


@pytest.mark.parametrize('kind',['wall','line','missing_top','foreground_depth'])
def test_insufficient_or_contradictory_evidence_remains_unknown(kind):
    p,c,d,_,_=S['boxes_scene'](2,4.2)
    yy,xx=np.nonzero(np.all(c==220,axis=2))
    if kind=='wall':
        c[:]=220
    elif kind=='line':
        c[:]=220
        c[int(yy.min()):int(yy.min())+2]=100
    elif kind=='missing_top':
        c[:int(yy.min()),int(xx.min()):int(xx.max())+1]=220
    else:
        d[:]=2.  # 连续近处实测面才是反证，8m背景回波不再否决远处RGB。
    assert detect(p,c,d if kind=='foreground_depth' else None) is None


def test_internal_seam_cannot_turn_two_layers_into_one():
    p,c,_,_,_=S['boxes_scene'](2,4.2)
    # 用真实投影的一层高度画接缝；上方白面仍连续。
    from car_pose import project
    u,v=project(CELL.center(CELL.one_z),p,OFFSET,0,INTR)
    yy,xx=np.nonzero(np.all(c==220,axis=2))
    c[int(v):int(v)+2,int(xx.min()):int(xx.max())+1]=100
    result=detect(p,c)
    assert result is None or result[0]=='2'


def test_local_rgb_keeps_single_frame_result_and_records_history():
    from report import stabilize_local_rgb
    from types import SimpleNamespace
    tracks={}
    piece=np.array([[[10,10]],[[40,10]],[[40,25]],[[10,25]]],np.int32)
    def view(state='1'):
        return SimpleNamespace(method='RGB局部盒面',conclusion=state,piece=piece,
                               measurements={},skip_vote=False,reason=None)
    for count in [1,2,3]:
        v=view();stabilize_local_rgb({'tower':v},tracks)
        assert v.conclusion=='1' and not v.skip_vote and v.reason is None
        assert v.measurements['rgb_local_track']['frames']==count
    v=view('2');stabilize_local_rgb({'tower':v},tracks)
    assert v.conclusion=='2'
    assert v.measurements['rgb_local_track']['frames']==1
    v=view(None);stabilize_local_rgb({'tower':v},tracks)
    assert not tracks


def test_elevated_visible_body_cannot_downgrade_absolute_two_to_one():
    p,c,d,render,_=S['boxes_scene'](0,4.2)
    render(CELL.x-.175,CELL.x+.175,CELL.y-.175,CELL.y+.175,.30,.65,(220,220,220))
    result=detect(p,c,d)
    assert result is None or result[0]=='2'


@pytest.mark.parametrize('layers',[1,2])
def test_bright_foreground_with_continuing_sides_cannot_shorten_tower(layers):
    p,c,_,_,_=S['boxes_scene'](layers,4.2)
    yy,xx=np.nonzero(np.all(c==220,axis=2))
    # 横纹仍在两侧边之间，不是完整盒面的底边。
    row=int(yy.min()+(yy.max()-yy.min())/2)
    c[row:row+8,int(xx.min())+5:int(xx.max())-4]=140
    result=detect(p,c)
    assert result is None or result[0]==str(layers)
