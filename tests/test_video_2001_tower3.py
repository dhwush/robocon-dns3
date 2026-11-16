"""录屏RGB复算：不把HUD取整后的姿态或绘制轮廓当作原始传感器数据。"""
import math
from pathlib import Path
from types import SimpleNamespace
import runpy

import cv2
import numpy as np
import pytest

from car_pose import CarPose,OPTICAL_FROM_CORNER,load_intrinsics
from classify import inspect_frame,_white_face_continues_above,_white_mask,_mask
from field_map import DISPLAY
from report import stabilize_local_rgb

ROOT=Path(__file__).resolve().parents[1]
CAPTURES=ROOT/'debug_captures/rgb_validation_20261116_2001'
POSES={38:(0,.042,.4,-1),42:(-.063,.068,22.5,-1),
       45:(-.075,.069,28.8,0),46:(-.088,.059,34,-1)}


@pytest.mark.parametrize('time',POSES)
@pytest.mark.parametrize('pitch_error',[-.5,0,.5])
def test_visible_video_top_stays_two_with_small_pitch_error(time,pitch_error):
    x,y,yaw,pitch=POSES[time]
    color=cv2.imread(str(CAPTURES/f'rgb_{time}.png'))
    assert color is not None
    view=inspect_frame(CarPose(x,y,math.radians(yaw)),OPTICAL_FROM_CORNER,
        pitch+pitch_error,load_intrinsics(str(ROOT/'config/intrinsics.yaml')),
        color,None,0)[DISPLAY[2]]
    assert view.conclusion=='2',(view.reason,view.measurements)
    # 框须落在真实上层顶边，不能仅把下方接缝改标签为两层。
    assert cv2.boundingRect(view.piece)[1]<340


def test_painted_over_video_top_does_not_make_seam_a_single_box():
    color=cv2.imread(str(CAPTURES/'rgb_35.png'))
    view=inspect_frame(CarPose(-.007,.039,math.radians(.4)),OPTICAL_FROM_CORNER,
        -1,load_intrinsics(str(ROOT/'config/intrinsics.yaml')),color,None,0)[DISPLAY[2]]
    assert view.conclusion is None
    assert view.measurements['rgb_single_seam_rejected']


def test_row_scanning_does_not_join_clutter_to_one_side_of_tower():
    color=cv2.imread(str(CAPTURES/'rgb_155.png'))
    view=inspect_frame(CarPose(-1.857,.353,math.radians(-8.2)),OPTICAL_FROM_CORNER,
        -1,load_intrinsics(str(ROOT/'config/intrinsics.yaml')),color,None,0)[DISPLAY[2]]
    assert view.conclusion is None


@pytest.mark.parametrize('layers',[1,2])
def test_real_horizontal_edges_work_when_hough_misses_them(monkeypatch,layers):
    from classify import _rgb_white_top
    from car_pose import to_camera
    scene=runpy.run_path(str(ROOT/'tests/test_robust_observation.py'))
    p,color,_,_,_=scene['boxes_scene'](layers,4.2)
    cell,offset,intr=scene['CELL'],scene['OFFSET'],scene['INTR']
    monkeypatch.setattr(cv2,'HoughLinesP',lambda *a,**kw:None)
    result=_rgb_white_top(cell,color,_white_mask(color),_mask(cell,p,offset,0,intr),
        p,offset,0,intr,float(to_camera(cell.center(cell.one_z),p,offset,0)[2]))
    assert result is not None and result[0]==str(layers)


def make_view(dx=0,state='2'):
    piece=np.array([[[10+dx,10]],[[50+dx,10]],[[50+dx,30]],[[10+dx,30]]],np.int32)
    return SimpleNamespace(method='RGB局部盒面',conclusion=state,piece=piece,
                           measurements={},skip_vote=False,reason=None)


def test_display_smoothing_preserves_current_detection_and_clears_unknown():
    tracks={}
    first=make_view();stabilize_local_rgb({'3':first},tracks)
    current=make_view(2);raw=current.piece.copy()
    stabilize_local_rgb({'3':current},tracks)
    assert np.array_equal(current.piece,raw)
    assert np.all(current.measurements['display_piece'][:,:,0]==raw[:,:,0]-1)
    assert current.conclusion=='2' and not current.skip_vote
    current.conclusion=None
    stabilize_local_rgb({'3':current},tracks)
    assert not tracks and 'display_piece' not in current.measurements
    changed=make_view(0,'1');stabilize_local_rgb({'3':changed},tracks)
    assert changed.conclusion=='1' and tracks['3']['count']==1


@pytest.mark.parametrize('camera',[(.02,0,0,0,0),(0,0,.02,0,0),
                                   (0,0,0,1,0),(0,0,0,0,1)])
def test_camera_motion_resets_display_smoothing(camera):
    tracks={}
    stabilize_local_rgb({'3':make_view()},tracks,(0,0,0,0,0))
    current=make_view(2)
    stabilize_local_rgb({'3':current},tracks,camera)
    assert tracks['3']['count']==1
    assert np.array_equal(current.measurements['display_piece'],current.piece)


def test_white_background_without_two_continuing_sides_keeps_single_box():
    scene=runpy.run_path(str(ROOT/'tests/test_robust_observation.py'))
    p,color,_,_,_=scene['boxes_scene'](1,4.2)
    cell,offset,intr=scene['CELL'],scene['OFFSET'],scene['INTR']
    from car_pose import project,to_camera
    from classify import _rgb_white_top
    z=float(to_camera(cell.center(cell.one_z),p,offset,0)[2])
    mask=_mask(cell,p,offset,0,intr)
    result=_rgb_white_top(cell,color,_white_mask(color),mask,p,offset,0,intr,z)
    assert result is not None and result[0]=='1'
    _,row=project(cell.center(cell.one_z),p,offset,0,intr)
    color[:int(row)-4]=220
    assert not _white_face_continues_above(result[1],color,_white_mask(color),mask,z,intr)
