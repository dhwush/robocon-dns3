"""同一录屏帧的姿态扰动不得让真实盒顶退化成接缝；同时核对框的位置。"""
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from car_pose import CarPose,OPTICAL_FROM_CORNER,load_intrinsics
from classify import _merge_white_top_lines,inspect_frame
from field_map import DISPLAY

ROOT=Path(__file__).resolve().parents[1]
FRAMES=ROOT/'debug_captures/rgb_validation_20261116_2001'
INTR=load_intrinsics(str(ROOT/'config/intrinsics.yaml'))


@pytest.mark.parametrize('second,pose,roi',[
    (38,(0,.042,.4,-1),(315,295,440,345)),
    (42,(-.063,.068,22.5,-1),(685,295,785,345)),
    (45,(-.075,.069,28.8,0),(790,295,915,345)),
    (46,(-.088,.059,34,-1),(890,290,1020,345)),
])
@pytest.mark.parametrize('pitch_delta',[-.5,0,.5])
def test_recorded_tower3_top_survives_pitch_error(second,pose,roi,pitch_delta):
    color=cv2.imread(str(FRAMES/f'rgb_{second}.png'))
    assert color is not None
    x,y,yaw,pitch=pose
    view=inspect_frame(CarPose(x,y,math.radians(yaw)),OPTICAL_FROM_CORNER,
                       pitch+pitch_delta,INTR,color,None,0)[DISPLAY[2]]
    assert view.conclusion=='2',(view.reason,view.measurements)
    x,y,w,h=cv2.boundingRect(view.piece)
    a,b,c,d=roi
    intersection=max(0,min(x+w,c)-max(x,a))*max(0,min(y+h,d)-max(y,b))
    assert intersection/(w*h)>.85,(x,y,w,h)


def test_obscured_recorded_tower3_does_not_report_seam_as_one_layer():
    color=cv2.imread(str(FRAMES/'rgb_155.png'))
    view=inspect_frame(CarPose(-1.857,.353,math.radians(-8.2)),OPTICAL_FROM_CORNER,
                       -1,INTR,color,None,0)[DISPLAY[2]]
    assert view.conclusion in (None,'2'),(view.reason,view.measurements)


def test_merge_top_fragments_does_not_join_different_heights_or_distant_objects():
    lines=np.array([[100,100,130,100],[134,101,170,101],
                    [134,120,170,120],[240,100,280,100]])
    result=_merge_white_top_lines(lines,4.,INTR)
    assert any(tuple(q)==(100,100,170,101) for q in result)
    assert not any(q[0]==100 and (q[2]>=240 or q[3]>=120) for q in result)


def test_merge_rejects_overwide_background_line():
    lines=np.array([[100,100,180,100],[184,100,280,100]])
    result=_merge_white_top_lines(lines,4.,INTR)
    assert len(result)==2
