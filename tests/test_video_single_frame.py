"""录像RGB抽帧同时校验层数和识别框，禁止把背景碰巧判对计为成功。"""
from pathlib import Path
import math

import cv2
import pytest

from car_pose import CarPose,OPTICAL_FROM_CORNER,load_intrinsics
from classify import inspect_frame
from field_map import DISPLAY

ROOT=Path(__file__).resolve().parents[1]
FRAMES=ROOT/'debug_captures/rgb_validation_20261116'

@pytest.mark.parametrize('second,pose,expected',[
    (29,(-.010,.021,-.5,-1),{3:'1'}),
    (133,(-.248,.191,-46.9,-1),{4:'1',5:None,6:'1'}),
    (135,(-.243,.195,-46.9,0),{4:'1',5:'1',6:'1'}),
])
def test_recorded_frame_result_and_target_location(second,pose,expected):
    source=FRAMES/f'rgb_{second}.png'
    if not source.exists():
        pytest.skip('本机录像RGB验证样本未附带')
    color=cv2.imread(str(source))
    x,y,yaw,pitch=pose
    views=inspect_frame(CarPose(x,y,math.radians(yaw)),OPTICAL_FROM_CORNER,pitch,
                        load_intrinsics(str(ROOT/'config/intrinsics.yaml')),color,None,0)
    rois={3:(290,390,410,465),4:(555,410,820,590),5:(905,460,1110,590),6:(455,360,540,445)}
    for number,state in expected.items():
        view=views[DISPLAY[number-1]]
        assert view.conclusion==state,(number,view.reason)
        if state is None:
            assert view.measurements['rgb_contour_rejected']=='1'
            continue
        assert view.piece is not None
        x,y,w,h=cv2.boundingRect(view.piece)
        a,b,c,d=rois[number]
        intersection=max(0,min(x+w,c)-max(x,a))*max(0,min(y+h,d)-max(y,b))
        assert intersection/(w*h)>.5,(number,(x,y,w,h),rois[number])
