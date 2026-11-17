"""真实录屏RGB+受控深度分支测试，不将合成深度称为现场测量。"""
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from classify import inspect_frame,_rgb_white_top,_mask,_white_mask
from car_pose import CarPose,OPTICAL_FROM_CORNER,load_intrinsics,to_camera
from field_map import tower_cells,DISPLAY

ROOT=Path(__file__).resolve().parents[1]
FRAMES=ROOT/'debug_captures/far_white_20261117'
CASES=json.loads((FRAMES/'cases.json').read_text())


@pytest.mark.parametrize('case',CASES,ids=lambda c:c['id'])
@pytest.mark.parametrize('mode',['rgb','missing','background','foreground'])
def test_recorded_far_tower_uses_real_top_and_only_reliable_depth_veto(case,mode):
    color=cv2.imread(str(FRAMES/(case['id']+'.png')))
    assert color is not None
    depth=None if mode=='rgb' else np.full(color.shape[:2],
        {'missing':np.nan,'background':8.,'foreground':2.}[mode],np.float32)
    x,y,yaw,pitch=case['pose']
    pose=CarPose(x,y,math.radians(yaw))
    intr=load_intrinsics(str(ROOT/'config/intrinsics.yaml'))
    cell=tower_cells()[2]
    result=_rgb_white_top(cell,color,_white_mask(color),_mask(cell,pose,OPTICAL_FROM_CORNER,pitch,intr),
        pose,OPTICAL_FROM_CORNER,pitch,intr,
        float(to_camera(cell.center(cell.one_z),pose,OPTICAL_FROM_CORNER,pitch)[2]),depth=depth)
    view=inspect_frame(pose,OPTICAL_FROM_CORNER,pitch,intr,color,depth,0)[DISPLAY[2]]
    if mode=='foreground':
        assert result is None
        assert view.conclusion is None,(view.reason,view.measurements)
    else:
        assert result is not None and result[0]=='2'
        assert view.conclusion=='2',(view.reason,view.measurements)
        assert view.method=='RGB局部盒面'
        # 校验框位于真正盒顶，而不是给下方接缝贴上两层标签。
        y=cv2.boundingRect(view.piece)[1]
        expected=330 if case['id'].startswith('a') or case['id']=='b30' else 485
        assert abs(y-expected)<12,(case['id'],y)
