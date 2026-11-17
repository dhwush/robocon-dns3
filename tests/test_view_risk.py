"""背景策略只限制0/1层，不能抢先禁用两层/顶块的观察机会。"""
import json
import math
import runpy
from pathlib import Path
import cv2
import pytest
import numpy as np
import classify
import motion
from car_pose import CarPose, OPTICAL_FROM_CORNER, load_intrinsics
from field_map import DISPLAY, tower_cells, observation_risk

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'debug_captures/video_170811'

@pytest.mark.parametrize('number',[2,3,4,5,6])
@pytest.mark.parametrize('y',[2.858,3.9639,3.964,4.2])
def test_right_corridor_does_not_restrict_other_towers(number,y):
    assert observation_risk(tower_cells()[number-1],2.2,y) is None

@pytest.mark.parametrize('y,active',[(2.858,False),(3.9639,False),(3.964,True),(4.2,True)])
def test_only_tower_one_above_six_midline(y,active):
    assert bool(observation_risk(tower_cells()[0],2.2,y)) == active
    assert observation_risk(tower_cells()[0],1.5,y) is None

@pytest.mark.parametrize('mode',['rgb','missing','background'])
@pytest.mark.parametrize('second,tower',[(3,6),(163,1)])
def test_recorded_frame_follows_narrowed_policy(second,tower,mode):
    case=next(c for c in json.loads((HERE/'cases.json').read_text()) if c['second']==second)
    x,y,yaw,pitch=case['pose']
    pose=CarPose(x,y,math.radians(yaw))
    intr=load_intrinsics(str(ROOT/'config/intrinsics.yaml'))
    color=cv2.imread(str(HERE/f'{second}.png'))
    depth=None if mode=='rgb' else np.full(color.shape[:2],np.nan if mode=='missing' else 8.,np.float32)
    view=classify.inspect_frame(pose,OPTICAL_FROM_CORNER,pitch,intr,color,depth,0)[DISPLAY[tower-1]]
    risk=view.measurements.get('view_risk')
    if tower==1:
        assert risk is None  # 旧纸板规则不能在6号中线以下触发。
    else:
        assert risk and view.conclusion != '1'

S=runpy.run_path(str(ROOT/'tests/test_robust_observation.py'))

@pytest.mark.parametrize('layers,cap',[(0,False),(1,False),(2,False),(2,True)])
@pytest.mark.parametrize('with_depth',[False,True])
def test_risk_preserves_two_layers_and_cap_but_rejects_low_tower(monkeypatch,layers,cap,with_depth):
    p,c,d,_,_=S['boxes_scene'](layers,2.5,cap=cap)
    args=(p,S['OFFSET'],0.,S['INTR'],c,d if with_depth else None)
    before=classify.inspect_frame(*args)[S['CELL'].name]
    # 同一画面固定识别证据，仅切换区域风险，隔离策略效果。
    monkeypatch.setattr(classify,'observation_risk',lambda *args:'底线左侧看6号：铁皮柜背景风险')
    after=classify.inspect_frame(*args)[S['CELL'].name]
    if before.conclusion == '1':
        assert after.conclusion is None and after.skip_vote and after.needs_parallax
    else:
        assert after.conclusion == before.conclusion
        assert after.method == before.method
        assert after.needs_parallax == before.needs_parallax
    if layers==2:
        assert before.conclusion == ('红' if cap else '2')
        assert not after.measurements['view_risk']['rejected']

def test_planner_retains_risky_observation_opportunities(monkeypatch):
    cell=tower_cells()[5]
    p=CarPose(0,0,0)
    offset=np.zeros(2)
    intr=load_intrinsics(str(ROOT/'config/intrinsics.yaml'))
    before=list(motion.observation_poses(cell,p,offset,intr,True))
    monkeypatch.setattr(motion,'observation_risk',lambda *args:None)
    quality=motion.observation_quality(cell,p,offset,intr)
    monkeypatch.setattr(motion,'observation_risk',lambda *args:'风险')
    after=list(motion.observation_poses(cell,p,offset,intr,True))
    assert before and [(a,b.yaw,c) for a,b,c in after]==[(a,b.yaw,c) for a,b,c in before]
    assert quality > 0
    assert motion.observation_quality(cell,p,offset,intr)==pytest.approx(.8*quality)


def test_new_video_lower_corridor_restores_original_judgment():
    replay=runpy.run_path(str(ROOT/'debug_captures/video_191715/replay.py'))
    assert len(replay['run']())==3


@pytest.mark.parametrize('x,active',[(-.3,False),(0.,False),(.001,True),(.5,True),(.5849,True),(.585,False),(.7,False)])
def test_bottom_six_only_between_origin_and_green_base(x,active):
    assert bool(observation_risk(tower_cells()[5],x,0.)) == active
    assert observation_risk(tower_cells()[5],x,.855) is None
