"""中距离外沿间距、前后塔投影分离及兜底选择。"""
import math
from dataclasses import replace

import numpy as np

import motion
from car_pose import CarPose, load_intrinsics
from field_map import DISPLAY, tower_cells

INTR = load_intrinsics('config/intrinsics.yaml')
OFFSET = np.zeros(2)


def camera_at(x,y):
    return CarPose(x+.211,y-.05,0.)


def test_clearance_uses_nearest_tower_edge_instead_of_center(monkeypatch):
    cell = replace(tower_cells()[0],name='target',x=0.,y=2.)
    monkeypatch.setattr(motion,'_cells',lambda:{'target':cell})
    near = camera_at(0.,.65)  # 中心距1.35m，实际外沿距只有1.10m。
    boundary = camera_at(0.,.55)
    assert math.isclose(motion.tower_clearance(cell,near,OFFSET),1.10)
    assert not motion._observation_preferences(near,OFFSET,0.,INTR)['target'][0]
    assert motion._observation_preferences(boundary,OFFSET,0.,INTR)['target'][0]


def test_far_tower_alignment_prefers_a_side_view(monkeypatch):
    front = replace(tower_cells()[0],name='front',x=0.,y=2.)
    far = replace(front,name='far',y=4.5)
    monkeypatch.setattr(motion,'_cells',lambda:{'front':front,'far':far})
    aligned = motion._observation_preferences(camera_at(0.,0.),OFFSET,0.,INTR)
    side = motion._observation_preferences(camera_at(1.5,0.),OFFSET,0.,INTR)
    assert aligned['far'][3] > motion.MAX_PROJECTION_OVERLAP
    assert side['far'][3] < .1
    assert aligned['front'][3] == 0.  # 远方背景不会反过来否决前景塔。
    assert aligned['far'][1] > side['far'][1]


def test_candidate_ring_starts_beyond_1_2m_even_on_diagonal():
    cell = tower_cells()[0]
    for distance in motion._distances(cell,INTR):
        for degrees in range(0,360,45):
            angle = math.radians(degrees)
            pose = camera_at(cell.x+distance*math.cos(angle),cell.y+distance*math.sin(angle))
            assert motion.tower_clearance(cell,pose,OFFSET) >= 1.2-1e-6


def test_one_middle_distance_target_beats_two_unreliable_targets(monkeypatch):
    names = list(DISPLAY[:2])
    monkeypatch.setattr(motion,'_distances',lambda *args:[])
    monkeypatch.setattr(motion,'_parking_clear',lambda x,y,*args: abs(y)<1e-8 and .09<x<.21)
    monkeypatch.setattr(motion,'_ground_blocked',lambda *args:False)
    monkeypatch.setattr(motion,'_bypass_points',lambda *args:[])
    monkeypatch.setattr(motion,'_travel',lambda x,y,a,b,*args:(math.hypot(a-x,b-y),[(x,y),(a,b)]))
    monkeypatch.setattr(motion,'observation_poses',lambda c,p,*args: iter([(0,p,0.)]))
    def preferences(p,*args):
        return {n:(p.x>.15 and n==names[0], .8, .5, 0.) for n in DISPLAY}
    monkeypatch.setattr(motion,'_observation_preferences',preferences)
    action = motion.nearest_observation(CarPose(0,0,0),OFFSET,None,names)
    assert math.isclose(action.final_x,.2)
    assert action.cells == (names[0],)
    assert not action.observation_fallback
    monkeypatch.setattr(motion,'_observation_preferences',
                        lambda p,*args:{n:(False,.9 if p.x<.15 else .2,.5,0.) for n in DISPLAY})
    fallback = motion.nearest_observation(CarPose(0,0,0),OFFSET,None,names)
    assert math.isclose(fallback.final_x,.2)
    assert fallback.observation_fallback
