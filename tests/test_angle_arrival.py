import math
from types import SimpleNamespace
import pytest
from car_pose import CarPose
from motion import Action
from report import FieldWatch


@pytest.mark.parametrize('axis', ['pitch','yaw'])
@pytest.mark.parametrize('error,expected', [(2.99,True),(3.,True),(3.01,False),(-3.,True),(-3.01,False)])
def test_all_task3_angle_arrival_is_inclusive_three_degrees(axis,error,expected):
    watch=SimpleNamespace(pitch=error,last_pose=CarPose(0,0,math.radians(error)))
    action=Action(axis,'',pitch=0.,yaw=0.)
    assert FieldWatch._arrived(watch,action) is expected


def test_non_autonomous_arrival_explicitly_stops_before_next_action():
    watch=SimpleNamespace(pitch=3.,last_pose=CarPose(0,0,0),hold=Action('pitch','低头',pitch=0.))
    watch._arrived=lambda action:FieldWatch._arrived(watch,action)
    text=FieldWatch._held(watch,Action('yaw','向左转',yaw=1.))
    assert '可以了' in text and '停止' in text
    assert watch.hold is None


@pytest.mark.parametrize('raw,expected', [(-318.8,41.2),(361.,1.),(-540.,-180.),(180.,-180.),(720.,0.)])
def test_yaw_display_is_normalized_without_changing_pose(raw,expected):
    from car_pose import display_yaw_deg
    from terminal_ui import _pose_text
    pose=CarPose(0,0,math.radians(raw))
    assert display_yaw_deg(pose.yaw)==pytest.approx(expected)
    assert f'yaw={expected:+.1f}°' in _pose_text(pose,0.)
    assert pose.yaw==math.radians(raw)
