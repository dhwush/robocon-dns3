from types import SimpleNamespace

from car_pose import CarPose, Setup, load_intrinsics
from static_test import StaticFieldWatch, run_static
from view_field import ROOT
import numpy as np


def test_static_entry_keeps_radar_source(monkeypatch):
    def once(args, watch, pose):
        assert args.pose_source == 'radar'
        assert pose == CarPose(0., 0., 0.)
        assert watch.setup.origin_basic and watch.setup.topic == ''
        assert watch.autonomy is None
        assert watch.pitch == -8.
        return ['ok']
    monkeypatch.setattr('view_field._once', once)
    args = SimpleNamespace(pitch=-8., image='', once=True, mode='static',
                           pose_source='radar', pose=(3., 4., 90.))
    assert run_static(args) == ['ok']


def test_static_never_uses_motion_planner(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError('static mode called motion planner')
    monkeypatch.setattr('report.plan_action', fail)
    watch = StaticFieldWatch(Setup(False, np.zeros(2), 0., '', 'rad', True),
                             load_intrinsics(str(ROOT / 'config/intrinsics.yaml')))
    watch.update(CarPose(0., 0., 0.))
    watch._note_aim()
    assert watch.required_pitch() is None
    assert '不设通过门槛' in '\n'.join(watch.window_lines())
    assert all(not watch.views[name].needs_yaw for name in watch.targets())
    assert watch.autonomy is None


def test_basic_routes_to_fixed_camera_mode(monkeypatch):
    from view_field import run_basic
    args = SimpleNamespace()
    monkeypatch.setattr('static_test.run_static', lambda received: ['fixed'] if received is args else [])
    assert run_basic(args) == ['fixed']
