import math
from dataclasses import replace

import pytest

from camera_stability import CameraStability
from car_pose import CarPose


def sample(gate,t,pose=None,pitch=0.,roll=0.,valid=True):
    return gate.update(pose or CarPose(0.,0.,0.),pitch,roll,t,valid)


def test_requires_elapsed_time_and_several_samples_not_just_fast_frames():
    gate=CameraStability()
    for t in (0.,.02,.04,.06,.1,.2,.4):
        assert not sample(gate,t)
    assert sample(gate,.6)
    gate.reset()
    assert not sample(gate,0.)
    assert not sample(gate,.6)  # 单次间断超过0.5秒，不能当连续停稳。


@pytest.mark.parametrize('axis,value',[('x',.011),('y',.011),('yaw',math.radians(1.1)),('pitch',1.1),('roll',1.1)])
def test_motion_on_any_axis_requires_new_settling_period(axis,value):
    gate=CameraStability()
    for t in (0.,.2,.4,.6):
        settled=sample(gate,t)
    assert settled
    kwargs={axis:value} if axis in ('pitch','roll') else {'pose':replace(CarPose(0,0,0),**{axis:value})}
    assert not sample(gate,.8,**kwargs)
    assert not sample(gate,1.,**kwargs)
    assert not sample(gate,1.2,**kwargs)
    assert sample(gate,1.4,**kwargs)


def test_yaw_wrap_and_small_noise_do_not_prevent_settling():
    gate=CameraStability()
    for i,deg in enumerate((179.9,-179.9,179.8,-179.8)):
        settled=sample(gate,i*.2,CarPose(.001*(i%2),0.,math.radians(deg)),pitch=.1*(i%2),roll=.2*(i%2))
    assert settled


def test_range_detects_jitter_in_opposite_directions_and_slow_accumulated_drift():
    gate=CameraStability()
    sample(gate,0.)
    sample(gate,.2,CarPose(.008,0.,0.))
    assert not sample(gate,.4,CarPose(-.008,0.,0.))
    assert gate.samples == 1
    for i in range(1,7):
        assert not sample(gate,.4+i*.2,CarPose(-.008+.006*i,0.,0.))


def test_missing_frames_long_gaps_and_invalid_imu_clear_stability():
    gate=CameraStability()
    for t in (0.,.2,.4,.6):
        sample(gate,t)
    assert not sample(gate,.8,valid=False)
    assert gate.samples == 0
    assert not sample(gate,1.)
    assert not sample(gate,1.2,pitch=float('nan'))
    assert not sample(gate,1.4,pitch=None)
    for t in (2.,2.2,2.4,2.6):
        sample(gate,t)
    assert gate.settled
    assert not sample(gate,4.)


def test_repeated_radar_measurement_never_counts_as_multiple_samples():
    gate=CameraStability()
    pose=CarPose(0,0,0,100.)
    for t in (0.,.2,.4,.6,.8,1.):
        assert not sample(gate,t,pose)
    assert gate.samples == 1
    for i,t in enumerate((1.2,1.4,1.6,1.8),1):
        settled=sample(gate,t,replace(pose,stamp=100.+i*.2))
    assert settled


def test_capture_samples_still_settle_when_recognition_takes_over_half_second(monkeypatch):
    import numpy as np
    import report
    import view_field as vf
    from types import SimpleNamespace
    from car_pose import Setup,load_intrinsics
    from classify import CellView
    from field_map import DISPLAY
    from report import FieldWatch
    monkeypatch.setattr(vf,'pitch_from_gravity',lambda g: g[0])
    monkeypatch.setattr(vf,'roll_from_gravity',lambda g: g[1])
    views={n:CellView(None,False,False,False,False,0.,0.,0.,255.) for n in DISPLAY}
    monkeypatch.setattr(report,'inspect_frame',lambda *args:views)
    watch=FieldWatch(Setup(True,np.zeros(2),0.,'', 'rad'),load_intrinsics('config/intrinsics.yaml'))
    color=np.zeros((2,2,3),np.uint8)
    for wall,source_time in ((0.,100.),(1.4,101.4),(2.8,102.8)):
        poses=[CarPose(0,0,0,source_time+i*.2) for i in range(5)]
        frames=[(None,None,np.zeros(3),source_time+i/30,wall) for i in range(25)]
        stream=SimpleNamespace(measurements_between=lambda start,end:[p for p in poses if start<=p.stamp<=end])
        watch.stability_samples=vf.camera_stability_samples(frames,stream,poses[-1],frames[-1][3])
        monkeypatch.setattr(report.time,'monotonic',lambda:wall)
        watch.update(poses[-1],color)
        assert watch.camera_settled  # 连续原始采样已停稳，识别循环1.4秒不算断流。
        assert watch.stable_frames >= 3  # 原始采样已满足帧数，不必再等三轮慢速识别。
    # 缺失历史数据仍禁止确认，不能直接沿用上一轮的停稳结果。
    watch.stability_samples=[]
    watch.update(poses[-1],color)
    assert not watch.camera_settled


def test_historical_samples_before_b_cannot_satisfy_new_start_settling(monkeypatch):
    import numpy as np
    import report
    from car_pose import Setup,load_intrinsics
    from classify import CellView
    from field_map import DISPLAY
    from report import FieldWatch
    views={n:CellView(None,False,False,False,False,0.,0.,0.,255.) for n in DISPLAY}
    monkeypatch.setattr(report,'inspect_frame',lambda *args:views)
    watch=FieldWatch(Setup(True,np.zeros(2),0.,'', 'rad'),load_intrinsics('config/intrinsics.yaml'))
    watch.stability_after_stamp=100.6
    watch.stability_samples=[(CarPose(0,0,0,100.+i*.2),0.,0.,100.+i*.2) for i in range(5)]
    watch.update(watch.stability_samples[-1][0],np.zeros((2,2,3),np.uint8))
    assert not watch.camera_settled
