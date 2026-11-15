"""多帧实时循环：地图待测列表不能覆盖相机共享帧缓存。"""
from types import SimpleNamespace
import sys
import threading

import numpy as np
import pytest


@pytest.mark.parametrize('source', ['static', 'radar', 'ros', 'laptop', 'host'])
def test_live_camera_keeps_reading_frames_after_map_status_update(monkeypatch, tmp_path, source):
    import cv2
    import view_field as vf
    from car_pose import CarPose, Setup, load_intrinsics
    from field_map import DISPLAY
    from vote import Vote

    monkeypatch.setitem(sys.modules, 'pyrealsense2', SimpleNamespace())
    intr = load_intrinsics('config/intrinsics.yaml')
    color = np.zeros((8,8,3),np.uint8)
    frames = dict(color=color,depth=np.ones((8,8)),gravity=None,stamp=1.,received=1.,
                  intrinsics=intr,camera=dict(serial_number='test',distortion_model='none',distortion_coefficients=[]))
    stopped = []
    ready = SimpleNamespace(wait=lambda timeout: True,clear=lambda: None)
    pipeline = SimpleNamespace(stop=lambda: stopped.append(True))
    monkeypatch.setattr(vf,'_open_camera',lambda rs: (pipeline,None,ready,threading.Lock(),frames))
    monkeypatch.setattr(vf,'_apply_light',lambda *args: None)
    monkeypatch.setattr(vf,'paint_window',lambda image,*args,**kw: image)
    monkeypatch.setattr(vf,'live_panels',lambda *args: {})
    monkeypatch.setattr(cv2,'imshow',lambda *args: None)
    created_windows = set()
    def create_window(name, *args):
        assert name.isascii(), 'Qt回调窗口必须使用ASCII内部名称'
        created_windows.add(name)
    def bind_mouse(name, callback):
        assert name in created_windows
        assert name == vf.CAMERA_WINDOW
    monkeypatch.setattr(cv2, 'namedWindow', create_window)
    monkeypatch.setattr(cv2, 'setMouseCallback', bind_mouse)
    monkeypatch.setattr(cv2,'destroyAllWindows',lambda: None)
    keys = iter([-1,-1,ord('q')])
    monkeypatch.setattr(cv2,'waitKey',lambda delay: next(keys))
    votes = {n: Vote() for n in DISPLAY}
    pauses, suspensions = [], []
    session = SimpleNamespace(watch=None,pause=lambda: pauses.append(True),
        suspend_sampling=lambda: suspensions.append(True),save=lambda path: None,
        confirmation_target=lambda: None, metrics=lambda: dict(observation_count=0,distance_m=0.),
        votes=votes,message='等待稳定',lines=lambda: ['等待稳定'],route_display=lambda: {})
    # _live沿用已有会话，保留自主状态提示分支，隔离识别算法与硬件。
    seen = []
    watch = SimpleNamespace(votes=votes,autonomy=None,
        setup=Setup(False,np.zeros(2),0.,'', 'rad',True),light='daylight',pitch=0.,roll=0.,
        follow_pitch=lambda value: None,update=lambda pose,image,depth: seen.append(image),
        views={},lines=lambda mode: ['测试'],window_lines=lambda mode: ['光线 白天'],required_pitch=lambda: 0.)
    session.watch = watch
    matches = iter([None, CarPose(0,0,0), CarPose(0,0,0)])
    stream = SimpleNamespace(match=lambda stamp: next(matches),close=lambda: None,
                             process=SimpleNamespace(poll=lambda: None))
    monkeypatch.setattr('can_pose.CanPoseStream',lambda *args: stream)
    stream.error = '测试断流'
    monkeypatch.setattr(vf, 'open_radar', lambda args: pytest.fail('手动模式不应读取雷达') if source == 'static' else stream)
    args = SimpleNamespace(mode='basic',pose_source=source,offline=False,survey_session=session,
        survey_report=tmp_path/'report.json')
    assert vf._live(args,watch,CarPose(0,0,0)) == ['测试']
    assert len(seen) == (3 if source == 'static' else 2)
    assert pauses == [True]  # 仅重开相机时完整暂停；断流不能再次调用。
    assert len(suspensions) == (0 if source == 'static' else 1)
    assert isinstance(frames,dict) and frames['color'] is color
    assert args.autonomous_map_lines[2].startswith('待测塔：')
    assert stopped == [True]


def test_delayed_radar_uses_historical_color_depth_and_gravity(monkeypatch):
    from view_field import matched_camera_frame
    from radar_pose import RadarPoseStream
    from car_pose import CarPose
    monkeypatch.setattr('radar_pose.time.time', lambda: 100.)
    stream = RadarPoseStream(start=False)
    stream.accept(CarPose(2, 3, .4, 99.55))
    old = (object(), object(), object(), 99.56, 1.)
    latest = (object(), object(), object(), 100., 1.5)
    frame, pose = matched_camera_frame([old, latest], stream, float('-inf'))
    assert frame is old
    assert pose.stamp == 99.55
    assert matched_camera_frame([old, latest], stream, old[3]) is None
    assert matched_camera_frame([latest], stream, float('-inf')) is None
    stream.invalidate('定位失败')
    assert matched_camera_frame([old], stream, float('-inf')) is None


def test_history_does_not_revive_stale_pose(monkeypatch):
    from view_field import matched_camera_frame
    from radar_pose import RadarPoseStream
    from car_pose import CarPose
    monkeypatch.setattr('radar_pose.time.time', lambda: 100.)
    stream = RadarPoseStream(start=False)
    stream.accept(CarPose(2, 3, .4, 99.55))
    monkeypatch.setattr('radar_pose.time.time', lambda: 101.)
    assert matched_camera_frame([(None,None,None,99.55,1.)], stream, 0.) is None
