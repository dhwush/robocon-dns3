"""多帧实时循环：地图待测列表不能覆盖相机共享帧缓存。"""
from types import SimpleNamespace
import sys
import threading

import numpy as np


def test_live_camera_keeps_reading_frames_after_map_status_update(monkeypatch, tmp_path):
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
    monkeypatch.setattr(cv2,'destroyAllWindows',lambda: None)
    keys = iter([-1,-1,ord('q')])
    monkeypatch.setattr(cv2,'waitKey',lambda delay: next(keys))
    votes = {n: Vote() for n in DISPLAY}
    session = SimpleNamespace(watch=None,pause=lambda: None,save=lambda path: None,
        votes=votes,message='等待稳定',lines=lambda: ['等待稳定'],route_display=lambda: {})
    # _live沿用已有会话，保留自主状态提示分支，隔离识别算法与硬件。
    seen = []
    watch = SimpleNamespace(votes=votes,autonomy=None,
        setup=Setup(False,np.zeros(2),0.,'', 'rad',True),light='daylight',pitch=0.,roll=0.,
        follow_pitch=lambda value: None,update=lambda pose,image,depth: seen.append(image),
        views={},lines=lambda mode: ['测试'],window_lines=lambda mode: ['光线 白天'],required_pitch=lambda: 0.)
    session.watch = watch
    args = SimpleNamespace(mode='basic',pose_source='static',survey_session=session,
        survey_report=tmp_path/'report.json')
    assert vf._live(args,watch,CarPose(0,0,0)) == ['测试']
    assert len(seen) == 3
    assert isinstance(frames,dict) and frames['color'] is color
    assert args.autonomous_map_lines[2].startswith('待测塔：')
    assert stopped == [True]
