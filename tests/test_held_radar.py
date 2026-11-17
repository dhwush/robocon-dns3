"""缓存位姿可延续静止识别，不能伪造移动后停稳。"""
from types import SimpleNamespace
import numpy as np
import pytest
from car_pose import CarPose,Setup,Intrinsics
from report import FieldWatch
from classify import CellView
from field_map import DISPLAY


def watcher(monkeypatch):
    seen=[]
    def inspect(p,o,pitch,i,color,depth,roll):
        seen.append(color)
        return {n:CellView(None,True,False,False,False,0.,0.,0.,255.) for n in DISPLAY}
    monkeypatch.setattr('report.inspect_frame',inspect)
    w=FieldWatch(Setup(True,np.zeros(3),0.,'', 'rad',False),Intrinsics(8,8,8,8,4,4))
    w.live_motion_guard=True
    w.light_hand=True
    clock=[0.]
    monkeypatch.setattr('report.time.monotonic',lambda:clock[0])
    c=np.ones((8,8,3),np.uint8)
    for t in (.1,.3,.5,.7):
        clock[0]=t
        w.update(CarPose(0,0,0,t),c,None)
    assert w.camera_settled and seen[-1] is c
    return w,clock,c,seen


def test_loss_continues_on_last_real_pose_without_new_settle_samples(monkeypatch):
    w,clock,c,seen=watcher(monkeypatch)
    p=w.last_pose
    samples=w.camera_stability.samples
    w.pose_reused=True
    for t in (1.,2.,3.):
        clock[0]=t
        w.stability_samples=[]
        w.update(p,c,None)
        assert w.camera_settled
        assert seen[-1] is c
        assert w.last_pose is p and w.last_pose.stamp==.7
        assert w.camera_stability.samples==samples


@pytest.mark.parametrize('kind',['move','yaw','pitch'])
def test_motion_then_loss_requires_fresh_settling(monkeypatch,kind):
    w,clock,c,seen=watcher(monkeypatch)
    p=w.last_pose
    w.pose_reused=True
    w.hold=SimpleNamespace(kind=kind)
    clock[0]=1.
    w.update(p,c,None)
    assert not w.camera_settled and seen[-1] is None
    w.hold=None
    for t in (1.2,1.4,1.6):
        clock[0]=t
        w.update(p,c,None)
        assert not w.camera_settled and seen[-1] is None
    w.pose_reused=False
    for t in (2.,2.2,2.4,2.6):
        clock[0]=t
        w.update(CarPose(.2,0,0,t),c,None)
    assert w.camera_settled and seen[-1] is c


def test_imu_change_during_loss_stops_recognition_until_fresh_pose(monkeypatch):
    w,clock,c,seen=watcher(monkeypatch)
    w.pose_reused=True
    w.pitch=3.
    clock[0]=1.
    w.update(w.last_pose,c,None)
    assert not w.camera_settled and seen[-1] is None
    w.pitch=0.
    clock[0]=1.2
    w.update(w.last_pose,c,None)
    assert not w.camera_settled and seen[-1] is None


def test_never_settled_cannot_settle_on_repeated_cached_pose(monkeypatch):
    w,clock,c,seen=watcher(monkeypatch)
    w.camera_settled=False
    w.pose_reused=True
    for t in (1.,1.2,1.4,1.6):
        clock[0]=t
        w.update(w.last_pose,c,None)
        assert not w.camera_settled and seen[-1] is None
