from types import SimpleNamespace
import numpy as np
from car_pose import CarPose,Setup,load_intrinsics
from report import FieldWatch
from classify import CellView
from field_map import DISPLAY
from motion import Action


def setup(monkeypatch):
    import report
    clock=SimpleNamespace(now=10.)
    w=FieldWatch(Setup(True,np.zeros(2),0.,'','rad'),load_intrinsics('config/intrinsics.yaml'))
    w.enable_autonomy(lambda:clock.now)
    w.autonomy.stationary_scan=True
    w.live_motion_guard=True
    monkeypatch.setattr(w.autonomy,'_schedule',lambda now:None)
    calls=[]
    def inspect(*args):
        calls.append(args[4])
        return {n:CellView('蓝',True,False,False,False,0,0,0,100) for n in DISPLAY}
    monkeypatch.setattr(report,'inspect_frame',inspect)
    return w,clock,calls


def test_waiting_and_reused_pose_keep_preview_without_confirmation(monkeypatch):
    w,c,calls=setup(monkeypatch)
    w.pose_reused=True
    color=np.zeros((2,2,3),np.uint8)
    w.update(CarPose(0,0,0,10),color,None)
    assert calls[-1] is color
    assert all(v.conclusion=='蓝' for v in w.views.values())
    assert all(v.locked is None for v in w.votes.values())
    assert not w.confirmation_allowed


def test_brief_sync_gap_preserves_samples_and_recovers(monkeypatch):
    w,c,calls=setup(monkeypatch)
    image=np.zeros((2,2,3),np.uint8)
    def feed(t,reused=False):
        c.now=t;w.pose_reused=reused
        p=CarPose(0,0,0,t)
        w.stability_samples=[] if reused else [(p,0.,0.,t)]
        w.update(p,image,None)
    feed(10.);feed(10.2)
    assert w.camera_stability.samples==2
    feed(10.3,True)
    assert w.camera_stability.samples==2
    assert not w.confirmation_allowed
    feed(10.4);feed(10.6)
    assert w.camera_settled and w.confirmation_allowed
    assert all(v.locked=='蓝' for v in w.votes.values())


def test_arrival_resets_once_then_uses_new_samples(monkeypatch):
    w,c,_=setup(monkeypatch)
    image=np.zeros((2,2,3),np.uint8)
    w.last_pose=CarPose(0,0,0,10.)
    s=w.autonomy
    s.action=Action('move','到位',x=0,y=0)
    assert s._stop_at_arrival(10.)
    first=s.stop_until
    assert s._stop_at_arrival(10.2) and s.stop_until==first
    for i in range(1,7):
        t=10.+i*.2;c.now=t;p=CarPose(0,0,0,t)
        w.stability_samples=[(p,0,0,t)]
        w.update(p,image,None)
    assert w.camera_settled and s.stop_until is None
