import runpy
from dataclasses import replace
import pytest
from motion import Action
from field_map import DISPLAY

T=runpy.run_path('tests/test_autonomous.py')

def setup(monkeypatch):
    survey,watch,clock,name=T['make_survey'](monkeypatch)
    survey.stationary_scan=False
    for n in DISPLAY:watch.votes[n].locked='1'
    watch.votes[name].locked=None
    watch.views[name].conclusion=None
    survey.final_fallback=True
    return survey,watch,clock,name

@pytest.mark.parametrize('compensated',[False,True])
def test_final_unknown_after_valid_near_or_compensation_becomes_zero(monkeypatch,compensated,tmp_path):
    import autonomous
    s,w,c,n=setup(monkeypatch)
    calls=[]
    def planner(*args,**kwargs):
        calls.append(kwargs['require_move'])
        if compensated and kwargs['require_move']:
            return Action('blocked','不可靠近')
        return Action('wait','最终观察',cell=n,cells=(n,),observation_x=0,observation_y=0,
                      observation_yaw=0,observation_pitch=0,observation_fallback=True)
    monkeypatch.setattr(autonomous,'nearest_observation',planner)
    s._plan_final_observation([n],0)
    assert calls==([True,False] if compensated else [True])
    # 精确模拟已到位、停稳且连续有效的最终观察窗口。
    s.look_key=(n,0);s.look_started=0
    s.final_valid_start=0;s.final_valid_last=2.1;s.final_valid_samples=8
    assert s._expire_look(2.1)
    assert w.votes[n].locked=='0' and s.complete
    assert n in s.fallback_zero
    assert s.history[n][-1]['event']=='fallback_zero'
    assert '兜底' in '\n'.join(s.lines())
    assert n in s.snapshot()['fallback_zero']

@pytest.mark.parametrize('condition',['image','stable','short','samples','gap'])
def test_fallback_does_not_use_invalid_observation(monkeypatch,condition):
    s,w,c,n=setup(monkeypatch)
    s.final_goal_names=(n,);s.look_key=(n,0);s.look_started=0
    s.final_valid_start=0;s.final_valid_last=2.1;s.final_valid_samples=8
    if condition=='image':w.have_color=False
    if condition=='stable':w.stable_frames=0
    if condition=='short':s.final_valid_start=1.
    if condition=='samples':s.final_valid_samples=2
    if condition=='gap':s.final_valid_last=1.
    assert not s._expire_look(2.1)
    assert w.votes[n].locked is None

def test_no_reachable_compensation_does_not_blindly_infer(monkeypatch):
    import autonomous
    s,w,c,n=setup(monkeypatch)
    monkeypatch.setattr(autonomous,'nearest_observation',lambda *a,**k:Action('blocked','无点'))
    s._plan_final_observation([n],0)
    assert s.action.kind=='blocked' and w.votes[n].locked is None

def test_real_result_not_replaced_by_fallback(monkeypatch):
    s,w,c,n=setup(monkeypatch)
    s.final_goal_names=(n,);s.look_key=(n,0);s.look_started=0
    s.final_valid_start=0;s.final_valid_last=2.1;s.final_valid_samples=8
    w.votes[n].locked='蓝'
    s._expire_look(2.1)
    assert w.votes[n].locked=='蓝' and n not in s.fallback_zero


def test_update_runs_compensation_wait_to_final_output(monkeypatch):
    import autonomous
    s,w,c,n=setup(monkeypatch)
    def planner(*a,**kw):
        if kw['require_move']:return Action('blocked','不能靠近')
        return Action('wait','补偿',cell=n,cells=(n,),observation_x=0,observation_y=0,
                      observation_yaw=0,observation_pitch=0)
    monkeypatch.setattr(autonomous,'nearest_observation',planner)
    s._plan_final_observation([n],0.)
    for i in range(1,18):
        c.now=i*.2
        s.update()
        if s.complete:break
    assert s.complete and w.votes[n].locked=='0'
    assert [e['event'] for e in s.history[n]]==['fallback_compensation','fallback_zero']
    from terminal_ui import terminal_status
    w.autonomy=s
    rows,_,_=terminal_status(w,w.last_pose,w.pitch)
    assert '兜底推定0层' in '\n'.join(rows)
