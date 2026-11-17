import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

import autonomous
from autonomous import AutonomousSurvey, Evidence
from car_pose import CarPose, load_intrinsics
from classify import CellView
from field_map import DISPLAY
from motion import Action
from vote import Vote


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def make_survey(monkeypatch):
    clock = Clock()
    name = DISPLAY[0]
    views = {n: CellView(None, False, True, False, False, 0., 0., 0., 255.) for n in DISPLAY}
    views[name] = CellView('1', True, False, False, False, 0., 0., .5, 150.)
    watch = SimpleNamespace(last_pose=CarPose(0, 0, 0), pitch=0., roll=0.,
        have_color=True, stable_frames=10, views=views,
        setup=SimpleNamespace(center_to_corner=np.zeros(2)),
        intr=load_intrinsics('config/intrinsics.yaml'),
        votes={n: Vote() for n in DISPLAY}, _arrived=lambda action: True)
    monkeypatch.setattr(autonomous, 'fits_when_aimed', lambda cell, *args: cell.name == name)
    monkeypatch.setattr(autonomous, 'fits_rgb_when_aimed', lambda *args: False)
    monkeypatch.setattr(autonomous, 'nearest_observation', lambda *args: Action('blocked', '无安全位置'))
    def aim(n, v, p, o, pitch, color, intr, attempt, targets=None):
        if attempt == 1 and abs(pitch-8.) > 1:
            return Action('pitch', '抬头到8度', pitch=8., cell=n, attempt=1)
        return Action('wait', '', cell=n, attempt=attempt)
    monkeypatch.setattr(autonomous, '_aim_cell', aim)
    survey = AutonomousSurvey(watch, clock, stationary_scan=True)
    monkeypatch.setattr(survey, '_targets', lambda n: tuple(zip(autonomous.GOAL_BEARING,
                         (watch.views[n].pitch_target_deg+d for d in autonomous.PITCH_OFFSET))))
    return survey, watch, clock, name


def observe(survey, clock, seconds):
    for _ in range(round(seconds*10)):
        clock.now += .1
        survey.update()


def test_confirmation_requires_elapsed_time_and_several_windows():
    evidence = Evidence()
    for index in range(1000):
        assert evidence.see('1', index/1000) is None
    result = None
    for index in range(11, 66):
        result = evidence.see('1', index/10) or result
    assert result == '1'


def test_unknown_ratio_and_frame_gaps_prevent_confirmation():
    evidence = Evidence()
    for index in range(100):
        assert evidence.see('1' if index % 2 else None, index*.1) is None
    evidence.clear()
    for index in range(35):
        evidence.see('1', index*.1)
    assert evidence.windows == 1
    evidence.see('1', 10.)
    assert evidence.windows == 0


def test_first_valid_result_is_remembered_without_conflict_rechecks(monkeypatch, tmp_path):
    survey, watch, clock, name = make_survey(monkeypatch)
    survey.update()
    assert watch.votes[name].locked == '1'
    assert not survey.pitch_checks[name]
    watch.views[name].conclusion = '2'
    observe(survey, clock, 9)
    assert watch.votes[name].locked == '1'
    assert not watch.votes[name].recheck
    assert [item['event'] for item in survey.history[name]] == ['confirmed']
    output = tmp_path/'memory.json'
    survey.save(output)
    saved = json.loads(output.read_text())
    assert saved['results'][name] == '1'
    assert name not in saved['pending']
    assert not saved['conflict_results']


def test_missing_color_and_unstable_camera_cannot_record_results(monkeypatch):
    survey, watch, clock, name = make_survey(monkeypatch)
    watch.have_color = False
    survey.update()
    assert watch.votes[name].locked is None
    watch.have_color = True
    watch.stable_frames = 0
    survey.update()
    assert watch.votes[name].locked is None
    watch.stable_frames = 10
    survey.update()
    assert watch.votes[name].locked == '1'


def test_unknown_keeps_task_open_instead_of_finishing(monkeypatch):
    survey, watch, clock, name = make_survey(monkeypatch)
    watch.views[name].conclusion = None
    monkeypatch.setattr(autonomous, '_aim_cell', lambda n,v,p,o,pi,c,i,a,*args: Action('wait', '', cell=n, attempt=a))
    observe(survey, clock, autonomous.LOOK_SECONDS * (autonomous.MAX_TRIES + 1))
    assert not survey.complete
    assert watch.votes[name].locked is None
    assert survey.attempts[name] == autonomous.MAX_TRIES
    assert survey.action.kind == 'blocked'
    assert not any('最终结果' in line for line in survey.lines())


def test_yaw_jitter_clears_evidence_without_restarting_attempt_timeout(monkeypatch):
    survey, watch, clock, name = make_survey(monkeypatch)
    watch.views[name].conclusion = None
    observe(survey, clock, 1)
    started = survey.look_started
    for yaw in (4., 0., 4., 0.):
        watch.last_pose = CarPose(0, 0, math.radians(yaw))
        observe(survey, clock, 2)
        assert survey.look_started == started
    observe(survey, clock, 2)
    assert survey.attempts[name] == 1
    assert survey.action.kind == 'pitch'
    assert any(item['event'] == 'observation_timeout' for item in survey.history[name])


def test_unstable_frames_do_not_restart_attempt_timeout(monkeypatch):
    survey, watch, clock, name = make_survey(monkeypatch)
    watch.views[name].conclusion = None
    observe(survey, clock, 1)
    watch.stable_frames = 0
    observe(survey, clock, autonomous.NO_PROGRESS_SECONDS + 1)
    assert survey.attempts[name] == 1
    assert watch.votes[name].locked is None


def test_partial_progress_cannot_hold_same_attempt_forever(monkeypatch):
    survey, watch, clock, name = make_survey(monkeypatch)
    watch.views[name].conclusion = None
    observe(survey, clock, 1)
    started = survey.look_started
    # 有一个有效段，之后长期未知：允许延长观察，但仍有本档总时限。
    survey.evidence[name].windows = 1
    clock.now = started + autonomous.NO_PROGRESS_SECONDS + 1
    survey.last_update = clock.now
    survey._schedule(clock.now)
    assert survey.attempts[name] == 0
    clock.now = started + autonomous.LOOK_SECONDS + .1
    survey._schedule(clock.now)
    assert survey.attempts[name] == 1


def test_frame_gap_preserves_attempt_deadline_but_discards_evidence(monkeypatch):
    survey, watch, clock, name = make_survey(monkeypatch)
    watch.views[name].conclusion = None
    observe(survey, clock, 1)
    started = survey.look_started
    clock.now += autonomous.NO_PROGRESS_SECONDS + 1
    survey.update()
    assert survey.attempts[name] == 1
    assert survey.look_started is None
    assert watch.votes[name].locked is None


def test_final_summary_requires_a_result_for_every_tower(monkeypatch):
    survey, watch, clock, name = make_survey(monkeypatch)
    for vote in watch.votes.values():
        vote.locked = '1'
    survey.update()
    assert survey.complete
    assert '最终结果' in survey.lines()
    watch.votes[name].locked = None
    assert not survey.complete
    assert '最终结果' not in survey.lines()


def test_nearest_observation_prefers_small_move_and_shared_tasks(monkeypatch):
    import motion
    monkeypatch.setattr(motion, '_distances', lambda *args: [])
    monkeypatch.setattr(motion, '_parking_clear', lambda *args: True)
    monkeypatch.setattr(motion, '_ground_blocked', lambda *args: False)
    monkeypatch.setattr(motion, 'fits_when_aimed', lambda *args: True)
    monkeypatch.setattr(motion, '_bypass_points', lambda *args: [])
    monkeypatch.setattr(motion, '_travel', lambda x,y,a,b,*args: (math.hypot(x-a,y-b), [(x,y),(a,b)]))
    monkeypatch.setattr(motion, 'observation_poses', lambda cell,p,*args: iter([(0,p,0.)]))
    action = motion.nearest_observation(CarPose(0,0,0), np.zeros(2), None, list(DISPLAY))
    assert action.kind == 'wait'
    assert action.observation_x == 0 and action.observation_y == 0
    assert set(action.cells) == set(DISPLAY)
    monkeypatch.setattr(motion, '_travel', lambda *args: None)
    assert motion.nearest_observation(CarPose(0,0,0), np.zeros(2), None, list(DISPLAY)).kind == 'blocked'


def test_field_watch_records_first_stable_result_without_vote_delay(monkeypatch):
    from car_pose import Setup, OBS_STABLE_FRAMES
    from report import FieldWatch
    import report
    survey, fake, clock, name = make_survey(monkeypatch)
    watch = FieldWatch(Setup(True, np.zeros(2), 0., '', 'rad'), fake.intr)
    watch.enable_autonomy(clock)
    monkeypatch.setattr(report, 'inspect_frame', lambda *args: fake.views)
    color = np.zeros((2, 2, 3), np.uint8)
    for _ in range(OBS_STABLE_FRAMES):
        clock.now += .05
        watch.update(fake.last_pose, color)
        assert watch.votes[name].locked is None
    clock.now += .7
    watch.update(fake.last_pose, color)
    assert watch.votes[name].locked is None  # 采样间断不能当作持续停稳。
    for _ in range(4):
        clock.now += .2
        watch.update(fake.last_pose,color)
    assert watch.votes[name].locked == '1'
    assert len(watch.autonomy.history[name]) == 1


def test_parallax_request_uses_move_planner_after_failed_stationary_observation(monkeypatch):
    survey, watch, clock, name = make_survey(monkeypatch)
    watch.views[name].conclusion = None
    watch.views[name].needs_parallax = True
    observe(survey, clock, 1)
    assert survey.action.kind == 'wait'
    observe(survey, clock, autonomous.LOOK_SECONDS + 1)
    assert survey.action.kind == 'blocked'  # 本测试的换位规划器没有安全落点。
    assert survey.attempts[name] == autonomous.MAX_TRIES


def test_yaw_arrival_accepts_two_degree_error_and_wraparound():
    from report import FieldWatch
    watch = SimpleNamespace(last_pose=CarPose(0,0,math.radians(-179)))
    assert FieldWatch._arrived(watch, Action('yaw', '', yaw=math.radians(179)))
    watch.last_pose = CarPose(0,0,math.radians(175))
    assert not FieldWatch._arrived(watch, Action('yaw', '', yaw=math.radians(179)))


def test_stationary_scan_has_seven_yaws_and_skips_invisible_angles(monkeypatch):
    import motion
    assert set(motion.GOAL_BEARING) == {0.,8.,-8.,16.,-16.,24.,-24.}
    name = DISPLAY[0]
    view = CellView(None,True,False,False,False,0.,0.,0.,255.)
    pose = CarPose(0,0,0)
    monkeypatch.setattr(motion, '_parking_clear', lambda *args: True)
    # 只允许正中附近，偏角全都出画；不能回到已经看过的正中。
    monkeypatch.setattr(motion, '_sees', lambda cell,p,*args: abs(math.degrees(p.yaw)) < 5)
    assert motion._aim_cell(name,view,pose,np.zeros(2),0.,True,object(),3) is None
    monkeypatch.setattr(motion, '_sees', lambda *args: True)
    action = motion._aim_cell(name,view,pose,np.zeros(2),0.,True,object(),3)
    assert action.kind == 'yaw'
    assert math.isclose(math.degrees(action.yaw),8.)
    view.bearing_deg = 6.
    action = motion._aim_cell(name,view,pose,np.zeros(2),0.,True,object(),3)
    assert action.kind == 'wait'


def test_every_yaw_has_three_pitch_options():
    from motion import GOAL_BEARING, PITCH_OFFSET, SCAN_YAWS
    assert len(GOAL_BEARING) == 21
    for yaw in SCAN_YAWS:
        assert [pitch for goal,pitch in zip(GOAL_BEARING,PITCH_OFFSET) if goal == yaw] == [0.,8.,-8.]


def test_other_pitch_does_not_reopen_identified_tower(monkeypatch):
    survey, watch, clock, name = make_survey(monkeypatch)
    survey.update()
    watch.pitch = 8.
    watch.views[name].conclusion = '2'
    observe(survey, clock, 7)
    assert watch.votes[name].locked == '1'
    assert not watch.votes[name].recheck
    assert not survey.pitch_checks[name]


def test_identified_result_survives_change_of_position(monkeypatch):
    survey, watch, clock, name = make_survey(monkeypatch)
    survey.update()
    watch.last_pose = CarPose(.11,0,0)
    watch.views[name].conclusion = None
    observe(survey, clock, 1)
    assert watch.votes[name].locked == '1'
    assert name not in survey.snapshot()['pending']


def test_rotation_translation_preserves_scan_and_pitch_checks(monkeypatch):
    survey, watch, clock, name = make_survey(monkeypatch)
    watch.views[name].conclusion = None
    observe(survey, clock, 1)
    survey.attempts[name] = 3
    survey.pitch_checks[name] = [dict(result='蓝', pitch=0.)]
    survey.evidence[name].windows = 1
    survey.evidence[name].samples = ['蓝']
    watch.last_pose = CarPose(.04, -.03, math.radians(8))
    survey.update()
    assert survey.attempts[name] == 3
    assert survey.pitch_checks[name] == [dict(result='蓝', pitch=0.)]
    assert survey.evidence[name].windows == 0
    assert survey.station_pose == CarPose(0, 0, 0)


def test_slow_translation_uses_fixed_station_origin_and_waits_for_stability(monkeypatch):
    survey, watch, clock, name = make_survey(monkeypatch)
    watch.views[name].conclusion = None
    observe(survey, clock, 1)
    survey.attempts[name] = 3
    survey.pitch_checks[name] = [dict(result='蓝', pitch=0.)]
    for x in (.025, .05, .075, .10):
        watch.last_pose = CarPose(x, 0, 0)
        survey.update()
        assert survey.attempts[name] == 3
    watch.last_pose = CarPose(.11, 0, 0)
    watch.stable_frames = 0
    survey.update()
    assert survey.attempts[name] == 3
    watch.stable_frames = 3
    clock.now += autonomous.STOP_HOLD_SECONDS + .1
    survey.update()
    assert survey.attempts[name] == 0
    assert survey.pitch_checks[name] == []
    assert survey.station_pose.x == .11


def test_commanded_small_move_resets_only_after_arrival(monkeypatch):
    survey, watch, clock, name = make_survey(monkeypatch)
    watch.views[name].conclusion = None
    observe(survey, clock, 1)
    survey.attempts[name] = 3
    survey.pitch_checks[name] = [dict(result='蓝', pitch=0.)]
    survey.action = Action('move', '', x=.05, y=0.)
    watch._arrived = lambda action: watch.last_pose.x == .05
    watch.last_pose = CarPose(.03, 0, 0)
    survey.update()
    assert survey.attempts[name] == 3
    assert survey.evidence[name].samples == []
    watch.last_pose = CarPose(.05, 0, 0)
    watch.stable_frames = 0
    survey.update()
    assert survey.attempts[name] == 3
    watch.stable_frames = 3
    clock.now += autonomous.STOP_HOLD_SECONDS + .1
    survey.update()
    assert survey.attempts[name] == 0
    assert survey.pitch_checks[name] == []


def test_heading_targets_remain_fixed_despite_live_pose_changes(monkeypatch):
    import motion
    survey, watch, clock, name = make_survey(monkeypatch)
    monkeypatch.setattr(autonomous, '_aim_cell', motion._aim_cell)
    monkeypatch.setattr(motion, '_parking_clear', lambda *args: True)
    monkeypatch.setattr(motion, '_sees', lambda *args: True)
    watch.last_pose = CarPose(0, 0, math.radians(24.6))
    watch.views[name].bearing_deg = 9.6
    survey.attempts[name] = 3
    action = survey._aim(name)
    assert math.isclose(math.degrees(action.yaw), 23.)
    watch.last_pose = CarPose(.04, -.03, math.radians(28.1))
    watch.views[name].bearing_deg = 12.1
    action = survey._aim(name)
    assert math.isclose(math.degrees(action.yaw), 23.)
    assert math.isclose(survey.heading_center(name), 15.)
    # 搜索框用的实时视图没有被固定目标角度覆盖。
    assert watch.views[name].bearing_deg == 12.1


def test_window_shows_view_adjustment_without_cross_checks(monkeypatch):
    survey, watch, clock, name = make_survey(monkeypatch)
    watch.views[name].conclusion = None
    survey.update()
    text = '\n'.join(survey.lines())
    assert '候选相机俯仰' in text
    assert '仅未看清时继续调整视角' in text
    assert '俯仰复核' not in text


def test_candidate_coverage_can_outweigh_nearest_single_tower(monkeypatch):
    import motion
    names = list(DISPLAY[:2])
    monkeypatch.setattr(motion, '_distances', lambda *args: [])
    monkeypatch.setattr(motion, '_parking_clear', lambda *args: True)
    monkeypatch.setattr(motion, '_ground_blocked', lambda *args: False)
    monkeypatch.setattr(motion, '_bypass_points', lambda *args: [])
    def profiles(cell, pose, *args):
        if abs(pose.y) > 1e-6:
            return iter(())
        if abs(pose.x-.1) < 1e-6 and cell.name == names[0]:
            return iter([(0,pose,0.)])
        if abs(pose.x-.2) < 1e-6:
            return iter([(0,pose,0.),(3,pose,0.)])
        return iter(())
    monkeypatch.setattr(motion, 'observation_poses', profiles)
    monkeypatch.setattr(motion, '_travel', lambda x,y,a,b,*args: (math.hypot(x-a,y-b),[(x,y),(a,b)]))
    pose = CarPose(0,0,0)
    action = motion.nearest_observation(pose,np.zeros(2),None,names)
    assert math.isclose(action.final_x,.2)
    assert set(action.cells) == set(names)
    # 新评分优先观测质量和覆盖；绕行稍长也保留可覆盖两塔的点。
    monkeypatch.setattr(motion, '_travel', lambda x,y,a,b,*args: (1. if a>.15 else .1,[(x,y),(a,b)]))
    action = motion.nearest_observation(pose,np.zeros(2),None,names)
    assert math.isclose(action.final_x,.2)
    assert set(action.cells) == set(names)
    # 真正需要视差时，共享点也不能绕过侧向基线要求。
    monkeypatch.setattr(motion, '_different_view', lambda *args: False)
    action = motion.nearest_observation(pose,np.zeros(2),None,names,references={n:(0,0) for n in names})
    assert action.kind == 'blocked'


def test_point_can_be_useful_when_only_offset_yaw_is_visible(monkeypatch):
    import motion
    cell = motion._cells()[DISPLAY[0]]
    pose = CarPose(0,0,0)
    monkeypatch.setattr(motion, 'aimed_pose', lambda *args: pose)
    monkeypatch.setattr(motion, 'bearing_deg', lambda point,p,*args: math.degrees(p.yaw))
    monkeypatch.setattr(motion, '_parking_clear', lambda *args: True)
    monkeypatch.setattr(motion, '_slant', lambda *args: 2.)
    monkeypatch.setattr(motion, 'visible_height', lambda *args: .35)
    monkeypatch.setattr(motion, '_pitch_target', lambda *args: 0.)
    monkeypatch.setattr(motion, '_sees', lambda cell,p,*args: abs(math.degrees(p.yaw)-8.)<.1)
    poses = list(motion.observation_poses(cell,pose,np.zeros(2),None))
    assert [index for index,_,_ in poses] == [i for i,goal in enumerate(motion.GOAL_BEARING) if goal == 8.]
    assert motion.fits_when_aimed(cell,pose,np.zeros(2),None)


def test_schedule_prefers_smaller_next_turn_and_preserves_active_observation(monkeypatch):
    survey, watch, clock, first = make_survey(monkeypatch)
    second = DISPLAY[1]
    watch.views[first].bearing_deg = -90.
    watch.views[second] = CellView(None,True,False,False,False,-5.,0.,0.,255.)
    monkeypatch.setattr(autonomous, 'fits_when_aimed', lambda cell,*args: cell.name in (first,second))
    def aim(name,*args):
        return Action('yaw','',cell=name,yaw=math.radians(90 if name == first else 5))
    monkeypatch.setattr(autonomous, '_aim_cell', aim)
    survey._schedule(0.)
    assert survey.action.cell == second
    # 正在积累的观察段不能被另一塔较小的转角打断。
    survey.action = Action('wait','',cell=first)
    monkeypatch.setattr(autonomous, '_aim_cell', lambda name,*args: Action('wait','',cell=name))
    survey._schedule(1.)
    assert survey.action.cell == first


def test_pitch_arrival_accepts_three_degrees_but_not_more():
    from report import FieldWatch
    watch = SimpleNamespace(last_pose=CarPose(0,0,0), pitch=9.)
    action = Action('pitch', '', pitch=6.)
    assert FieldWatch._arrived(watch, action)
    watch.pitch = 9.1
    assert not FieldWatch._arrived(watch, action)


def test_low_frame_rate_collects_enough_samples_instead_of_discarding():
    evidence = Evidence()
    result = None
    for index in range(18):
        result = evidence.see('2', index*.8) or result
    assert result == '2'
    evidence.see('2', 15.)
    evidence.see('2', 17.)
    assert evidence.windows == 0
    assert evidence.samples == ['2']


def test_seventy_percent_agreement_accepts_unknowns_but_not_split_results():
    for values, expected in [(['2']*7+[None]*3, '2'), (['2']*6+['1']*4, None)]:
        evidence = Evidence()
        result = None
        for index, value in enumerate(values):
            result = evidence.see(value, index*.2 if index < 9 else 2.1) or result
        assert result is None
        assert evidence.windows == (1 if expected else 0)


def test_motion_does_not_erase_identified_result(monkeypatch):
    survey, watch, clock, name = make_survey(monkeypatch)
    survey.update()
    watch.last_pose = CarPose(.025, 0, math.radians(2.5))
    watch.pitch = 6.
    watch.roll = 2.5
    survey.update()
    assert watch.votes[name].locked == '1'
    assert len(survey.history[name]) == 1


def test_window_reports_identified_and_unseen_towers(monkeypatch):
    survey, watch, clock, name = make_survey(monkeypatch)
    survey.update()
    text = '\n'.join(survey.lines())
    assert '1；已识别' in text
    assert '尚未看清' in text
    assert '冲突' not in text


def test_far_rgb_result_finishes_without_extra_pitch_or_movement(monkeypatch):
    from field_map import tower_cells
    survey, watch, clock, name = make_survey(monkeypatch)
    cell = next(c for c in tower_cells() if c.name == name)
    watch.last_pose = CarPose(cell.x,cell.y-4.5,0.)
    watch.views[name].conclusion = '红'
    watch.views[name].method = 'RGB顶块'
    for n,vote in watch.votes.items():
        if n != name:
            vote.locked = '1'
    survey.update()
    assert watch.votes[name].locked == '红'
    assert survey.action.kind == 'stop'
    assert survey.complete


def test_skip_vote_prevents_recording_even_when_conclusion_present(monkeypatch):
    survey, watch, clock, name = make_survey(monkeypatch)
    watch.views[name].skip_vote = True
    survey.update()
    assert watch.votes[name].locked is None
    watch.views[name].skip_vote = False
    survey.update()
    assert watch.votes[name].locked == '1'


@pytest.mark.parametrize('value', ['1', '2'])
def test_valid_far_white_is_recorded_without_pitch_checks(monkeypatch, value):
    from field_map import tower_cells
    survey, watch, clock, name = make_survey(monkeypatch)
    cell = next(c for c in tower_cells() if c.name == name)
    watch.last_pose = CarPose(cell.x,cell.y-4.5,0.)
    view = watch.views[name]
    view.conclusion, view.method = value, '远处白盒'
    view.measurements = {'far_white': {'upper_clear': True}}
    survey.update()
    assert watch.votes[name].locked == value
    assert not survey.pitch_checks[name]


def test_unknown_far_white_remains_pending(monkeypatch):
    survey, watch, clock, name = make_survey(monkeypatch)
    watch.views[name].conclusion = None
    watch.views[name].method = '远处白盒'
    survey.update()
    assert watch.votes[name].locked is None
    assert name in survey.snapshot()['pending']


def test_far_cap_pixel_budget_enables_opportunity_without_changing_depth_range(monkeypatch):
    import motion
    from field_map import tower_cells
    cell = tower_cells()[0]
    offset = np.zeros(2)
    intr = load_intrinsics('config/intrinsics.yaml')
    # 隔离成像预算；实际行驶边界另有停车、路线和转向用例覆盖。
    monkeypatch.setattr(motion,"_parking_clear",lambda *args:True)
    monkeypatch.setattr(motion,"_right_front_turn_clear",lambda *args:True)
    pose = CarPose(cell.x-4.5,cell.y,-math.pi/2)
    assert not motion.fits_when_aimed(cell,pose,offset,intr)
    assert motion.fits_rgb_when_aimed(cell,pose,offset,intr)
    assert not motion.fits_rgb_when_aimed(cell,CarPose(cell.x-40.,cell.y,-math.pi/2),offset,intr)
    assert motion.DEPTH_FAR_M == 3.


def test_only_unseen_towers_are_passed_to_dynamic_planner(monkeypatch):
    from motion import RouteStop
    survey, watch, clock, name = make_survey(monkeypatch)
    for n,vote in watch.votes.items():
        if n != name:
            vote.locked = '1'
    watch.views[name].conclusion = None
    survey.primary_stops = [RouteStop(.5,0,(),'沿途复看')]
    survey.attempts[name] = autonomous.MAX_TRIES
    calls = []
    def planner(pose, offset, intr, pending, exhausted, references, pitch, failed):
        calls.append(pending)
        return Action('move', '查看未看清目标', x=.2, y=0.,
                      final_x=.2, final_y=0., cells=tuple(pending))
    monkeypatch.setattr(autonomous, 'nearest_observation', planner)
    survey._schedule(clock())
    assert calls == [[name]]
    assert survey.action.cells == (name,)
    assert not survey.primary_stops


def test_primary_path_is_short_and_every_station_has_looks():
    import motion
    pose = CarPose(0,0,0)
    stops,line = motion.primary_route(pose,np.zeros(2),load_intrinsics('config/intrinsics.yaml'))
    assert stops[0].kind == 'A起点'
    assert (stops[0].x,stops[0].y) == (0,0)
    assert len(stops) >= 2
    assert stops[-1].kind == 'B近点'
    assert sum(math.dist(a,b) for a,b in zip(line,line[1:])) <= 1.8
    assert all(s.looks for s in stops)
    assert {look.number for look in stops[0].looks} >= {'3','6'}
    assert all(motion._parking_clear(s.x,s.y,pose.yaw) for s in stops)
    assert all(motion._path_clear(*a,*b,pose.yaw) for a,b in zip(line,line[1:]))


def test_pose_dropout_preserves_scan_target_and_freezes_wait_timer(monkeypatch):
    survey, watch, clock, name = make_survey(monkeypatch)
    watch.views[name].conclusion = None
    observe(survey, clock, 1)
    survey.attempts[name] = 3
    survey.pitch_checks[name] = [dict(result='蓝', pitch=0.)]
    survey.action = Action('wait', '', cell=name, attempt=3, yaw=math.radians(23))
    survey.look_key = (name, 3)
    survey.look_started = clock.now
    survey.heading_centers[name] = 15.
    survey.evidence[name].windows = 2
    target = survey.action
    survey.suspend_sampling()
    clock.now += 20
    survey.suspend_sampling()
    assert survey.evidence[name].windows == 0
    assert survey.action == target
    assert survey.look_key == (name, 3)
    watch.last_pose = CarPose(.04, -.03, math.radians(23))
    survey.update()
    assert survey.attempts[name] == 3
    assert survey.pitch_checks[name] == [dict(result='蓝', pitch=0.)]
    assert survey.heading_centers[name] == 15.
    assert survey.look_started == clock.now
    assert survey.station_pose == CarPose(0, 0, 0)
    assert survey.sampling_suspended_at is None


def test_pose_dropout_does_not_hide_large_translation(monkeypatch):
    survey, watch, clock, name = make_survey(monkeypatch)
    watch.views[name].conclusion = None
    observe(survey, clock, 1)
    survey.attempts[name] = 3
    survey.pitch_checks[name] = [dict(result='蓝', pitch=0.)]
    survey.suspend_sampling()
    clock.now += 2
    watch.last_pose = CarPose(.15, 0, 0)
    watch.stable_frames = 0
    survey.update()
    assert survey.attempts[name] == 3
    watch.stable_frames = 3
    survey.update()
    assert survey.attempts[name] == 0
    assert survey.pitch_checks[name] == []
    assert survey.station_pose.x == .15


def test_manual_confirmation_skips_later_conflicts(monkeypatch):
    survey, watch, clock, name = make_survey(monkeypatch)
    survey.action = Action('wait', '', cell=name)
    target = survey.confirmation_target()
    assert target == (name, '1')
    assert survey.confirm_current(target)
    assert watch.votes[name].locked == '1'
    assert name in survey.manual_confirmed
    watch.views[name].conclusion = '2'
    observe(survey, clock, 10)
    assert watch.votes[name].locked == '1'
    assert not watch.votes[name].recheck
    assert survey.history[name][-1]['event'] == 'manual_confirmed'
    assert survey.action.cell != name


def test_manual_confirmation_rejects_stale_result(monkeypatch):
    survey, watch, clock, name = make_survey(monkeypatch)
    survey.action = Action('wait', '', cell=name)
    target = survey.confirmation_target()
    watch.views[name].conclusion = '2'
    assert not survey.confirm_current(target)
    assert not survey.manual_confirmed


def test_observation_count_is_per_started_attempt_not_frame(monkeypatch):
    survey, watch, clock, name = make_survey(monkeypatch)
    observe(survey, clock, 1)
    assert survey.metrics()['observation_count'] == 1
    watch.last_pose = CarPose(.03, .04, 0)
    survey.update()
    assert survey.metrics()['distance_m'] == .05
    assert survey.snapshot()['metrics']['observations'][0]['cell'] == name
    survey.pause()
    assert survey.metrics()['observation_count'] == 1


@pytest.mark.parametrize('kind', ['move', 'yaw', 'pitch'])
def test_arrival_announces_stop_before_stability_and_next_action(monkeypatch, kind):
    survey, watch, clock, name = make_survey(monkeypatch)
    survey.station_pose = watch.last_pose
    survey.action = Action(kind, '执行动作', cell=name, x=0., y=0., yaw=0., pitch=0.)
    watch.stable_frames = 0
    survey.update()
    assert survey.stop_until is not None
    assert '停止！' in '\n'.join(survey.lines())
    assert '下一步' not in '\n'.join(survey.lines())
    assert survey.action.kind == kind
    assert not survey.evidence[name].samples
    clock.now += 1.1
    survey.update()
    assert survey.stop_until is not None  # 时间够了但仍未停稳，不下发下一动作。
    watch.stable_frames = 10
    survey.update()
    assert survey.stop_until is None
    assert watch.votes[name].locked == '1'
    assert survey.action.cell != name
    assert '停止！' in '\n'.join(survey.lines())


def test_primary_route_keeps_one_intermediate_and_one_reserve():
    import motion
    stops, line = motion.primary_route(CarPose(0,0,0), np.zeros(2),
                                      load_intrinsics('config/intrinsics.yaml'))
    assert [s.kind for s in stops] == ['A起点', '沿途复看', 'B近点']
    assert len(stops.reserve) == 1
    reserve = stops.reserve[0]
    assert reserve.kind == '沿途备用'
    assert reserve not in stops
    assert reserve.looks
    assert motion._parking_clear(reserve.x, reserve.y, 0.)
    # 沿途主用/备用仍取原折线的1/3、2/3位置，不新增坐标或姿态。
    length = sum(math.dist(a,b) for a,b in zip(line,line[1:]))
    expected = []
    for fraction in (1/3, 2/3):
        remaining = fraction*length
        for a, b in zip(line, line[1:]):
            span = math.dist(a,b)
            if span and remaining <= span:
                expected.append((a[0]+(b[0]-a[0])*remaining/span,
                                 a[1]+(b[1]-a[1])*remaining/span))
                break
            remaining -= span
    assert all(any(math.dist((s.x,s.y),p) < 1e-8 for p in expected)
               for s in (stops[1], reserve))


def test_fixed_reserve_points_are_not_visited(monkeypatch):
    from motion import RouteStop
    survey, watch, clock, name = make_survey(monkeypatch)
    for n, vote in watch.votes.items():
        if n != name:
            vote.locked = '1'
    watch.views[name].conclusion = None
    survey.reserve_stops = [RouteStop(.5,0,(),'沿途备用')]
    survey.attempts[name] = autonomous.MAX_TRIES
    survey._schedule(clock())
    assert survey.action.kind == 'blocked'
    assert not survey.reserve_stops
    assert survey.route_display()['reserve_points'] == []


def test_completed_survey_does_not_visit_reserve(monkeypatch):
    from motion import RouteStop
    survey, watch, clock, name = make_survey(monkeypatch)
    survey.primary_stops = []
    survey.reserve_stops = [RouteStop(.5,0,(),'沿途备用')]
    for vote in watch.votes.values():
        vote.locked = '1'
    survey._schedule(clock())
    assert survey.action.kind == 'stop'
    assert survey.reserve_index == 0


def test_parallax_failure_after_observation_skips_remaining_in_place_angles(monkeypatch):
    survey, watch, clock, name = make_survey(monkeypatch)
    watch.views[name].conclusion = None
    watch.views[name].needs_parallax = True
    survey.look_key = (name,0)
    survey.look_started = 0.
    assert survey._expire_look(autonomous.LOOK_SECONDS+1)
    assert survey.attempts[name] == autonomous.MAX_TRIES
    assert survey.look_key is None


@pytest.mark.parametrize('flag', ['in_view', 'needs_yaw', 'needs_pitch', 'skip_vote'])
def test_invalid_view_is_not_assumed_identified(monkeypatch, flag):
    survey, watch, clock, name = make_survey(monkeypatch)
    setattr(watch.views[name], flag, flag != 'in_view')
    survey.update()
    assert watch.votes[name].locked is None
    assert name in survey.snapshot()['pending']


def test_same_frame_records_all_visible_results_and_stops(monkeypatch):
    from dataclasses import replace
    survey, watch, clock, name = make_survey(monkeypatch)
    for n in DISPLAY:
        watch.views[n] = replace(watch.views[name], conclusion='0')
    survey.update()
    assert all(v.locked == '0' for v in watch.votes.values())
    assert survey.complete
    assert survey.action.kind == 'stop'
    assert survey.snapshot()['pending'] == []
    assert all(len(items) == 1 for items in survey.history.values())


def test_map_preview_adds_only_points_for_previously_unseen_towers():
    import motion
    from car_pose import body_center
    pose = CarPose(0,0,0)
    stops, line = motion.survey(pose, np.zeros(2), load_intrinsics('config/intrinsics.yaml'))
    assert (stops[0].x,stops[0].y) == (pose.x,pose.y)
    observed = {look.number for look in stops[0].looks}
    assert 1 < len(stops) <= 3
    for stop in stops[1:]:
        assert stop.looks
        assert len({(look.yaw_deg,look.pitch_deg) for look in stop.looks}) == 1
        fresh = {look.number for look in stop.looks}
        assert not fresh & observed
        observed.update(fresh)
        assert math.dist(body_center(pose),body_center(CarPose(stop.x,stop.y,math.radians(stop.looks[0].yaw_deg)))) >= .30-1e-6
    assert observed | {motion.TOWER_NO[n] for n in stops.unobserved} == set('123456')
    assert not stops.reserve


@pytest.mark.parametrize('case, expected', [('visible', 'wait'), ('yaw_only', 'yaw'), ('pitch_only', 'pitch')])
def test_joint_candidates_can_change_yaw_and_pitch_together(monkeypatch, case, expected):
    import motion
    survey, watch, clock, name = make_survey(monkeypatch)
    monkeypatch.setattr(survey, '_targets', AutonomousSurvey._targets.__get__(survey))
    monkeypatch.setattr(autonomous, '_aim_cell', motion._aim_cell)
    monkeypatch.setattr(motion, '_parking_clear', lambda *args: True)
    view = watch.views[name]
    view.conclusion = None
    view.bearing_deg = 18.
    view.pitch_target_deg = 20.
    if case == 'visible':
        sees = lambda *args: True
    elif case == 'yaw_only':
        view.in_view, view.needs_yaw = False, True
        view.bearing_deg = 60.
        watch.pitch = 5.
        sees = lambda c,p,o,pi,i: math.degrees(p.yaw) < -30.
    else:
        view.in_view, view.needs_pitch = False, True
        sees = lambda c,p,o,pi,i: pi >= 12.
    monkeypatch.setattr(autonomous, '_sees', sees)
    monkeypatch.setattr(motion, '_sees', sees)
    action = survey._aim(name)
    assert action.kind == expected
    if case == 'visible':
        assert math.isclose(action.yaw, watch.last_pose.yaw, abs_tol=1e-9)
        assert action.pitch == watch.pitch
    if case == 'yaw_only':
        assert action.pitch != watch.pitch
    if case == 'pitch_only':
        assert action.pitch == 12.
    # 候选和绝对目标固定；实时角度变化不重新排序或回到已尝试的档。
    targets = survey._targets(name)
    watch.pitch += 1.
    assert survey._targets(name) == targets
    assert len(targets) == len(set(targets))


def test_scheduler_does_not_charge_yaw_and_pitch_as_two_observations(monkeypatch):
    survey, watch, clock, first = make_survey(monkeypatch)
    second = DISPLAY[1]
    watch.views[first].conclusion = None
    watch.views[second] = CellView(None, True, False, False, False, 0., 0., 0., 255.)
    monkeypatch.setattr(autonomous, 'fits_when_aimed', lambda c,*args: c.name in (first, second))
    monkeypatch.setattr(autonomous, '_sees', lambda *args: True)
    def aim(n):
        if n == first:
            return Action('yaw', '', yaw=math.radians(4.), pitch=10., cell=n)
        return Action('yaw', '', yaw=math.radians(15.), pitch=watch.pitch, cell=n)
    monkeypatch.setattr(survey, '_aim', aim)
    survey._schedule(clock())
    assert survey.action.cell == first


def test_timeout_advances_reordered_candidates_without_repeating(monkeypatch):
    import motion
    survey, watch, clock, name = make_survey(monkeypatch)
    monkeypatch.setattr(survey, '_targets', AutonomousSurvey._targets.__get__(survey))
    monkeypatch.setattr(autonomous, '_aim_cell', motion._aim_cell)
    monkeypatch.setattr(autonomous, '_sees', lambda *args: True)
    monkeypatch.setattr(motion, '_sees', lambda *args: True)
    monkeypatch.setattr(motion, '_parking_clear', lambda *args: True)
    watch.views[name].conclusion = None
    survey._schedule(clock())
    targets = survey._targets(name)
    assert survey.action.kind == 'wait'
    clock.now += autonomous.NO_PROGRESS_SECONDS+1
    assert survey._expire_look(clock())
    assert survey.attempts[name] == 1
    action = survey._aim(name)
    assert action.attempt >= 1
    assert targets[action.attempt] != targets[0]



def test_scheduler_shares_a_pose_when_adjustment_costs_match(monkeypatch):
    survey, watch, clock, first = make_survey(monkeypatch)
    second = DISPLAY[1]
    watch.views[first].conclusion = None
    watch.views[second] = CellView(None, True, False, False, False, 0., 0., 0., 255.)
    monkeypatch.setattr(autonomous, 'fits_when_aimed', lambda c,*args: c.name in (first, second))
    monkeypatch.setattr(autonomous, '_sees', lambda c,p,*args: p.yaw < 0 or c.name == first)
    monkeypatch.setattr(survey, '_aim', lambda n: Action('yaw', '', cell=n, pitch=0.,
                         yaw=math.radians(10. if n == first else -10.)))
    survey._schedule(clock())
    assert survey.action.cell == second


def test_relocation_treats_pitch_change_as_part_of_same_observation(monkeypatch):
    import motion
    monkeypatch.setattr(motion, '_distances', lambda *args: [])
    monkeypatch.setattr(motion, '_parking_clear', lambda x,y,*args: abs(y) < 1e-8 and .09 < x < .21)
    monkeypatch.setattr(motion, '_ground_blocked', lambda *args: False)
    monkeypatch.setattr(motion, '_bypass_points', lambda *args: [])
    monkeypatch.setattr(motion, '_travel', lambda x,y,a,b,*args: (math.hypot(a-x,b-y), [(x,y),(a,b)]))
    monkeypatch.setattr(motion, 'observation_poses', lambda c,p,*args: iter([(0,p,0. if p.x < .15 else 20.)]))
    pose = CarPose(0,0,0)
    first = motion.nearest_observation(pose, np.zeros(2), None, [DISPLAY[0]], pitch=0.)
    second = motion.nearest_observation(pose, np.zeros(2), None, [DISPLAY[0]], pitch=20.)
    assert math.isclose(first.final_x, .1)
    assert math.isclose(second.final_x, .1)
    assert second.observation_pitch == 0.



def test_joint_goal_moves_turns_and_tilts_before_one_shared_observation(monkeypatch):
    from car_pose import turned_pose
    from report import FieldWatch
    survey, watch, clock, first = make_survey(monkeypatch)
    second = DISPLAY[1]
    for n,v in watch.votes.items():
        if n not in (first, second):
            v.locked = '1'
    watch.views[first].conclusion = None
    watch.views[second] = CellView(None, True, False, False, False, 0., 0., 0., 255.)
    watch._arrived = lambda action: FieldWatch._arrived(watch, action)
    target = turned_pose(CarPose(.4,.4,0), math.radians(30.))
    goal = Action('move', '', x=.4, y=.4, final_x=.4, final_y=.4, cells=(first,second),
                  observation_x=target.x, observation_y=target.y,
                  observation_yaw=target.yaw, observation_pitch=12.)
    monkeypatch.setattr(autonomous, 'nearest_observation', lambda *args: goal)
    monkeypatch.setattr(autonomous, '_travel', lambda x,y,a,b,*args: (math.hypot(a-x,b-y),[(x,y),(a,b)]))
    monkeypatch.setattr(autonomous, '_bypass_points', lambda *args: [])
    survey._schedule(clock())
    assert survey.action.kind == 'move'
    assert survey.metrics()['observation_count'] == 0
    watch.last_pose = CarPose(survey.action.x, survey.action.y, 0.)
    watch.views[first].conclusion = '1'
    survey.update()
    clock.now += 1.1
    survey.update()
    assert survey.action.kind == 'yaw'
    assert watch.votes[first].locked is None
    assert survey.metrics()['observation_count'] == 0
    watch.last_pose = turned_pose(watch.last_pose, target.yaw)
    survey.update()
    clock.now += 1.1
    survey.update()
    assert survey.action.kind == 'pitch'
    assert survey.metrics()['observation_count'] == 0
    # 原地转身会改变右前角坐标，但不能被误判为还需重新移动。
    watch.pitch = 12.
    survey.update()
    clock.now += 1.1
    watch.views[first].conclusion = watch.views[second].conclusion = '1'
    survey.update()
    assert survey.complete
    assert survey.action.kind == 'stop'
    assert survey.metrics()['observation_count'] == 1
    assert set(survey.observations[0]['cells']) == {first,second}
    assert survey.observation_goal is None


def test_common_coverage_does_not_union_incompatible_headings(monkeypatch):
    import motion
    names = list(DISPLAY[:2])
    monkeypatch.setattr(motion, '_distances', lambda *args: [])
    monkeypatch.setattr(motion, '_parking_clear', lambda x,y,*args: x == 0 and y == 0)
    monkeypatch.setattr(motion, '_ground_blocked', lambda *args: False)
    monkeypatch.setattr(motion, '_bypass_points', lambda *args: [])
    def profiles(cell,p,*args):
        yaw = 0. if cell.name == names[0] else math.pi/2
        return iter([(0,CarPose(p.x,p.y,yaw),0.)])
    monkeypatch.setattr(motion, 'observation_poses', profiles)
    action = motion.nearest_observation(CarPose(0,0,0), np.zeros(2), None, names)
    assert len(action.cells) == 1


def test_failed_joint_posture_is_not_selected_again(monkeypatch):
    import motion
    name = DISPLAY[0]
    monkeypatch.setattr(motion, '_distances', lambda *args: [])
    monkeypatch.setattr(motion, '_parking_clear', lambda x,y,*args: x == 0 and y == 0)
    monkeypatch.setattr(motion, '_ground_blocked', lambda *args: False)
    monkeypatch.setattr(motion, '_bypass_points', lambda *args: [])
    monkeypatch.setattr(motion, 'observation_poses', lambda c,p,*args: iter([(0,p,0.)]))
    failed = [dict(cells=[name], x=0., y=0., yaw_deg=0., pitch_deg=0.)]
    action = motion.nearest_observation(CarPose(0,0,0), np.zeros(2), None, [name], failed=failed)
    assert action.kind == 'blocked'


def test_joint_search_selects_moving_rotated_tilted_pose_for_shared_coverage(monkeypatch):
    import motion
    from car_pose import turned_pose
    names = list(DISPLAY[:2])
    monkeypatch.setattr(motion, '_distances', lambda *args: [])
    monkeypatch.setattr(motion, '_parking_clear', lambda x,y,yaw=0.: abs(yaw)>1e-8 or abs(y)<1e-8 and .09<x<.21)
    monkeypatch.setattr(motion, '_ground_blocked', lambda *args: False)
    monkeypatch.setattr(motion, '_bypass_points', lambda *args: [])
    monkeypatch.setattr(motion, '_travel', lambda x,y,a,b,*args: (math.hypot(a-x,b-y),[(x,y),(a,b)]))
    def profiles(cell,p,*args):
        if p.x < .15:
            return iter([(0,p,0.)]) if cell.name == names[0] else iter(())
        return iter([(0,turned_pose(p,math.radians(30.)),12.)])
    monkeypatch.setattr(motion, 'observation_poses', profiles)
    action = motion.nearest_observation(CarPose(0,0,0), np.zeros(2), None, names)
    assert math.isclose(action.final_x, .2)
    assert set(action.cells) == set(names)
    assert math.isclose(action.observation_yaw, math.radians(30.))
    assert action.observation_pitch == 12.



def test_camera_panel_shows_joint_goal_instead_of_obsolete_pitch_checks(monkeypatch):
    from view_field import live_panels
    survey, watch, clock, name = make_survey(monkeypatch)
    watch.autonomy = survey
    survey.action = Action('pitch', '', cell=name, yaw=.5, pitch=12.)
    survey.observation_goal = Action('wait', '', cells=(name,), observation_x=.4, observation_y=.5,
                                     observation_yaw=.5, observation_pitch=12.)
    panels = live_panels(watch, watch.last_pose, 'static')
    text = '\n'.join(panels['left']+panels['bottom'])
    assert '本次联合观测' in text
    assert '抬头到 12 度' in text
    assert '只计一次观测' in text
    assert '俯仰复核' not in text
    assert '两档' not in text and '三档' not in text



def test_default_survey_only_requests_new_points_for_unobserved_targets(monkeypatch):
    from car_pose import turned_pose
    _, watch, clock, first = make_survey(monkeypatch)
    second = DISPLAY[1]
    for n,v in watch.votes.items():
        if n not in (first,second):
            v.locked = '1'
    watch.views[second] = CellView(None,False,True,False,False,0.,0.,0.,255.)
    survey = AutonomousSurvey(watch,clock)
    monkeypatch.setattr(survey, '_aim', lambda n: pytest.fail('默认流程不得原地逐档扫描'))
    calls = []
    target = turned_pose(CarPose(.5,0.,0.),math.radians(30.))
    def plan(*args):
        calls.append(args)
        return Action('move', '', cells=(second,), observation_x=target.x, observation_y=target.y,
                      observation_yaw=target.yaw, observation_pitch=12.)
    monkeypatch.setattr(autonomous, 'nearest_observation', plan)
    monkeypatch.setattr(autonomous, '_travel', lambda x,y,a,b,*args: (math.hypot(a-x,b-y),[(x,y),(a,b)]))
    monkeypatch.setattr(autonomous, '_bypass_points', lambda *args: [])
    survey.update()
    assert watch.votes[first].locked == '1'
    assert calls[0][3] == [second]
    assert calls[0][-2] is True
    assert calls[0][-1]  # 排除初始点和已观测位置。
    assert survey.action.kind == 'move'
    assert survey.observation_goal.cells == (second,)
    assert survey.route_display()['kinds'] == ['初始点','补充点']


def test_default_survey_does_not_fall_back_to_many_angles_when_blocked(monkeypatch):
    _, watch, clock, name = make_survey(monkeypatch)
    watch.views[name].conclusion = None
    survey = AutonomousSurvey(watch,clock)
    monkeypatch.setattr(survey, '_aim', lambda n: pytest.fail('无补充点时不得改为逐档扫描'))
    survey.update()
    assert survey.action.kind == 'blocked'
    assert survey.look_key is None
    assert survey.observation_goal is None



def test_default_point_times_out_without_long_stationary_scan(monkeypatch):
    _, watch, clock, name = make_survey(monkeypatch)
    survey = AutonomousSurvey(watch,clock)
    watch.views[name].conclusion = None
    survey.observation_goal = Action('wait', '', cells=(name,), observation_x=0., observation_y=0.,
                                    observation_yaw=0., observation_pitch=0.)
    survey.look_key, survey.look_started = (name,0), clock()
    assert not survey._expire_look(autonomous.POINT_LOOK_SECONDS-.1)
    assert survey._expire_look(autonomous.POINT_LOOK_SECONDS+.1)
    assert survey.observation_goal is None
    assert len(survey.failed_observations) == 1


def test_default_skips_new_unreliable_view_without_changing_confirmed_memory(monkeypatch):
    _,watch,clock,name = make_survey(monkeypatch)
    survey = AutonomousSurvey(watch,clock)
    monkeypatch.setattr(autonomous,'_observation_preferences',
                        lambda *args:{n:(False,1.,.7,0.) for n in DISPLAY})
    survey.update()
    assert watch.votes[name].locked is None
    watch.votes[name].locked = '2'
    survey.update()
    assert watch.votes[name].locked == '2'


def test_explicit_fallback_goal_can_use_its_own_result(monkeypatch):
    _,watch,clock,name = make_survey(monkeypatch)
    for n,v in watch.votes.items():
        if n != name:
            v.locked = '1'
    survey = AutonomousSurvey(watch,clock)
    monkeypatch.setattr(autonomous,'_observation_preferences',
                        lambda *args:{n:(False,1.,.7,0.) for n in DISPLAY})
    survey.observation_goal = Action('wait','',cells=(name,),observation_x=0.,observation_y=0.,
        observation_yaw=0.,observation_pitch=0.,observation_fallback=True)
    survey.update()
    assert watch.votes[name].locked == '1'
    assert survey.complete


def test_initial_route_is_cached_and_adds_middle_distance_only_after_last_stop(monkeypatch):
    from motion import RouteStop, RouteStops, Look
    _, watch, clock, _ = make_survey(monkeypatch)
    survey = AutonomousSurvey(watch, clock)
    calls = []
    def route(pose, offset, intr, pitch, **kwargs):
        calls.append(kwargs)
        names = kwargs['pending']
        limit = kwargs['max_stops']
        stops = RouteStops([RouteStop(pose.x,pose.y,(), '初始点')])
        for i,n in enumerate(names[:limit-1],1):
            stops.append(RouteStop(float(i),0.,(Look(autonomous.TOWER_NO[n],0.,0.),)))
        return stops, [(s.x,s.y) for s in stops]
    monkeypatch.setattr(autonomous, 'supplementary_route', route)
    monkeypatch.setattr(survey, '_advance_observation', lambda: survey.observation_goal is not None)
    survey._schedule(clock())
    assert len(survey.primary_stops) == 3
    assert len(calls) == 1
    survey._schedule(clock())
    assert len(calls) == 1  # 前往第一站时不重算、不追加。
    for i in range(2):
        for n in survey.observation_goal.cells:
            watch.votes[n].locked = '1'
        survey.observation_goal = None
        survey._schedule(clock())
        if i == 0:
            assert len(calls) == 1  # 第二站沿用初始路线。
    assert len(calls) == 2
    assert calls[-1]['medium_only'] is True
    assert calls[-1]['max_stops'] == 2
    assert all(watch.votes[n].locked is None for n in calls[-1]['pending'])
    assert len(survey.additional_stops) == 1
    assert len(survey.primary_stops) == 3


def test_completed_initial_route_does_not_append_when_all_towers_recognized(monkeypatch):
    from motion import RouteStop, RouteStops
    _, watch, clock, _ = make_survey(monkeypatch)
    survey = AutonomousSurvey(watch, clock)
    survey.primary_stops = RouteStops([RouteStop(0.,0.,())])
    survey.primary_index = 1
    for vote in watch.votes.values():
        vote.locked = '1'
    monkeypatch.setattr(autonomous, 'supplementary_route', lambda *a,**k: pytest.fail('已识别全部塔不得补点'))
    survey._schedule(clock())
    assert survey.action.kind == 'stop'
    assert not survey.additional_stops


@pytest.mark.parametrize('source', ['laptop','host'])
@pytest.mark.parametrize('x,y,yaw,pitch', [(-2.3,.1,.7,-18.), (2.65,.2,-2.1,23.), (0.,0.,1.8,7.)])
def test_confirmed_random_start_uses_actual_pose_and_pitch_for_initial_plan(monkeypatch, source, x,y,yaw,pitch):
    from motion import RouteStop, RouteStops, Look
    from view_field import prepare_start_confirmation
    _,watch,clock,_ = make_survey(monkeypatch)
    survey = AutonomousSurvey(watch,clock)
    watch.autonomy = survey
    prepare_start_confirmation(watch,source)
    assert survey.awaiting_start
    assert survey.initial_pose is None
    assert not survey.route_display()['points']
    # 确认前移动、转向、改变俯仰都不能建立路线或记住识别结果。
    watch.last_pose = CarPose(x,y,yaw)
    watch.pitch = pitch
    survey.update()
    assert survey.primary_stops is None
    assert all(v.locked is None for v in watch.votes.values())
    confirmed = watch.last_pose
    assert survey.confirm_start(confirmed,pitch)
    assert survey.initial_pose == confirmed
    assert survey.initial_pitch == pitch
    assert survey.distance_m == 0.
    assert not survey.confirm_start(CarPose(0,0,0),0.)  # B不会重置已开始的路线。
    calls = []
    def plan(pose,offset,intr,pi,**kwargs):
        calls.append((pose,pi,kwargs))
        return RouteStops([RouteStop(pose.x,pose.y,()),
                           RouteStop(pose.x+.5,pose.y,(Look(autonomous.TOWER_NO[DISPLAY[0]],0.,0.),))]), []
    monkeypatch.setattr(autonomous,'supplementary_route',plan)
    monkeypatch.setattr(survey,'_advance_observation',lambda: False)
    # 后续反馈不能悄悄覆盖按键确认的初始姿态。
    watch.last_pose = CarPose(x+.03,y+.02,yaw+.1)
    watch.pitch = pitch+2.
    survey._schedule(clock())
    assert calls[0][0] == confirmed
    assert calls[0][1] == pitch
    assert calls[0][2]['max_stops'] == 3
    assert survey.snapshot()['initial_pitch'] == pitch


def test_start_confirmation_rejects_missing_pitch_pose_or_forbidden_position(monkeypatch):
    _,watch,clock,_ = make_survey(monkeypatch)
    survey = AutonomousSurvey(watch,clock)
    survey.wait_for_start()
    for pose,pitch in ((None,0.),(CarPose(0,0,0),None),
                       (CarPose(0,0,float('nan')),0.),(CarPose(-4.,-.7,0.),5.)):
        assert not survey.confirm_start(pose,pitch)
        assert survey.awaiting_start
        assert survey.initial_pose is None
        assert survey.primary_stops is None
    assert survey.confirm_start(CarPose(-2.3,.2,-1.),-12.)


def test_cached_goal_outside_confirmed_x_limits_is_cancelled(monkeypatch):
    from field_map import right_front_x_limits
    _,watch,clock,name = make_survey(monkeypatch)
    survey = AutonomousSurvey(watch,clock)
    survey.observation_goal = Action('wait','',cells=(name,),
        observation_x=right_front_x_limits()[1]+.01,observation_y=.2,
        observation_yaw=0.,observation_pitch=0.)
    assert survey._advance_observation()
    assert survey.observation_goal is None
    assert survey.action.kind == 'blocked'
    assert watch.votes[name].locked is None


def test_camera_must_settle_before_automatic_manual_confirmation_and_observation_count(monkeypatch):
    _,watch,clock,name = make_survey(monkeypatch)
    survey = AutonomousSurvey(watch,clock)
    watch.camera_settled = False
    watch.stable_frames = 100  # 帧数足够也不能绕过持续停稳条件。
    survey.action = Action('wait','',cell=name,cells=(name,))
    assert survey.confirmation_target() is None
    assert not survey.confirm_current((name,'1'))
    survey.update()
    assert watch.votes[name].locked is None
    assert not survey.observations
    watch.camera_settled = True
    survey.update()
    assert watch.votes[name].locked == '1'
    assert len(survey.observations) == 1
    watch.camera_settled = False
    survey.update()
    assert watch.votes[name].locked == '1'  # 晃动不清除已确认结果。


def test_unsettled_camera_resets_default_look_timeout_without_losing_target(monkeypatch):
    _,watch,clock,name = make_survey(monkeypatch)
    survey=AutonomousSurvey(watch,clock)
    watch.camera_settled=False
    goal=Action('wait','',cells=(name,),observation_x=0.,observation_y=0.,
                observation_yaw=0.,observation_pitch=0.)
    survey.observation_goal=goal
    survey.look_key,survey.look_started=(name,0),clock()
    clock.now=5.
    survey.update()
    assert survey.look_key is None and survey.look_started is None
    assert survey.observation_goal is goal
    assert not survey.failed_observations


def test_b_start_plans_route_even_when_camera_not_settled(monkeypatch):
    from motion import RouteStop,RouteStops,Look
    _,watch,clock,name=make_survey(monkeypatch)
    survey=AutonomousSurvey(watch,clock)
    survey.wait_for_start()
    assert survey.confirm_start(CarPose(0.,.2,.3),12.)
    watch.camera_settled=False
    calls=[]
    def plan(pose,offset,intr,pitch,**kwargs):
        calls.append((pose,pitch))
        return RouteStops([RouteStop(pose.x,pose.y,()),
                           RouteStop(.5,.2,(Look(autonomous.TOWER_NO[name],0.,0.),))]),[(pose.x,pose.y),(.5,.2)]
    monkeypatch.setattr(autonomous,'supplementary_route',plan)
    survey.update()
    assert calls == [(survey.initial_pose,12.)]
    assert len(survey.route_display()['points']) == 2
    assert watch.votes[name].locked is None
    assert not survey.observations
    survey.update()
    assert len(calls) == 1  # 等待停稳不能反复重算路线。
