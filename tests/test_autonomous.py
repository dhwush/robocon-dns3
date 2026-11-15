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
    monkeypatch.setattr(autonomous, 'primary_route', lambda *args: ([],[]))
    monkeypatch.setattr(autonomous, 'nearest_observation', lambda *args: Action('blocked', '无安全位置'))
    def aim(n, v, p, o, pitch, color, intr, attempt):
        if attempt == 1 and abs(pitch-8.) > 1:
            return Action('pitch', '抬头到8度', pitch=8., cell=n, attempt=1)
        return Action('wait', '', cell=n, attempt=attempt)
    monkeypatch.setattr(autonomous, '_aim_cell', aim)
    survey = AutonomousSurvey(watch, clock)
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


def test_automatic_memory_conflict_and_different_view_resolution(monkeypatch, tmp_path):
    survey, watch, clock, name = make_survey(monkeypatch)
    observe(survey, clock, 7)
    assert watch.votes[name].locked is None
    assert len(survey.pitch_checks[name]) == 1
    assert survey.action.kind == 'pitch'
    watch.pitch = 8.
    observe(survey, clock, 7)
    assert watch.votes[name].locked == '1'
    assert any(item['event'] == 'confirmed' for item in survey.history[name])
    watch.views[name].conclusion = '2'
    observe(survey, clock, 9)
    assert watch.votes[name].locked == '1'
    assert watch.votes[name].recheck
    text = '\n'.join(survey.lines())
    assert '原确认：一层；新稳定识别：两层；当前复核：两层' in text
    assert '流程改为优先复核' in text
    assert '旧结果暂留，不计完成' in text
    assert survey.snapshot()['conflict_results'][name]['detected'] == '2'
    assert not survey.complete
    # 同一俯仰重复长时观察，不能补足第二档。
    observe(survey, clock, 8)
    assert watch.votes[name].locked == '1'
    watch.pitch = 0.
    observe(survey, clock, 7)
    assert watch.votes[name].locked == '2'
    assert not watch.votes[name].recheck
    assert survey.history[name][-1]['event'] == 'resolved'
    assert survey.history[name][-1]['previous'] == '1'
    output = tmp_path/'memory.json'
    survey.save(output)
    saved = json.loads(output.read_text())
    assert saved['results'][name] == '2'
    assert name not in saved['conflict_results']
    assert '冲突已解决：一层→两层' in '\n'.join(survey.lines())
    assert not saved['complete']


def test_missing_frames_and_movement_do_not_accumulate_confirmation(monkeypatch):
    survey, watch, clock, name = make_survey(monkeypatch)
    observe(survey, clock, 3)
    watch.have_color = False
    observe(survey, clock, 1)
    watch.have_color = True
    observe(survey, clock, 3)
    assert watch.votes[name].locked is None
    watch.last_pose = CarPose(.05, 0, 0)
    observe(survey, clock, 3)
    assert watch.votes[name].locked is None
    observe(survey, clock, 4)
    assert watch.votes[name].locked is None
    watch.pitch = 8.
    observe(survey, clock, 7)
    assert watch.votes[name].locked == '1'


def test_unknown_keeps_task_open_instead_of_finishing(monkeypatch):
    survey, watch, clock, name = make_survey(monkeypatch)
    watch.views[name].conclusion = None
    monkeypatch.setattr(autonomous, '_aim_cell', lambda n,v,p,o,pi,c,i,a: Action('wait', '', cell=n, attempt=a))
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


def test_final_summary_only_after_every_tower_has_no_conflict(monkeypatch):
    survey, watch, clock, name = make_survey(monkeypatch)
    for vote in watch.votes.values():
        vote.locked = '1'
    survey.update()
    assert survey.complete
    assert '最终结果' in survey.lines()
    watch.votes[name].recheck = True
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
    assert action.kind == 'move'
    assert math.isclose(math.hypot(action.final_x, action.final_y), .1)
    assert set(action.cells) == set(DISPLAY)
    monkeypatch.setattr(motion, '_travel', lambda *args: None)
    assert motion.nearest_observation(CarPose(0,0,0), np.zeros(2), None, list(DISPLAY)).kind == 'blocked'


def test_field_watch_live_supervisor_bypasses_short_vote_lock(monkeypatch):
    from car_pose import Setup
    from report import FieldWatch
    import report
    survey, fake, clock, name = make_survey(monkeypatch)
    watch = FieldWatch(Setup(True, np.zeros(2), 0., '', 'rad'), fake.intr)
    watch.enable_autonomy(clock)
    monkeypatch.setattr(report, 'inspect_frame', lambda *args: fake.views)
    color = np.zeros((2, 2, 3), np.uint8)
    for _ in range(30):
        clock.now += .05
        watch.update(fake.last_pose, color)
    assert watch.votes[name].locked is None
    for _ in range(140):
        clock.now += .05
        watch.update(fake.last_pose, color)
    assert watch.votes[name].locked is None
    watch.pitch = 8.
    for _ in range(150):
        clock.now += .05
        watch.update(fake.last_pose, color)
    assert watch.votes[name].locked == '1'
    assert any('1号' in line for line in watch.lines('basic'))
    assert watch.autonomy.history[name]


def test_parallax_request_finishes_stationary_angles_before_moving(monkeypatch):
    survey, watch, clock, name = make_survey(monkeypatch)
    watch.views[name].conclusion = None
    watch.views[name].needs_parallax = True
    observe(survey, clock, 1)
    assert survey.action.kind == 'wait'
    observe(survey, clock, autonomous.LOOK_SECONDS + 1)
    assert survey.action.kind == 'pitch'
    assert survey.attempts[name] == 1


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


def test_pitch_disagreement_discards_old_evidence(monkeypatch):
    survey, watch, clock, name = make_survey(monkeypatch)
    observe(survey, clock, 7)
    watch.pitch = 8.
    watch.views[name].conclusion = '2'
    observe(survey, clock, 7)
    assert watch.votes[name].locked is None
    assert survey.pitch_checks[name] == [dict(result='2',pitch=8.)]
    watch.pitch = -8.
    observe(survey, clock, 7)
    assert watch.votes[name].locked == '2'


def test_pitch_candidates_do_not_accumulate_across_positions(monkeypatch):
    survey, watch, clock, name = make_survey(monkeypatch)
    observe(survey, clock, 7)
    watch.last_pose = CarPose(.11,0,0)
    watch.pitch = 8.
    observe(survey, clock, 7)
    assert watch.votes[name].locked is None
    assert len(survey.pitch_checks[name]) == 1


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


def test_window_shows_pitch_sequence_and_cross_check_progress(monkeypatch):
    survey, watch, clock, name = make_survey(monkeypatch)
    observe(survey, clock, 7)
    assert '停止！' in '\n'.join(survey.lines())
    watch.pitch = 8.
    clock.now += autonomous.STOP_HOLD_SECONDS + .1
    survey.update()
    text = '\n'.join(survey.lines())
    assert '俯仰复核1/2档' in text
    assert '每个朝向的相机姿态' in text
    assert '抬头到 8 度' in text and '低头到 8 度' in text


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
    # 共享点有绕障代价时，不能只看直线距离或覆盖数量。
    monkeypatch.setattr(motion, '_travel', lambda x,y,a,b,*args: (1. if a>.15 else .1,[(x,y),(a,b)]))
    action = motion.nearest_observation(pose,np.zeros(2),None,names)
    assert math.isclose(action.final_x,.1)
    assert action.cells == (names[0],)
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


def test_observation_jitter_does_not_clear_evidence_but_real_motion_does(monkeypatch):
    survey, watch, clock, name = make_survey(monkeypatch)
    observe(survey, clock, 3)
    watch.last_pose = CarPose(.025, 0, math.radians(2.5))
    watch.pitch = 2.5
    watch.roll = 2.5
    observe(survey, clock, .1)
    assert survey.evidence[name].windows == 1
    watch.pitch = 6.
    observe(survey, clock, .1)
    assert survey.evidence[name].windows == 0


def test_conflict_window_explains_movement_and_blocked_followup(monkeypatch):
    survey, watch, clock, name = make_survey(monkeypatch)
    watch.votes[name].locked = '红'
    watch.votes[name].recheck = True
    survey.conflict_results[name] = dict(previous='红',detected='蓝',candidate='红')
    survey.action = Action('move', '换位', x=1., y=0., cell=name)
    text = '\n'.join(survey.lines())
    assert '原确认：红顶；新稳定识别：蓝顶；当前复核：红顶' in text
    assert '换到新观察点' in text
    survey.action = Action('blocked', '等待')
    assert '保留冲突，等待条件变化后重试' in '\n'.join(survey.lines())


def test_far_rgb_is_tried_here_and_two_pitches_remove_move_task(monkeypatch):
    from field_map import tower_cells
    from car_pose import camera_position
    survey, watch, clock, name = make_survey(monkeypatch)
    cell = next(c for c in tower_cells() if c.name == name)
    watch.last_pose = CarPose(cell.x,cell.y-4.5,0.)
    monkeypatch.setattr(autonomous, 'fits_when_aimed', lambda *args: False)
    monkeypatch.setattr(autonomous, 'fits_rgb_when_aimed', lambda c,*args: c.name == name)
    watch.views[name].conclusion = '红'
    watch.views[name].method = 'RGB顶块'
    for n,vote in watch.votes.items():
        if n != name:
            vote.locked = '1'
    observe(survey,clock,7)
    assert watch.votes[name].locked is None
    assert survey.action.kind == 'pitch'
    watch.pitch = 8.
    observe(survey,clock,7)
    assert watch.votes[name].locked is None
    watch.pitch = -8.
    observe(survey,clock,7)
    assert watch.votes[name].locked == '红'
    assert survey.action.kind == 'stop'
    assert survey.complete


def test_far_rgb_opportunity_cannot_confirm_empty_or_white_only(monkeypatch):
    from field_map import tower_cells
    survey, watch, clock, name = make_survey(monkeypatch)
    cell = next(c for c in tower_cells() if c.name == name)
    watch.last_pose = CarPose(cell.x,cell.y-4.5,0.)
    monkeypatch.setattr(autonomous,'fits_rgb_when_aimed',lambda c,*args: c.name == name)
    for value in ('0','1','2'):
        watch.views[name].conclusion = value
        observe(survey,clock,7)
        watch.pitch += 8.
        observe(survey,clock,7)
        assert watch.votes[name].locked is None
        assert not survey.pitch_checks[name]


@pytest.mark.parametrize('value', ['1', '2'])
def test_verified_far_white_accumulates_three_pitch_checks(monkeypatch, value):
    from field_map import tower_cells
    survey, watch, clock, name = make_survey(monkeypatch)
    cell = next(c for c in tower_cells() if c.name == name)
    watch.last_pose = CarPose(cell.x,cell.y-4.5,0.)
    view = watch.views[name]
    view.conclusion, view.method = value, '远处白盒'
    view.measurements = {'far_white': {'upper_clear': True}}
    observe(survey,clock,7)
    assert len(survey.pitch_checks[name]) == 1
    watch.pitch = 8.
    observe(survey,clock,7)
    assert len(survey.pitch_checks[name]) == 2
    assert watch.votes[name].locked is None
    watch.pitch = -8.
    observe(survey,clock,7)
    assert watch.votes[name].locked == value


def test_far_white_without_upper_space_verification_is_not_counted(monkeypatch):
    from field_map import tower_cells
    survey, watch, clock, name = make_survey(monkeypatch)
    cell = next(c for c in tower_cells() if c.name == name)
    watch.last_pose = CarPose(cell.x,cell.y-4.5,0.)
    watch.views[name].method = '远处白盒'
    watch.views[name].measurements = {'far_white': {'upper_clear': False}}
    observe(survey,clock,7)
    assert not survey.pitch_checks[name]


def test_far_cap_pixel_budget_enables_opportunity_without_changing_depth_range():
    import motion
    from field_map import tower_cells
    cell = tower_cells()[0]
    offset = np.zeros(2)
    intr = load_intrinsics('config/intrinsics.yaml')
    pose = CarPose(cell.x,cell.y-4.5,0.)
    assert not motion.fits_when_aimed(cell,pose,offset,intr)
    assert motion.fits_rgb_when_aimed(cell,pose,offset,intr)
    assert not motion.fits_rgb_when_aimed(cell,CarPose(cell.x,cell.y-40.,0.),offset,intr)
    assert motion.DEPTH_FAR_M == 3.


def test_primary_stations_run_before_far_fallback(monkeypatch):
    from motion import RouteStop
    survey, watch, clock, name = make_survey(monkeypatch)
    for n,vote in watch.votes.items():
        if n != name:
            vote.locked = '1'
    watch.views[name].conclusion = None
    survey.primary_stops = [RouteStop(0,0,(),'A起点'),RouteStop(.5,0,(),'沿途复看'),RouteStop(1,0,(),'B近点')]
    survey.primary_line = [(0,0),(1,0)]
    survey.attempts[name] = autonomous.MAX_TRIES
    monkeypatch.setattr(autonomous,'fits_rgb_when_aimed',lambda c,*args: c.name == name)
    monkeypatch.setattr(autonomous,'_travel',lambda x,y,a,b,*args: (abs(a-x),[(x,y),(a,b)]))
    def no_fallback(*args):
        raise AssertionError('主线尚未走完，不应提前去远处')
    monkeypatch.setattr(autonomous,'nearest_observation',no_fallback)
    survey._schedule(clock())
    assert survey.action.kind == 'move'
    assert survey.action.chase_to == 'primary'
    assert survey.action.final_x == .5
    # 已在沿途点停稳：重新累计原地观察，不直接跳到远端。
    survey.anchor = CarPose(0,0,0)
    survey.anchor_pitch = survey.anchor_roll = 0.
    watch.last_pose = CarPose(.5,0,0)
    observe(survey,clock,autonomous.STOP_HOLD_SECONDS + .3)
    assert survey.primary_index == 2
    assert survey.action.kind == 'wait'


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
    survey.update()
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
    survey.update()
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
    assert survey.action.kind == 'wait'
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


def test_reserve_runs_after_primary_before_fallback_and_only_once(monkeypatch):
    from motion import RouteStop
    survey, watch, clock, name = make_survey(monkeypatch)
    for n, vote in watch.votes.items():
        if n != name:
            vote.locked = '1'
    survey.primary_stops = [RouteStop(0,0,(),'A起点')]
    survey.reserve_stops = [RouteStop(.5,0,(),'沿途备用')]
    survey.attempts[name] = autonomous.MAX_TRIES
    monkeypatch.setattr(autonomous, 'fits_rgb_when_aimed', lambda c,*args: c.name == name)
    monkeypatch.setattr(autonomous, '_travel', lambda x,y,a,b,*args: (abs(a-x), [(x,y),(a,b)]))
    survey._schedule(clock())
    assert survey.action.kind == 'move'
    assert survey.action.chase_to == 'reserve'
    assert survey.action.cells == (name,)
    watch.last_pose = CarPose(.5,0,0)
    survey.action = None
    survey._schedule(clock())
    assert survey.reserve_index == 1
    assert survey.action.kind == 'blocked'
    assert survey.route_display()['reserve_points'] == [(.5,0)]


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
