import os
from types import SimpleNamespace

from terminal_ui import LiveUI,terminal_results
from field_map import DISPLAY
import pytest


def test_terminal_reads_commands_and_eof_without_graphical_backend():
    read_fd,write_fd=os.pipe()
    os.write(write_fd,b'b\nq\n')
    os.close(write_fd)
    with os.fdopen(read_fd) as stream:
        ui=LiveUI(None,True,stream)
        ui.namedWindow('camera')
        ui.imshow('camera',None)
        assert ui.waitKey(0)==ord('b')
        assert ui.waitKey(0)==ord('q')
        assert ui.waitKey(0)==ord('q')


def test_terminal_does_not_reuse_results_when_current_views_are_missing():
    votes={n:SimpleNamespace(locked='2',recheck=True) for n in DISPLAY}
    rows=terminal_results(SimpleNamespace(views=None,votes=votes))
    assert len(rows)==6
    assert all('本帧=未知' in row and '已确认=未确认' in row for row in rows)


@pytest.mark.parametrize('ssh,headless,expected',[(False,False,'laptop'),
    (False,True,'host'),(True,False,'host')])
def test_cli_selects_local_radar_for_headless_host(monkeypatch,ssh,headless,expected):
    import view_field
    import sys
    seen=[]
    monkeypatch.setattr(sys,'argv',['view_field.py','extend']+(['--headless'] if headless else []))
    monkeypatch.delenv('DISPLAY',raising=False)
    if ssh:
        monkeypatch.setenv('SSH_CONNECTION','test')
    else:
        monkeypatch.delenv('SSH_CONNECTION',raising=False)
    monkeypatch.setattr(view_field,'run',lambda args: seen.append(args))
    view_field.main()
    assert seen[0].pose_source==expected
    assert seen[0].headless==(headless or ssh)


def test_mode3_map_entry_runs_host_without_a_map_window(monkeypatch):
    import cv2
    import map_test
    import view_field
    import sys
    seen=[]
    monkeypatch.setattr(sys,'argv',['map_test.py','--pose-source','host','--headless'])
    monkeypatch.setattr(cv2,'namedWindow',lambda *a: pytest.fail('不应创建地图窗口'))
    monkeypatch.setattr(view_field,'run',lambda args: seen.append(args))
    map_test.main()
    assert seen[0].pose_source=='host'
    assert seen[0].mode=='extend' and seen[0].headless


def test_laptop_launcher_runs_camera_and_radar_on_remote_host(monkeypatch,tmp_path):
    import subprocess
    from pathlib import Path
    ssh=tmp_path/'ssh'
    ssh.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    ssh.chmod(0o755)
    monkeypatch.setenv('PATH',str(tmp_path)+os.pathsep+os.environ['PATH'])
    monkeypatch.setenv('TASK3_REMOTE_DIR','project with spaces')
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run(['/bin/bash',str(root/'run_task3_ssh.sh'),'user@host','static'],
                          check=True,capture_output=True,text=True)
    assert 'user@host' in result.stdout and '-tt' in result.stdout
    assert 'static --headless --pose-source host' in result.stdout
    assert 'project\\ with\\ spaces' in result.stdout


def dashboard_watch():
    from car_pose import CarPose
    from motion import Action
    votes={n:SimpleNamespace(locked=None,recheck=False) for n in DISPLAY}
    goal=Action('wait','',cells=(DISPLAY[0],),observation_x=.5,observation_y=.4,
                observation_yaw=.6,observation_pitch=12.)
    survey=SimpleNamespace(action=Action('move','向右平移',x=.2,y=.1),observation_goal=goal,
        awaiting_start=False,complete=False,initial_pose=CarPose(0,0,0),initial_pitch=4.,
        stop_until=None,message='到达目标后观察',metrics=lambda:dict(observation_count=1,distance_m=.5))
    return SimpleNamespace(autonomy=survey,votes=votes,pitch=4.,last_pose=CarPose(0,0,0))


def test_panel_shows_current_pose_safe_waypoint_and_full_observation_goal():
    from terminal_ui import terminal_status
    from car_pose import CarPose
    watch=dashboard_watch()
    rows,_,_=terminal_status(watch,CarPose(.1,.15,.3),5.)
    text='\n'.join(rows)
    assert '当前位置：x=+0.100 m  y=+0.150 m  yaw=+17.2°  俯仰=+5.0°' in text
    assert '下一位置：x=+0.200 m  y=+0.100 m' in text
    assert '预览，不可直接前往）：x=+0.500 m  y=+0.400 m  yaw=+34.4°  俯仰=+12.0°' in text
    assert '本步操作：移动到本步要求的 x/y' in text
    assert '下一位置观察目标：1号（未完成）' in text


def test_yaw_requirement_uses_rotated_right_front_corner():
    from terminal_ui import terminal_status,_pose_text
    from car_pose import CarPose,turned_pose
    from motion import Action
    watch=dashboard_watch()
    watch.autonomy.action=Action('yaw','',yaw=.8,pitch=12.)
    current=CarPose(.1,.2,0.)
    rows,_,_=terminal_status(watch,current,4.)
    assert '下一位置：'+_pose_text(turned_pose(current,.8),4.) in rows


def test_dashboard_refreshes_same_screen_then_identification_and_summary_once():
    from terminal_ui import TerminalDashboard
    from io import StringIO
    watch=dashboard_watch()
    output=StringIO()
    panel=TerminalDashboard(output,interactive=True)
    panel.show(watch,watch.last_pose,4.)
    panel.show(watch,watch.last_pose,5.)
    assert output.getvalue().count('\x1b[?1049h')==1
    assert output.getvalue().count('\x1b[H\x1b[J')==2
    watch.votes[DISPLAY[0]].locked='2'
    panel.show(watch,watch.last_pose,5.)
    assert '识别好了：1号=2；请查看本步操作' in output.getvalue()
    panel.finish(watch)
    final=output.getvalue().split('\x1b[?1049l')[-1]
    assert final.count('最终汇总')==1
    assert '1号：2' in final and '6号：未确认' in final
    assert '观测1次；累计路程0.500米' in final


def test_redirected_status_does_not_repeat_when_only_feedback_coordinates_change():
    from terminal_ui import TerminalDashboard
    from car_pose import CarPose
    from io import StringIO
    output=StringIO()
    panel=TerminalDashboard(output,interactive=False)
    watch=dashboard_watch()
    panel.show(watch,CarPose(0,0,0),4.)
    before=output.getvalue()
    panel.show(watch,CarPose(.1,.2,.3),5.)
    assert output.getvalue()==before
    watch.votes[DISPLAY[0]].locked='1'
    panel.show(watch,CarPose(.1,.2,.3),5.)
    assert '识别好了：1号=1' in output.getvalue()[len(before):]


def test_missing_pose_and_blocked_path_do_not_suggest_return_to_start():
    from terminal_ui import terminal_status
    from motion import Action
    watch=dashboard_watch()
    watch.autonomy.action=Action('blocked','无安全路径')
    rows,_,_=terminal_status(watch,None,None)
    text='\n'.join(rows)
    assert '暂停识别和确认' in text
    assert '下一位置：暂无移动要求' in text
    assert '已确认：无' in text


@pytest.mark.parametrize('headless', [True, False])
def test_real_tty_accepts_single_key_without_enter_and_restores_terminal(headless):
    import pty
    import termios
    master,slave=pty.openpty()
    try:
        with os.fdopen(slave) as stream:
            before=termios.tcgetattr(stream.fileno())
            backend=SimpleNamespace(waitKey=lambda delay:-1)
            ui=LiveUI(backend,headless,stream)
            ui.start_input()
            os.write(master,b'b')
            assert ui.waitKey(100)==ord('b')
            os.write(master,b'q')
            assert ui.waitKey(100)==ord('q')
            ui.close_input()
            assert termios.tcgetattr(stream.fileno())==before
    finally:
        os.close(master)


def test_graphical_key_works_with_terminal_input_enabled():
    import pty
    master,slave=pty.openpty()
    try:
        with os.fdopen(slave) as stream:
            ui=LiveUI(SimpleNamespace(waitKey=lambda delay:ord('c')),False,stream)
            ui.start_input()
            try:
                assert ui.waitKey(1)==ord('c')
            finally:
                ui.close_input()
    finally:
        os.close(master)


@pytest.mark.parametrize('headless', [True, False])
def test_ssh_controlling_terminal_with_redirected_stdin(headless):
    """独立会话模拟SSH的PTY，stdin重定向时仍从控制终端读取。"""
    import pty
    import select
    import signal
    pid, master = pty.fork()
    if pid == 0:
        try:
            import sys
            sys.stdin = open(os.devnull)
            ui = LiveUI(SimpleNamespace(waitKey=lambda delay:255),headless)
            ui.start_input()
            assert ui.saved_terminal is not None
            os.write(1,b'INPUT_READY\n')
            received=[]
            import time
            deadline=time.monotonic()+5
            while len(received)<7 and time.monotonic()<deadline:
                key=ui.waitKey(10)
                if key != -1:
                    received.append(chr(key))
            ui.close_input()
            os._exit(0 if ''.join(received)=='bd12345' else 2)
        except BaseException:
            os._exit(3)
    try:
        assert select.select([master],[],[],5)[0], '终端初始化超时'
        assert b'INPUT_READY' in os.read(master,4096)
        os.write(master,b'bd12345')
        import time
        deadline=time.monotonic()+7
        while time.monotonic()<deadline:
            done,status=os.waitpid(pid,os.WNOHANG)
            if done:
                assert os.waitstatus_to_exitcode(status)==0
                pid=None
                break
            time.sleep(.01)
        else:
            pytest.fail('SSH按键接收超时')
    finally:
        os.close(master)
        if pid is not None:
            os.kill(pid,signal.SIGKILL)
            os.waitpid(pid,0)


def test_operator_panel_wraps_yaw_and_separates_plan_from_command():
    import math
    from car_pose import CarPose
    from terminal_ui import terminal_status
    from dataclasses import replace
    watch=dashboard_watch()
    watch.autonomy.observation_goal=replace(watch.autonomy.observation_goal,
                                          observation_yaw=math.radians(-318.8))
    text='\n'.join(terminal_status(watch,CarPose(.1,.15,.3),5.)[0])
    assert 'yaw=+41.2°' in text
    assert '该目标最终观测姿态（预览，不可直接前往）：x=' in text
    assert 'Δx=+0.100 m  Δy=-0.050 m' in text
    assert '向右平移' not in text
    assert '提示：到达目标后观察' not in text


def test_missing_pose_overrides_start_and_hold_commands():
    from terminal_ui import terminal_status
    watch=dashboard_watch()
    watch.autonomy.awaiting_start=True
    watch.autonomy.stop_until=123.
    text='\n'.join(terminal_status(watch,None,None)[0])
    assert '状态：定位无效' in text
    assert '本步操作：停止！等待定位恢复' in text
    assert '下一位置：暂无移动要求' in text


def test_yaw_command_explains_corner_motion():
    from terminal_ui import terminal_status
    from motion import Action
    watch=dashboard_watch()
    watch.autonomy.action=Action('yaw','',yaw=.8)
    text='\n'.join(terminal_status(watch,watch.last_pose,4.)[0])
    assert '绕车身中心原地转' in text
    assert '右前角 x/y 会随转动变化' in text
    assert '位置保持不动' not in text


def test_inferred_zero_never_announced_as_identification_success():
    from io import StringIO
    from terminal_ui import TerminalDashboard,terminal_status
    watch=dashboard_watch()
    watch.autonomy.fallback_zero={DISPLAY[0]:'兜底'}
    watch.votes[DISPLAY[0]].locked='0'
    output=StringIO()
    panel=TerminalDashboard(output,interactive=False)
    panel.show(watch,watch.last_pose,4.)
    assert '识别好了：1号' not in output.getvalue()
    panel.finish(watch)
    assert '1号：兜底推定0层（非识别确认）' in output.getvalue()


def test_panel_distinguishes_current_observation_and_last_outcome():
    from terminal_ui import terminal_status
    watch=dashboard_watch()
    watch.autonomy.observation_status='本点尚未识别成功'
    watch.autonomy.last_observation_outcome='上一点已确认2号'
    text='\n'.join(terminal_status(watch,watch.last_pose,4.)[0])
    assert '当前观测结果：本点尚未识别：请先完成上方本步操作' in text
    assert '最近观测结论：上一点已确认2号' in text
    assert '本帧=' not in text


@pytest.mark.parametrize('kind', ['yaw','pitch'])
@pytest.mark.parametrize('error,ok', [(2.99,True),(3.,True),(-3.,True),(3.01,False),(-3.01,False)])
def test_panel_angle_deadband_matches_arrival(kind,error,ok):
    import math
    from car_pose import CarPose
    from motion import Action
    from terminal_ui import terminal_status
    watch=dashboard_watch()
    watch.autonomy.action=Action(kind,'',yaw=0.,pitch=0.)
    rows,_,_=terminal_status(watch,CarPose(0,0,math.radians(error)),error)
    label='旋转' if kind=='yaw' else '俯仰'
    status=next(r for r in rows if r.startswith(label+'：'))
    command=next(r for r in rows if r.startswith('本步操作：'))
    assert ('可以了' in status) is ok
    assert ('可以了' in command) is ok


def test_panel_missing_pitch_does_not_claim_arrival():
    from motion import Action
    from terminal_ui import terminal_status
    watch=dashboard_watch()
    watch.autonomy.action=Action('pitch','',pitch=0.)
    rows,_,_=terminal_status(watch,watch.last_pose,None)
    assert '俯仰：实测无效，等待反馈' in rows


def test_completed_run_does_not_label_last_observation_as_initial_point():
    from terminal_ui import terminal_status
    watch=dashboard_watch()
    watch.autonomy.complete=True
    watch.autonomy.observation_goal=None
    watch.autonomy.action=None
    text='\n'.join(terminal_status(watch,watch.last_pose,4.)[0])
    assert '本轮已结束，无后续观测位置' in text
    assert '最终观测位置：当前起点' not in text


def test_observation_targets_show_completion_and_start_state():
    from terminal_ui import terminal_status
    from motion import Action
    watch=dashboard_watch()
    watch.autonomy.action=Action('wait','',cells=(DISPLAY[0],))
    watch.votes[DISPLAY[0]].locked='2'
    text='\n'.join(terminal_status(watch,watch.last_pose,4.)[0])
    assert '当前位置观察目标：1号（已完成）' in text
    watch.votes[DISPLAY[0]].recheck=True
    text='\n'.join(terminal_status(watch,watch.last_pose,4.)[0])
    assert '当前位置观察目标：1号（未完成）' in text
    watch.autonomy.awaiting_start=True
    text='\n'.join(terminal_status(watch,watch.last_pose,4.)[0])
    assert '下一位置：等待起点确认' in text
    assert '下一位置观察目标：等待起点确认后规划' in text
    assert '本站观察：' not in text and '最终观测位置：' not in text


@pytest.mark.parametrize('data_ready',[True,False])
def test_empty_start_never_commands_wait_for_recognition(data_ready):
    from terminal_ui import terminal_status
    watch=dashboard_watch()
    watch.autonomy.action=None
    watch.autonomy.observation_goal=None
    watch.have_color=data_ready
    watch.views={n:SimpleNamespace(in_view=False,needs_yaw=False,needs_pitch=False)
                 for n in DISPLAY} if data_ready else {}
    rows,_,_=terminal_status(watch,watch.last_pose,4.)
    text='\n'.join(rows)
    assert '等待当前点识别完成' not in text
    assert '当前点继续观察' not in text
    assert '旋转：可以了' not in text
    if data_ready:
        assert '下一位置：正在规划' in text
        assert '跳过起点识别' in text
    else:
        assert '等待有效视野数据' in text
        assert '起点无可观测塔' not in text


@pytest.mark.parametrize('kind,label',[('move','本段移动'),('yaw','本次旋转'),('pitch','本次俯仰调整')])
def test_transition_hold_is_not_presented_as_observation_arrival(kind,label):
    from motion import Action
    from terminal_ui import terminal_status
    watch=dashboard_watch()
    watch.autonomy.action=Action(kind,'',x=0.,y=0.,yaw=0.,pitch=4.)
    watch.autonomy.stop_until=123.
    text='\n'.join(terminal_status(watch,watch.last_pose,4.)[0])
    assert '状态：'+label+'已到位，短暂停顿后继续下一步' in text
    assert '本步操作：结束本次调整，短暂停顿' in text
    assert '状态：已到位，停止！' not in text
    assert '尚未开始识别' in text


def test_wait_reports_stability_before_recognition():
    from motion import Action
    from terminal_ui import terminal_status
    watch=dashboard_watch()
    watch.autonomy.action=Action('wait','')
    watch.camera_settled=False
    watch.confirmation_allowed=False
    text='\n'.join(terminal_status(watch,watch.last_pose,4.)[0])
    assert '等待相机停稳及有效数据' in text
    assert '正在识别' not in text


def test_hold_has_no_competing_angle_instructions():
    from motion import Action
    from terminal_ui import terminal_status
    watch=dashboard_watch()
    watch.autonomy.action=Action('yaw','',yaw=1.,pitch=30.)
    watch.autonomy.stop_until=123.
    text='\n'.join(terminal_status(watch,watch.last_pose,4.)[0])
    assert '旋转：' not in text and '俯仰：' not in text
    assert '当前观测结果：本步调整完成' in text


def test_completed_targets_remain_at_current_pose_until_departure():
    from terminal_ui import TerminalDashboard
    from car_pose import CarPose
    from io import StringIO
    watch=dashboard_watch()
    output=StringIO()
    panel=TerminalDashboard(output,interactive=True)
    panel.show(watch,watch.last_pose,4.)
    watch.votes[DISPLAY[0]].locked='2'
    panel.show(watch,watch.last_pose,4.)
    assert '当前位置观察目标：1号（已完成）' in output.getvalue().split('\x1b[H\x1b[J')[-1]
    panel.show(watch,CarPose(.1,0,0),4.)
    assert '当前位置观察目标：1号（已完成）' not in output.getvalue().split('\x1b[H\x1b[J')[-1]
