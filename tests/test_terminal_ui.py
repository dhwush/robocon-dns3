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
