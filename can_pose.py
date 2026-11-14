"""SocketCAN 位姿流；SSH 只传输原始 CAN 帧，时间统一使用接收端单调时钟。"""
from __future__ import annotations

import math
import shlex
import struct
import subprocess
import sys
import threading
import time
from collections import deque

from car_pose import CarPose

# 仅使用 Python 标准库，远端无需安装 python-can 或 candump。
CAN_SETUP = '''import json,os,subprocess,sys
def setup_can(interface, interactive=False):
 result = subprocess.run(['ip','-details','-json','link','show','dev',interface],
                         check=True,capture_output=True,text=True)
 links = json.loads(result.stdout)
 if len(links) != 1:
  raise RuntimeError('CAN interface not found: ' + interface)
 link = links[0]
 info = link.get('linkinfo', {})
 if info.get('info_kind') != 'can':
  raise RuntimeError('Not a CAN interface: ' + interface)
 data = info.get('info_data', {})
 if ('UP' in link.get('flags', [])
     and data.get('bittiming', {}).get('bitrate') == 1000000
     and data.get('state') not in ('BUS-OFF', 'STOPPED')):
  print('CAN ready: ' + interface + ' 1000000 bit/s (CAN1 Classic)', file=sys.stderr,flush=True)
  return
 prefix = [] if os.geteuid() == 0 else (['sudo'] if interactive else ['sudo','-n'])
 for args in (['down'], ['type','can','bitrate','1000000'], ['up']):
  try:
   subprocess.run(prefix + ['ip','link','set','dev',interface] + args, check=True)
  except subprocess.CalledProcessError as error:
   raise RuntimeError('CAN setup failed. On the CAN host run: sudo ip link set dev '
                      + interface + ' down; sudo ip link set dev ' + interface
                      + ' type can bitrate 1000000; sudo ip link set dev ' + interface
                      + ' up. Then reopen the camera.'
                      + (' Check sudo authentication and CAN interface errors.' if interactive
                         else ' Automatic setup requires sudo -n permission.')) from error
 print('CAN configured: ' + interface + ' 1000000 bit/s (CAN1 Classic)', file=sys.stderr,flush=True)
'''

READER = CAN_SETUP + '''import socket,struct
setup_can(sys.argv[1])
s=socket.socket(socket.AF_CAN,socket.SOCK_RAW,socket.CAN_RAW)
s.bind((sys.argv[1],))
while True:
 data=s.recv(16)
 ident,size,payload=struct.unpack("=IB3x8s",data)
 if ident in (0xAA,0xAB) and size==8:
  print("%X %s"%(ident,payload.hex()),flush=True)
'''


def align_can_pose(raw: CarPose, raw_origin: CarPose, field_origin: CarPose) -> CarPose:
    """用起始位姿建立 CAN 到场地的刚体变换，保留接收时间。"""
    angle = field_origin.yaw - raw_origin.yaw
    c, s = math.cos(angle), math.sin(angle)
    dx, dy = raw.x - raw_origin.x, raw.y - raw_origin.y
    yaw = math.atan2(math.sin(raw.yaw + angle), math.cos(raw.yaw + angle))
    return CarPose(field_origin.x + c * dx - s * dy,
                   field_origin.y + s * dx + c * dy, yaw, raw.stamp)


class PoseDecoder:
    def __init__(self):
        self.xy = None

    def feed(self, line: str, now: float) -> CarPose | None:
        try:
            ident, payload = line.split()
            ident = int(ident, 16)
            data = bytes.fromhex(payload)
            if len(data) != 8:
                return None
            first, second = struct.unpack('<ff', data)
            if not all(map(math.isfinite, (first, second, now))):
                return None
        except (ValueError, struct.error):
            return None
        if ident == 0xAA:
            self.xy = (first / 1000.0, second / 1000.0, now)
        elif ident == 0xAB and self.xy is not None:
            x, y, stamp = self.xy
            self.xy = None
            if 0 <= now - stamp <= 0.1:
                return CarPose(x, y, first, now)
        return None


class CanPoseStream:
    def __init__(self, source='ssh', interface='can0', host='volleyball@192.168.3.194'):
        if source == 'ssh':
            ssh = ['ssh', '-o', 'ConnectTimeout=8', '-o', 'ServerAliveInterval=2',
                   '-o', 'ServerAliveCountMax=3']
            setup = CAN_SETUP + '\nsetup_can(sys.argv[1], interactive=True)\n'
            print('正在初始化远端 CAN；如需认证，请在此终端输入 SSH / sudo 密码。',
                  file=sys.stderr, flush=True)
            try:
                # 初始化继承终端输入输出；PTY 让远端 sudo 能正常询问密码。
                # 读取另用无 PTY 的连接，避免提示或终端回显混入 CAN 帧。
                subprocess.run(ssh + ['-t', host,
                    'python3 -u -c ' + shlex.quote(setup) + ' ' + shlex.quote(interface)],
                    check=True)
            except subprocess.CalledProcessError as error:
                raise RuntimeError('SSH CAN 初始化失败，请查看终端中的连接、密码或接口错误；'
                                   '修复后重新打开相机。') from error
            command = ssh + [host,
                       'python3 -u -c ' + shlex.quote(READER) + ' ' + shlex.quote(interface)]
        else:
            command = [sys.executable, '-u', '-c', READER, interface]
        # SSH 密码由终端提示输入，不保存进源码或命令参数。
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.samples = deque(maxlen=100)
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        decoder = PoseDecoder()
        for line in self.process.stdout:
            pose = decoder.feed(line, time.monotonic())
            if pose is not None:
                with self.lock:
                    self.samples.append(pose)

    def match(self, stamp: float, max_delta=0.15):
        with self.lock:
            samples = list(self.samples)
        if not samples:
            return None
        pose = min(samples, key=lambda item: abs(item.stamp - stamp))
        if time.monotonic() - pose.stamp > max_delta or abs(pose.stamp - stamp) > max_delta:
            return None
        return pose

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.thread.join(timeout=1)
        self.process.stdout.close()
