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
READER = '''import socket,struct,sys
s=socket.socket(socket.AF_CAN,socket.SOCK_RAW,socket.CAN_RAW)
s.bind((sys.argv[1],))
while True:
 data=s.recv(16)
 ident,size,payload=struct.unpack("=IB3x8s",data)
 if ident in (0xAA,0xAB) and size==8:
  print("%X %s"%(ident,payload.hex()),flush=True)
'''


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
            command = ['ssh', '-o', 'ConnectTimeout=8', '-o', 'ServerAliveInterval=2',
                       '-o', 'ServerAliveCountMax=3', host,
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
