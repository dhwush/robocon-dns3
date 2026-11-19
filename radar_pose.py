"""雷达位姿接入：使用测量时间，统一输出场地右前角（米、弧度）。"""
from __future__ import annotations

from car_pose import display_yaw_deg

import json
import math
import shlex
import subprocess
import threading
import time
from collections import deque
from urllib.request import urlopen

from car_pose import BODY_HALF_M, CarPose, closest_pose, lidar_pose_to_corner


def convert_reference(pose, reference):
    if reference == 'lidar':
        return lidar_pose_to_corner(pose)
    if reference == 'center':
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        return CarPose(pose.x + BODY_HALF_M*(c-s), pose.y + BODY_HALF_M*(s+c), pose.yaw, pose.stamp)
    if reference != 'corner':
        raise ValueError('未知的定位参考点')
    return pose


def decode_pose(payload, reference='corner'):
    """HTTP/JSON约定：x/y米，yaw弧度，stamp_ns原始Unix纳秒。"""
    if not isinstance(payload, dict):
        raise ValueError('雷达位姿必须为JSON对象')
    for key in ('valid', 'ready', 'ok'):
        if key in payload and payload[key] is not True:
            raise ValueError('雷达定位未就绪')
    data = payload.get('pose', payload)
    if not isinstance(data, dict):
        raise ValueError('雷达没有有效pose')
    for key in ('valid', 'ready', 'ok'):
        if key in data and data[key] is not True:
            raise ValueError('雷达定位未就绪')
    pose = CarPose(float(data['x']), float(data['y']), float(data['yaw']),
                   int(data['stamp_ns']) / 1e9)
    if not all(math.isfinite(v) for v in (pose.x, pose.y, pose.yaw, pose.stamp)) or pose.stamp <= 0:
        raise ValueError('雷达位姿或时间戳无效')
    return convert_reference(pose, reference)


class RadarPoseStream:
    def __init__(self, url='http://127.0.0.1:8765/pose', reference='corner', *, start=True):
        self.url, self.reference = url, reference
        self.samples = deque(maxlen=100)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.error = '等待雷达位姿'
        self.healthy = False
        self.thread = None
        if start:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def accept(self, pose):
        now = time.time()
        if not all(math.isfinite(v) for v in (pose.x, pose.y, pose.yaw, pose.stamp)):
            raise ValueError('雷达位姿不是有限数值')
        age = now - pose.stamp
        if not -.05 <= age <= .6:
            raise ValueError(f'雷达测量与本机时间差={age:+.3f}秒（允许-0.05至0.60秒）；'
                             '请检查两机时钟同步、主机位姿年龄和SSH延迟')
        with self.lock:
            if self.samples and pose.stamp < self.samples[-1].stamp:
                raise ValueError('雷达测量时间倒退')
            if not self.samples or pose.stamp != self.samples[-1].stamp:
                self.samples.append(pose)
            self.healthy = True
            self.error = ''

    def invalidate(self, error):
        with self.lock:
            self.healthy = False
            self.samples.clear()
            self.error = str(error)

    def _run(self):
        while not self.stop.is_set():
            try:
                with urlopen(self.url, timeout=.4) as response:
                    data = json.loads(response.read(1024*1024))
                self.accept(decode_pose(data, self.reference))
            except Exception as error:
                self.invalidate(error)
            self.stop.wait(.05)

    def match(self, stamp=None):
        with self.lock:
            if not self.healthy or not self.samples:
                return None
            pose = self.samples[-1]
            if not -.05 <= time.time() - pose.stamp <= .6:
                return None
            if stamp is None:
                return pose
            matched = closest_pose(list(self.samples), stamp, max_delta=.15)
            return matched if matched is not None and -.05 <= time.time()-matched.stamp <= .6 else None

    def measurements_between(self, start, end):
        """最新定位仍有效时，读取原始测量历史供相机停稳判断。"""
        with self.lock:
            if not self.healthy or not self.samples or not -.05 <= time.time()-self.samples[-1].stamp <= .6:
                return ()
            return tuple(p for p in self.samples if start <= p.stamp <= end)

    def match_frame(self, stamp, timeout=.12):
        # 5Hz位姿的下一次测量可能稍后到达；短暂等待再判断，避免每周期误报断流。
        deadline = time.monotonic() + timeout
        while not self.stop.is_set():
            pose = self.match(stamp)
            if pose is not None:
                return pose
            if time.monotonic() >= deadline:
                break
            self.stop.wait(.01)
        return None

    def wait(self, timeout=3.):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            pose = self.match()
            if pose is not None:
                return pose
            self.stop.wait(.02)
        return None

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=1.)


# 主机只需标准库；HTTP服务无需暴露到局域网，SSH保持原始测量时间。
SSH_READER = r"""
import json, sys, time
from urllib.request import urlopen
while True:
    try:
        with urlopen(sys.argv[1], timeout=.4) as response:
            data = json.loads(response.read(1024*1024))
        print(json.dumps(data), flush=True)
    except Exception as error:
        print(json.dumps({'valid': False, 'ssh_error': str(error)}), flush=True)
    time.sleep(.05)
"""


class SshRadarPoseStream(RadarPoseStream):
    def __init__(self, host, url='http://127.0.0.1:8765/pose', reference='corner'):
        super().__init__(url, reference, start=False)
        command = ['ssh', '-n', '-T', '-o', 'ConnectTimeout=8',
                   '-o', 'ServerAliveInterval=2', '-o', 'ServerAliveCountMax=3',
                   '--', host, 'python3 -u -c ' + shlex.quote(SSH_READER) + ' ' + shlex.quote(url)]
        print(f'通过 SSH 读取主机 {host} 的雷达服务；如需密码，请在终端输入。', flush=True)
        # 后台读取只接收数据，禁止继承界面终端输入，避免SSH抢走操作按键。
        self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                        stdout=subprocess.PIPE, text=True, bufsize=1)
        self.thread = threading.Thread(target=self._read_ssh, daemon=True)
        self.thread.start()

    def _read_ssh(self):
        try:
            for line in self.process.stdout:
                if self.stop.is_set():
                    break
                try:
                    data = json.loads(line)
                    if isinstance(data, dict) and 'ssh_error' in data:
                        raise ValueError(data['ssh_error'])
                    self.accept(decode_pose(data, self.reference))
                except Exception as error:
                    self.invalidate(error)
        except Exception as error:
            self.invalidate(error)
        finally:
            self.invalidate('SSH雷达连接已断开，请检查主机和坐标服务并重新切换模式')

    def close(self):
        self.stop.set()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1.)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        super().close()
        if not self.thread.is_alive():
            self.process.stdout.close()


class RosRadarPoseStream(RadarPoseStream):
    def __init__(self, topic, reference='corner'):
        super().__init__(reference=reference, start=False)
        import rclpy
        self.rclpy = rclpy
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node('task3_radar_pose')
        self.topic = topic
        self.subscription = None
        self.thread = threading.Thread(target=self._run_ros, daemon=True)
        self.thread.start()

    def _message(self, msg):
        try:
            if hasattr(msg, 'data'):
                pose = decode_pose(json.loads(msg.data), self.reference)
            else:
                p = msg.pose.pose if hasattr(msg.pose, 'pose') else msg.pose
                q = p.orientation
                norm = math.sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w)
                if not math.isfinite(norm) or abs(norm-1.) > .01:
                    raise ValueError('无效雷达四元数')
                yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
                pose = convert_reference(CarPose(p.position.x, p.position.y, yaw,
                    msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9), self.reference)
            self.accept(pose)
        except Exception as error:
            self.invalidate(error)

    def _run_ros(self):
        from rclpy.qos import qos_profile_sensor_data
        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import String
        types = {'nav_msgs/msg/Odometry': Odometry, 'geometry_msgs/msg/PoseStamped': PoseStamped,
                 'std_msgs/msg/String': String}
        try:
            while not self.stop.is_set():
                if self.subscription is None:
                    for name, names in self.node.get_topic_names_and_types():
                        if name == self.topic:
                            if len(names) != 1 or names[0] not in types:
                                raise ValueError(f'不支持的雷达消息类型：{names}')
                            self.subscription = self.node.create_subscription(types[names[0]], self.topic,
                                self._message, qos_profile_sensor_data)
                self.rclpy.spin_once(self.node, timeout_sec=.05)
        except Exception as error:
            self.invalidate(error)

    def close(self):
        super().close()
        if self.thread and not self.thread.is_alive():
            self.node.destroy_node()


def add_radar_arguments(parser):
    parser.add_argument('--radar-url', default='http://127.0.0.1:8765/pose', help='雷达HTTP位姿接口')
    parser.add_argument('--topic', default='/vision/robot_pose', help='ROS雷达位姿话题')
    parser.add_argument('--pose-reference', choices=('corner', 'lidar', 'center'), default='corner',
                        help='上游输出参考点；corner已为右前角，lidar雷达原点，center车身几何中心')
    parser.add_argument('--offline', action='store_true', help='明确启用离线手动调试，不读取雷达')


def normalize_source(args):
    args.pose_source = getattr(args, 'pose_source', 'radar')
    if getattr(args, 'offline', False):
        args.pose_source = 'static'
    else:
        # 旧命令保留为设备模式别名；笔记本通过SSH，主机读取本机雷达服务。
        args.pose_source = {'ssh': 'laptop', 'local': 'host'}.get(args.pose_source, args.pose_source)



def open_radar(args):
    reference = getattr(args, 'pose_reference', 'corner')
    if args.pose_source in ('laptop', 'ssh'):
        return SshRadarPoseStream(getattr(args, 'ssh_host', 'volleyball@192.168.3.194'),
                                  getattr(args, 'radar_url', 'http://127.0.0.1:8765/pose'), reference)
    if args.pose_source == 'ros':
        return RosRadarPoseStream(getattr(args, 'topic', '/vision/robot_pose'), reference)
    return RadarPoseStream(getattr(args, 'radar_url', 'http://127.0.0.1:8765/pose'), reference)


def radar_pose_text(pose):
    if pose is None:
        return '雷达等待有效位姿：x=-- m y=-- m yaw=--°；暂停识别和计算'
    sampled_at = time.strftime('%m-%d %H:%M:%S', time.localtime(pose.stamp))
    sampled_at += f'.{int(pose.stamp * 1000) % 1000:03d}'
    return (f'雷达·右前角 x={pose.x:+.3f} m y={pose.y:+.3f} m yaw={display_yaw_deg(pose.yaw):+.1f}° '
            f'采样时刻={sampled_at} 位姿延迟={max(0., time.time()-pose.stamp)*1000:.0f} ms')
