"""根据连续有效观测的位置和姿态，判断相机是否已停稳。"""
import math
from collections import deque

SETTLE_SECONDS = .6
MIN_SAMPLES = 3
POSITION_SPAN_M = .025
ANGLE_SPAN_DEG = 1.
MAX_SAMPLE_GAP_SECONDS = .5
WINDOW_SECONDS = 1.
EXIT_POSITION_SPAN_M = .04
EXIT_ANGLE_SPAN_DEG = 1.5


class CameraStability:
    def __init__(self):
        self.reset()

    def reset(self):
        self.started = self.last_time = self.last_stamp = None
        self.anchor_yaw = None
        self.low = self.high = None
        self.samples = 0
        self.settled = False
        self.history=deque()
        self.reason='等待有效采样'

    def summary(self):
        span = ([b-a for a,b in zip(self.low,self.high)] if self.low is not None else [0.]*5)
        return dict(settled=self.settled,samples=self.samples,reason=self.reason,
                    duration_s=round(self.last_time-self.started,3) if self.started is not None else 0.,
                    position_span_m=round(math.hypot(*span[:2]),4),
                    yaw_span_deg=round(span[2],3),pitch_span_deg=round(span[3],3),roll_span_deg=round(span[4],3))

    def update(self, pose, pitch, roll, now, valid=True):
        values = (pose.x,pose.y,pose.yaw,pitch,roll,now,pose.stamp) if pose is not None else ()
        if not valid or not values or not all(v is not None and math.isfinite(v) for v in values):
            self.reset()
            self.reason='位置、姿态或画面无效'
            return False
        if self.last_time is not None and not 0 <= now-self.last_time <= MAX_SAMPLE_GAP_SECONDS:
            self.reset()
        if self.last_stamp is not None and pose.stamp > 0 and pose.stamp < self.last_stamp:
            self.reset()
        if self.anchor_yaw is None:
            self.anchor_yaw = pose.yaw
        yaw = math.degrees(math.atan2(math.sin(pose.yaw-self.anchor_yaw),math.cos(pose.yaw-self.anchor_yaw)))
        point = (pose.x,pose.y,yaw,pitch,roll)
        # 只使用最近一秒的原始采样；重复位姿不能累加样本或停稳时间。
        if pose.stamp > 0 and pose.stamp == self.last_stamp:
            return self.settled
        while self.history and now-self.history[0][0]>WINDOW_SECONDS+1e-9:
            self.history.popleft()
        points=[q[1] for q in self.history]+[point]
        low=tuple(min(p[i] for p in points) for i in range(5))
        high=tuple(max(p[i] for p in points) for i in range(5))
        position_span=math.hypot(high[0]-low[0],high[1]-low[1])
        angle_span=max(high[i]-low[i] for i in (2,3,4))
        position_limit=EXIT_POSITION_SPAN_M if self.settled else POSITION_SPAN_M
        angle_limit=EXIT_ANGLE_SPAN_DEG if self.settled else ANGLE_SPAN_DEG
        if position_span>position_limit+1e-9 or angle_span>angle_limit+1e-9:
            self.reset()
            self.reason=('位置跨度超限' if position_span>position_limit else '朝向/俯仰/横滚跨度超限')
            self.anchor_yaw = pose.yaw
            point = (pose.x,pose.y,0.,pitch,roll)
            low = high = point
        self.low,self.high = low,high
        self.history.append((now,point))
        self.started=self.history[0][0]
        self.last_time,self.last_stamp = now,pose.stamp
        self.samples=len(self.history)
        self.settled = self.samples >= MIN_SAMPLES and now-self.started >= SETTLE_SECONDS-1e-9
        if self.settled:
            self.reason='已停稳'
        elif self.samples<MIN_SAMPLES:
            if self.reason not in ('位置跨度超限','朝向/俯仰/横滚跨度超限'):
                self.reason='有效采样不足3次'
        else:
            self.reason='连续停稳时间不足0.6秒'
        return self.settled
