"""根据连续有效观测的位置和姿态，判断相机是否已停稳。"""
import math

SETTLE_SECONDS = .6
MIN_SAMPLES = 3
POSITION_SPAN_M = .01
ANGLE_SPAN_DEG = 1.
MAX_SAMPLE_GAP_SECONDS = .5


class CameraStability:
    def __init__(self):
        self.reset()

    def reset(self):
        self.started = self.last_time = self.last_stamp = None
        self.anchor_yaw = None
        self.low = self.high = None
        self.samples = 0
        self.settled = False

    def summary(self):
        span = ([b-a for a,b in zip(self.low,self.high)] if self.low is not None else [0.]*5)
        return dict(settled=self.settled,samples=self.samples,
                    duration_s=round(self.last_time-self.started,3) if self.started is not None else 0.,
                    position_span_m=round(math.hypot(*span[:2]),4),
                    yaw_span_deg=round(span[2],3),pitch_span_deg=round(span[3],3),roll_span_deg=round(span[4],3))

    def update(self, pose, pitch, roll, now, valid=True):
        values = (pose.x,pose.y,pose.yaw,pitch,roll,now,pose.stamp) if pose is not None else ()
        if not valid or not values or not all(v is not None and math.isfinite(v) for v in values):
            self.reset()
            return False
        if self.last_time is not None and not 0 <= now-self.last_time <= MAX_SAMPLE_GAP_SECONDS:
            self.reset()
        if self.last_stamp is not None and pose.stamp > 0 and pose.stamp < self.last_stamp:
            self.reset()
        if self.anchor_yaw is None:
            self.anchor_yaw = pose.yaw
        yaw = math.degrees(math.atan2(math.sin(pose.yaw-self.anchor_yaw),math.cos(pose.yaw-self.anchor_yaw)))
        point = (pose.x,pose.y,yaw,pitch,roll)
        low = point if self.low is None else tuple(min(a,b) for a,b in zip(self.low,point))
        high = point if self.high is None else tuple(max(a,b) for a,b in zip(self.high,point))
        if (math.hypot(high[0]-low[0],high[1]-low[1]) > POSITION_SPAN_M+1e-9
                or any(high[i]-low[i] > ANGLE_SPAN_DEG+1e-9 for i in (2,3,4))):
            self.reset()
            self.anchor_yaw = pose.yaw
            point = (pose.x,pose.y,0.,pitch,roll)
            low = high = point
        self.low,self.high = low,high
        # 同一个雷达测量重复配画面不能累加停稳证据。
        if pose.stamp > 0 and pose.stamp == self.last_stamp:
            return self.settled
        if self.started is None:
            self.started = now
        self.last_time,self.last_stamp = now,pose.stamp
        self.samples += 1
        self.settled = self.samples >= MIN_SAMPLES and now-self.started >= SETTLE_SECONDS-1e-9
        return self.settled
