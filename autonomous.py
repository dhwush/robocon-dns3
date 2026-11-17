"""信任有效识别结果，仅为尚未看清的塔调度观察姿态与安全换位。"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field, replace
import math
import time

from car_pose import (CarPose, camera_position, observation_moved, turned_pose, body_center,
                      OBS_STABLE_FRAMES, OBS_ANGLE_JITTER_DEG)
from field_map import DISPLAY, TOWER_NO, tower_cells
from motion import (Action, MAX_TRIES, GOAL_BEARING, PITCH_OFFSET, heading_words, format_pitch, _aim_cell, _looks,
                    fits_when_aimed, fits_rgb_when_aimed, nearest_observation, _sees,
                    YAW_TOL_DEG, PITCH_TOL_DEG, MOVE_ARRIVE_M, _travel, _bypass_points, _move_toward, survey as supplementary_route, _observation_preferences, _parking_clear, _right_front_turn_clear)

WINDOW_SECONDS = 2.0
CONFIRM_WINDOWS = 3
MIN_SAMPLES = 6
MIN_AGREEMENT = 0.7
MAX_FRAME_GAP = 1.5
LOOK_SECONDS = 30.0
NO_PROGRESS_SECONDS = 10.0
POINT_LOOK_SECONDS = 2.0
STATION_RESET_M = 0.10
PITCH_RECHECK_DEG = 4.0
STOP_HOLD_SECONDS = 1.0


@dataclass
class Evidence:
    start: float | None = None
    last: float | None = None
    samples: list[str | None] = field(default_factory=list)
    candidate: str | None = None
    windows: int = 0

    def clear(self):
        self.start = self.last = None
        self.samples.clear()
        self.candidate = None
        self.windows = 0

    def see(self, value, now):
        if self.last is not None and now - self.last > MAX_FRAME_GAP:
            self.clear()
        if self.start is None:
            self.start = now
        self.last = now
        self.samples.append(value)
        if now - self.start < WINDOW_SECONDS or len(self.samples) < MIN_SAMPLES:
            return None
        counts = Counter(item for item in self.samples if item is not None)
        best, count = counts.most_common(1)[0] if counts else (None, 0)
        good = count >= MIN_AGREEMENT * len(self.samples)
        if good:
            self.windows = self.windows + 1 if best == self.candidate else 1
            self.candidate = best
        else:
            self.windows = 0
            self.candidate = None
        self.samples.clear()
        self.start = now
        if self.windows >= CONFIRM_WINDOWS:
            result = self.candidate
            self.clear()
            return result
        return None


class AutonomousSurvey:
    def __init__(self, watch, clock=time.monotonic, *, stationary_scan=False):
        self.watch = watch
        self.stationary_scan = stationary_scan
        self.initial_pose = watch.last_pose
        self.initial_pitch = watch.pitch
        self.awaiting_start = False
        self.confirmed_start = False
        self.clock = clock
        self.started = clock()
        self.evidence = {name: Evidence() for name in DISPLAY}
        self.history = {name: [] for name in DISPLAY}
        self.conflicts = {}
        self.conflict_results = {}
        self.pitch_checks = {name: [] for name in DISPLAY}
        self.exhausted = {name: [] for name in DISPLAY}
        self.attempts = {name: 0 for name in DISPLAY}
        self.action = None
        self.look_started = None
        self.look_key = None
        self.anchor = None
        self.anchor_pitch = None
        self.anchor_roll = None
        self.station_pose = None
        self.heading_centers = {}
        self.scan_targets = {}
        self.sampling_suspended_at = None
        self.last_update = None
        self.message = "等待相机稳定"
        self.blocked_until = 0.0
        self.primary_stops = None
        self.primary_line = []
        self.primary_index = 1
        self.initial_inflight = False
        self.additional_stops = []
        self.reserve_stops = []
        self.reserve_index = 0
        self.manual_confirmed = set()
        self.observations = []
        self.observation_goal = None
        self.failed_observations = []
        self.distance_m = 0.0
        self.distance_pose = watch.last_pose
        self.stop_until = None
        self.stop_message = ""

    def wait_for_start(self):
        """模式2/3先浏览实时姿态，按B才建立本轮起点。"""
        self.__init__(self.watch, self.clock, stationary_scan=self.stationary_scan)
        self.awaiting_start = True
        self.initial_pose = None
        self.action = self.observation_goal = None
        self.message = '移动、旋转至起始姿态；按 B 确认当前 x/y/yaw 和实测俯仰开始'

    def confirm_start(self, pose, pitch):
        if not self.awaiting_start:
            return False
        if (pose is None or pitch is None or
                not all(math.isfinite(v) for v in (pose.x,pose.y,pose.yaw,pitch))):
            self.message = '等待有效雷达位姿和相机实测俯仰，暂不能确认起点；就绪后按 B'
            return False
        if not _parking_clear(pose.x,pose.y,pose.yaw):
            self.message = '当前位置不满足整车行驶边界，请移至可达安全区域后按 B'
            return False
        from vote import Vote
        watch = self.watch
        watch.last_pose, watch.pitch = pose, float(pitch)
        watch.votes = {name: Vote() for name in DISPLAY}
        watch.stable_frames = 0
        watch.stability_after_stamp = pose.stamp
        if hasattr(watch,'camera_stability'):
            watch.camera_stability.reset()
            watch.camera_settled = False
        for attr in ('local_rgb_tracks', 'parallax_origins'):
            if hasattr(watch, attr):
                getattr(watch, attr).clear()
        self.__init__(watch, self.clock, stationary_scan=self.stationary_scan)
        self.confirmed_start = True
        self.message = (f'起点已确认：x={pose.x:+.3f} m y={pose.y:+.3f} m '
                        f'yaw={math.degrees(pose.yaw):+.1f}° 俯仰={pitch:+.1f}°；停稳后开始观测规划')
        return True

    def _camera_ready(self):
        return (self.watch.stable_frames >= OBS_STABLE_FRAMES
                and getattr(self.watch,'camera_settled',True))

    def confirmation_target(self):
        action = self.action
        if action is None or action.kind != 'wait' or action.cell in self.manual_confirmed:
            return None
        view = self.watch.views[action.cell]
        if not self.watch.have_color or not self._camera_ready() or not view.in_view:
            return None
        # 只确认画面当前显示的结论，不把旧锁定值当成本帧结果。
        if view.conclusion is None or view.skip_vote or view.needs_yaw or view.needs_pitch:
            return None
        return action.cell, view.conclusion

    def confirm_current(self, target):
        if target is None or target != self.confirmation_target():
            return False
        name, result = target
        vote = self.watch.votes[name]
        previous = vote.locked
        vote.locked, vote.recheck, vote.ready = result, False, False
        self.manual_confirmed.add(name)
        self.conflicts.pop(name, None)
        self.conflict_results.pop(name, None)
        self.evidence[name].clear()
        self._record(name, 'manual_confirmed', result, self.clock(), previous)
        self.look_key = self.look_started = None
        self.action = None
        self._schedule(self.clock())
        return True

    def metrics(self):
        return dict(observation_unit="settled_pose", distance_m=round(self.distance_m, 3),
                    observation_count=len(self.observations),
                    observations=self.observations,
                    manual_confirmed=[n for n in DISPLAY if n in self.manual_confirmed])

    @property
    def complete(self):
        return not self.awaiting_start and all(v.locked is not None for v in self.watch.votes.values())

    def pause(self):
        self.initial_inflight = False
        self.sampling_suspended_at = None
        self.pitch_checks = {name: [] for name in DISPLAY}
        for evidence in self.evidence.values():
            evidence.clear()
        self.look_started = None
        self.look_key = None
        self.anchor = None
        self.action = None
        self.observation_goal = None
        self.station_pose = None
        self.stop_until = None
        self.stop_message = ""
        self.heading_centers.clear()
        self.scan_targets.clear()
        self.attempts = {name: 0 for name in DISPLAY}

    def suspend_sampling(self):
        """位姿暂缺时丢弃本档稳定段，保留观察点、动作和扫描进度。"""
        if self.sampling_suspended_at is None:
            self.sampling_suspended_at = self.clock()
        for evidence in self.evidence.values():
            evidence.clear()
        self.anchor = None
        self.message = '停止！等待有效位姿，暂停采样并保留扫描进度'

    def _reset_station(self, pose):
        self.station_pose = pose
        self._freeze_headings(pose)
        self.pitch_checks = {name: [] for name in DISPLAY}
        self.attempts = {name: 0 for name in DISPLAY}
        self.look_key = self.look_started = None
        self.action = None
        for evidence in self.evidence.values():
            evidence.clear()

    def _freeze_headings(self, pose):
        self.scan_targets.clear()
        self.heading_centers = {name: math.degrees(pose.yaw) - view.bearing_deg
                                for name, view in self.watch.views.items()}

    def heading_center(self, name):
        if name not in self.heading_centers:
            self.heading_centers[name] = (math.degrees(self.watch.last_pose.yaw)
                                          - self.watch.views[name].bearing_deg)
        return self.heading_centers[name]

    def _targets(self, name):
        if name in self.scan_targets:
            return self.scan_targets[name]
        watch = self.watch
        pose, view = watch.last_pose, watch.views[name]
        center = self.heading_center(name)
        yaw_now = math.degrees(pose.yaw)
        wrap = lambda value: (value + 180.) % 360. - 180.
        # 当前视野足够就先保持；其余候选仍保留作未知时的替代视角。
        options = list(zip(GOAL_BEARING, (view.pitch_target_deg+d for d in PITCH_OFFSET)))
        if view.in_view and not view.needs_yaw and not view.needs_pitch:
            options.append((wrap(yaw_now-center), watch.pitch))
        options = list(dict.fromkeys(options))
        pending_cells = [c for c in tower_cells() if watch.votes[c.name].locked is None]
        coverage_at = {target: sum(_sees(c, turned_pose(pose, math.radians(center+target[0])),
                                        watch.setup.center_to_corner, target[1], watch.intr)
                                   for c in pending_cells) for target in options}
        ordered = []
        yaw, pitch = yaw_now, watch.pitch
        while options:
            def cost(target):
                goal, target_pitch = target
                turn = abs(wrap(center+goal-yaw))
                tilt = abs(target_pitch-pitch)
                return -coverage_at[target], turn+tilt
            target = min(options, key=cost)
            options.remove(target)
            ordered.append(target)
            yaw, pitch = center+target[0], target[1]
        self.scan_targets[name] = tuple(ordered)
        return self.scan_targets[name]

    def _aim(self, name):
        watch = self.watch
        center = self.heading_center(name)
        bearing = math.degrees(watch.last_pose.yaw) - center
        view = replace(watch.views[name], bearing_deg=bearing)
        targets = self._targets(name)
        action = _aim_cell(name, view, watch.last_pose, watch.setup.center_to_corner,
                           watch.pitch, True, watch.intr, self.attempts[name], targets)
        if action is not None and action.kind in ('yaw', 'pitch', 'wait'):
            goal, pitch = targets[action.attempt]
            target = math.radians(center + goal)
            action = replace(action, yaw=math.atan2(math.sin(target), math.cos(target)), pitch=pitch)
        if action is not None and action.kind in ('yaw', 'pitch', 'wait'):
            faced = turned_pose(watch.last_pose, action.yaw)
            if any(name in old['cells'] and math.hypot(faced.x-old['x'], faced.y-old['y']) < .08
                   and abs((math.degrees(faced.yaw)-old['yaw_deg']+180)%360-180) <= YAW_TOL_DEG
                   and abs(action.pitch-old['pitch_deg']) <= PITCH_TOL_DEG for old in self.failed_observations):
                self.attempts[name] = action.attempt+1
                return self._aim(name) if self.attempts[name] < MAX_TRIES else None
        return action

    def _note_observation(self, names, now, attempt=0):
        """同一停稳位置、yaw与俯仰只计一次，多塔共享该次观测。"""
        watch = self.watch
        if not watch.have_color or not self._camera_ready():
            return
        pose = watch.last_pose
        previous = self.observations[-1] if self.observations else None
        same = previous is not None and math.hypot(pose.x-previous['x'], pose.y-previous['y']) < .03
        same = same and abs((math.degrees(pose.yaw)-previous['yaw_deg']+180)%360-180) <= YAW_TOL_DEG
        same = same and abs(watch.pitch-previous['pitch_deg']) <= PITCH_TOL_DEG
        if same:
            previous['cells'] = list(dict.fromkeys(previous.get('cells', [previous['cell']])+list(names)))
            if previous['cell'] is None and names:
                previous['cell'] = names[0]
        else:
            self.observations.append(dict(cell=names[0] if names else None, cells=list(names), attempt=attempt,
                x=pose.x, y=pose.y, yaw_deg=math.degrees(pose.yaw),
                pitch_deg=watch.pitch, elapsed_s=now-self.started))

    def _advance_observation(self):
        """移动、绕障、转向和俯仰都是到同一次观测姿态的准备步骤。"""
        goal = self.observation_goal
        if goal is None:
            return False
        watch = self.watch
        names = tuple(n for n in goal.cells if watch.votes[n].locked is None)
        if not names:
            self.observation_goal = None
            return False
        pose = watch.last_pose
        target = CarPose(goal.observation_x, goal.observation_y, goal.observation_yaw)
        if not _parking_clear(target.x,target.y,target.yaw):
            self.observation_goal = None
            self.action = Action('blocked','原观察点不满足当前边界，取消该点并保留未识别目标')
            return True
        move_target = turned_pose(target, pose.yaw)
        if math.dist(body_center(pose), body_center(target)) > MOVE_ARRIVE_M:
            trip = _travel(pose.x, pose.y, move_target.x, move_target.y, _bypass_points(pose.yaw), pose.yaw)
            if trip is None:
                self.observation_goal = None
                self.action = Action('blocked', '联合观测姿态暂无安全路线')
                return True
            self.action = _move_toward(pose, move_target.x, move_target.y, trip[1], 'joint', names)
        elif abs((math.degrees(target.yaw-pose.yaw)+180)%360-180) > YAW_TOL_DEG:
            if not _right_front_turn_clear(pose,target.yaw):
                self.observation_goal = None
                self.action = Action('blocked','转向过程中右前点会越过左右边界，保留未识别目标')
                return True
            self.action = Action('yaw', '转到联合观测朝向', yaw=target.yaw,
                                 pitch=goal.observation_pitch, cell=names[0], cells=names)
        elif abs(goal.observation_pitch-watch.pitch) > PITCH_TOL_DEG:
            self.action = Action('pitch', '调到联合观测俯仰', yaw=target.yaw,
                                 pitch=goal.observation_pitch, cell=names[0], cells=names)
        else:
            self.action = Action('wait', '联合观测', yaw=target.yaw, pitch=goal.observation_pitch,
                                 cell=names[0], cells=names)
        self.message = '到达联合观测姿态后，一次查看全部可见的未识别塔'
        return True

    def _record(self, name, event, result, now, previous=None):
        watch = self.watch
        pose = watch.last_pose
        self.history[name].append(dict(event=event, result=result, previous=previous,
            elapsed_s=now-self.started, x=pose.x, y=pose.y,
            yaw_deg=math.degrees(pose.yaw), pitch_deg=watch.pitch))

    def _stop_at_arrival(self, now):
        """先发停止并留出反应时间，停稳后才能采样或下发下一动作。"""
        action = self.action
        if action is None or action.kind not in ('move', 'yaw', 'pitch'):
            return False
        if self.stop_until is None:
            if not self.watch._arrived(action):
                return False
            words = {'move': '本段移动到位', 'yaw': '完成旋转', 'pitch': '俯仰到位'}
            self.stop_message = words[action.kind] + '，可以了，停止！保持车身和相机不动，等待稳定'
            self.stop_until = now + STOP_HOLD_SECONDS
        if now < self.stop_until or not self._camera_ready():
            for evidence in self.evidence.values():
                evidence.clear()
            self.message = self.stop_message
            return True
        self.stop_until = None
        self.stop_message = ''
        if not self.watch._arrived(action):
            return False
        if action.kind == 'move':
            self._reset_station(self.watch.last_pose)
        else:
            self.action = None
        return False

    def update(self):
        if self.awaiting_start:
            return
        watch = self.watch
        now = self.clock()
        if self.confirmed_start and self.primary_stops is None:
            # 先生成路线预览；相机停稳只限制识别确认，不阻止规划。
            self.primary_stops,self.primary_line = supplementary_route(
                self.initial_pose,watch.setup.center_to_corner,watch.intr,self.initial_pitch,
                max_stops=3,planner=nearest_observation)
        if not self.stationary_scan and not self._camera_ready():
            self.look_key = self.look_started = None
        pose = watch.last_pose
        if self.initial_pose is None:
            self.initial_pose = pose
        if self.distance_pose is not None:
            self.distance_m += math.hypot(pose.x-self.distance_pose.x, pose.y-self.distance_pose.y)
        self.distance_pose = pose
        if self.sampling_suspended_at is not None:
            if self.look_started is not None:
                self.look_started += now - self.sampling_suspended_at
            self.sampling_suspended_at = None
            self.last_update = None
        if self.last_update is not None and now-self.last_update > MAX_FRAME_GAP:
            for evidence in self.evidence.values():
                evidence.clear()
            self.anchor = None
        self.last_update = now
        if self._stop_at_arrival(now):
            return
        if self.station_pose is None:
            self.station_pose = pose
            self._freeze_headings(pose)
        if self.action is not None and self.action.kind == 'move':
            if not watch._arrived(self.action) or not self._camera_ready():
                for evidence in self.evidence.values():
                    evidence.clear()
                self.message = '执行换点指令，到位稳定后重新扫描'
                return
            self._reset_station(pose)
        distance = math.hypot(pose.x-self.station_pose.x, pose.y-self.station_pose.y)
        if distance > STATION_RESET_M:
            if not self._camera_ready():
                for evidence in self.evidence.values():
                    evidence.clear()
                self.message = '位置变化超过10厘米，停稳后重新扫描'
                return
            self._reset_station(pose)
        moved = (self.anchor is None or observation_moved(self.anchor, pose)
                 or abs(watch.pitch-self.anchor_pitch) > OBS_ANGLE_JITTER_DEG
                 or abs(watch.roll-self.anchor_roll) > OBS_ANGLE_JITTER_DEG)
        if moved:
            self.anchor, self.anchor_pitch, self.anchor_roll = pose, watch.pitch, watch.roll
            for evidence in self.evidence.values():
                evidence.clear()
        if not watch.have_color or not self._camera_ready():
            for evidence in self.evidence.values():
                evidence.clear()
            self.message = "等待相机停稳：连续0.6秒、至少3次有效采样；位置变化≤1厘米，朝向/俯仰/横滚变化≤1°"
            if not self.stationary_scan:
                self.look_key = self.look_started = None
            elif self._expire_look(now):
                self._schedule(now)
            return
        if self.observation_goal is not None:
            self._advance_observation()
            if self.action is not None and self.action.kind != 'wait':
                return
        if not self.stationary_scan and not self.observations and not self.complete:
            visible = [n for n,v in watch.views.items() if v.in_view and not v.needs_yaw and not v.needs_pitch]
            self._note_observation(visible, now)
        preferences = (_observation_preferences(pose,watch.setup.center_to_corner,watch.pitch,watch.intr)
                       if not self.stationary_scan else None)
        for name, view in watch.views.items():
            evidence = self.evidence[name]
            vote = watch.votes[name]
            # 已有结果直接视为准确；后续帧不触发冲突或重复观察。
            if vote.locked is not None:
                vote.recheck = False
                evidence.clear()
                continue
            if (not view.in_view or view.needs_yaw or view.needs_pitch
                    or view.skip_vote or view.conclusion is None):
                evidence.clear()
                continue
            if preferences is not None and not preferences[name][0]:
                goal = self.observation_goal
                if goal is None or not goal.observation_fallback or name not in goal.cells:
                    continue
            result = view.conclusion
            vote.locked, vote.recheck, vote.ready = result, False, False
            self.conflicts.pop(name, None)
            self.conflict_results.pop(name, None)
            evidence.clear()
            self._record(name, "confirmed", result, now)
            self._note_observation([name], now, self.attempts[name])
            if self.action is not None and (self.action.cell == name
                    or self.action.kind == "blocked"):
                self.action = None
                self.look_key = self.look_started = None
                self.blocked_until = 0.0
            self.attempts[name] = 0
        self._schedule(now)

    def _expire_look(self, now):
        if self.look_key is None or self.look_started is None:
            return False
        name, attempt = self.look_key
        elapsed = now - self.look_started
        progress = self.evidence[name].windows
        if not self.stationary_scan:
            if elapsed < POINT_LOOK_SECONDS:
                return False
        elif elapsed < LOOK_SECONDS and (elapsed < NO_PROGRESS_SECONDS or progress > 0):
            return False
        if self.observation_goal is not None:
            goal = self.observation_goal
            self.failed_observations.append(dict(cells=list(goal.cells), x=goal.observation_x,
                y=goal.observation_y, yaw_deg=math.degrees(goal.observation_yaw),
                pitch_deg=goal.observation_pitch))
            self.observation_goal = None
        view = self.watch.views[name]
        # 已停稳观察仍缺前后分离证据，原地换21档不能产生视差；转入换位流程。
        if view.conclusion is None and (view.needs_parallax
                or (view.measurements or {}).get('needs_standoff')):
            self.attempts[name] = MAX_TRIES
        else:
            self.attempts[name] = max(self.attempts[name], attempt + 1)
        self._record(name, "observation_timeout", self.watch.views[name].conclusion, now)
        self.evidence[name].clear()
        self.look_key = self.look_started = None
        self.action = None
        return True

    def _schedule(self, now):
        if self.awaiting_start:
            return
        watch = self.watch
        pose, offset, intr = watch.last_pose, watch.setup.center_to_corner, watch.intr
        if self.initial_pose is None:
            self.initial_pose = pose
        self._expire_look(now)
        if self.action is not None and self.action.cells and all(
            watch.votes[n].locked is not None for n in self.action.cells
        ):
            self.action = None
        if self.action is not None and self.action.kind == "blocked" and now < self.blocked_until:
            return
        if self.action is not None and self.action.kind in ("move", "yaw", "pitch"):
            if not watch._arrived(self.action):
                self.message = "执行观察指令，到位稳定后自动检测"
                return
            self.action = None
        # 不生成必经复看主线，任务集只包含尚无识别结果的塔。
        if self.stationary_scan:
            self.primary_stops, self.primary_line = [], []
        self.reserve_stops = []
        pending = [n for n in DISPLAY if watch.votes[n].locked is None]
        if not pending:
            self.observation_goal = None
            self.look_key = self.look_started = None
            self.action = Action("stop", "停止！全部区域已自动确认")
            self.message = "停止！全部区域已自动确认，保持车身和相机不动"
            return
        if self._advance_observation():
            if self.action is not None and self.action.kind == 'wait':
                key = (self.action.cell, self.action.attempt)
                if self.look_key != key:
                    self.look_key, self.look_started = key, now
                self._note_observation(self.action.cells, now)
            return
        if not self.stationary_scan:
            # 默认只提供新的补充点，一点一组yaw/俯仰，不再逐档原地扫描。
            origin = camera_position(pose, offset, 0.)
            references = {n: (watch.views[n].measurements or {}).get('parallax_origin', origin[:2])
                          for n in pending if watch.views[n].needs_parallax}
            excluded = [body_center(self.initial_pose)]
            excluded += [body_center(CarPose(o['x'],o['y'],math.radians(o['yaw_deg'])))
                         for o in self.observations]
            if self.initial_inflight:
                # 上一初始点已完成、超时或不可达，才消费这一站。
                self.primary_index += 1
                self.initial_inflight = False
            if self.primary_stops is None:
                self.primary_stops, self.primary_line = supplementary_route(
                    self.initial_pose if self.confirmed_start else pose, offset, intr,
                    self.initial_pitch if self.confirmed_start else watch.pitch, pending=pending,
                    references=references, failed=self.failed_observations,
                    excluded_centers=excluded, planner=nearest_observation, max_stops=3)
            stop = None
            while self.primary_index < len(self.primary_stops):
                candidate = self.primary_stops[self.primary_index]
                if any(TOWER_NO[n] in {look.number for look in candidate.looks} for n in pending):
                    stop = candidate
                    self.initial_inflight = True
                    break
                self.primary_index += 1
            if stop is None:
                # 初始路线走完仍未识别，才追加一站；追加站严格使用中距离分离视角。
                plan, _ = supplementary_route(
                    pose, offset, intr, watch.pitch, pending=pending,
                    references=references, failed=self.failed_observations,
                    excluded_centers=excluded, planner=nearest_observation,
                    max_stops=2, medium_only=True)
                if len(plan) > 1:
                    stop = plan[1]
                    self.additional_stops.append(stop)
            goal = Action('blocked','暂无安全中距离补充点')
            if stop is not None:
                look = stop.looks[0]
                numbers = {item.number for item in stop.looks}
                names = tuple(n for n in pending if TOWER_NO[n] in numbers)
                goal = Action('wait','补充观测',cells=names,observation_x=stop.x,observation_y=stop.y,
                              observation_yaw=math.radians(look.yaw_deg),observation_pitch=look.pitch_deg,
                              observation_fallback=stop.fallback)
            self.look_key = self.look_started = None
            if goal.observation_yaw is not None:
                self.observation_goal = goal
                self._advance_observation()
            else:
                self.action = goal
                self.blocked_until = now+5.
                self.message = '暂无能看到剩余塔的安全中距离补充点，保留未观测目标'
            return
        cells = {cell.name: cell for cell in tower_cells()}
        eligible = {n: (fits_when_aimed(cells[n], pose, offset, intr)
                        or fits_rgb_when_aimed(cells[n], pose, offset, intr)
                        or watch.views[n].conclusion is not None and watch.views[n].in_view)
                    for n in pending}
        pending_names = tuple(pending)
        # 一次完整观测以共同覆盖为先；位置、yaw和俯仰不按轴分别计费。
        def adjustment_cost(name):
            if self.attempts[name] >= MAX_TRIES or not eligible[name]:
                return (math.inf, math.inf, 0)
            candidate = self._aim(name)
            if candidate is None:
                return (math.inf, math.inf, 0)
            yaw = candidate.yaw if candidate.yaw is not None else pose.yaw
            pitch = candidate.pitch if candidate.pitch is not None else watch.pitch
            turn = abs(math.degrees(math.atan2(math.sin(yaw-pose.yaw), math.cos(yaw-pose.yaw))))
            tilt = abs(pitch-watch.pitch)
            faced = turned_pose(pose, yaw)
            coverage = sum(_sees(cells[n], faced, offset, pitch, intr) for n in pending_names)
            return -coverage, turn+tilt, 0
        active = self.look_key[0] if self.look_key is not None else (
            self.action.cell if self.action is not None and self.action.kind == 'wait' else None)
        pending.sort(key=lambda n: (n != active, adjustment_cost(n)))
        references = {n: (watch.views[n].measurements or {}).get('parallax_origin',
                           camera_position(pose, offset, 0.)[:2])
                      for n in pending if watch.views[n].needs_parallax}
        joint = nearest_observation(pose, offset, intr, pending, self.exhausted,
                                    references, watch.pitch, self.failed_observations)
        if joint.observation_yaw is not None:
            # 搜索已同时包含当前位置和移动候选，直接执行最佳完整姿态。
            self.observation_goal = joint
            self.look_key = self.look_started = None
            self._advance_observation()
            if self.action.kind == 'wait':
                self.look_key, self.look_started = (self.action.cell, 0), now
                self._note_observation(self.action.cells, now)
            return
        for name in pending:
            view = watch.views[name]
            if self.attempts[name] >= MAX_TRIES:
                continue
            if not eligible[name]:
                continue
            action = self._aim(name)
            if action is None:
                self.attempts[name] = MAX_TRIES
                continue
            self.attempts[name] = action.attempt
            if self.look_key is not None and self.look_key != (name, action.attempt):
                self.look_key = self.look_started = None
            self.action = action
            if action.kind == "wait":
                key = (name, self.attempts[name])
                if key != self.look_key:
                    self.look_key, self.look_started = key, now
                    self._note_observation([name], now, action.attempt)
                self.message = f"{TOWER_NO[name]}号：等待当前视角的有效识别结果"
                if view.conclusion is None:
                    reason = view.reason or "尚无通过校验的识别结果"
                    rgb = (view.measurements or {}).get("rgb_cap")
                    if rgb is not None:
                        reason += f"；独立RGB顶块合格候选{rgb['candidates']}个（需唯一候选）"
                    self.message += f"；{reason}；本档已等{now-self.look_started:.0f}秒，无进展{NO_PROGRESS_SECONDS:g}秒换档"
                return
            self.message = f"调整{TOWER_NO[name]}号未看清目标的观察角度（{action.attempt+1}/{MAX_TRIES}）"
            return
        # 只有尚未看清的塔参与换位，不访问固定复看点或备用点。
        references = {}
        origin = camera_position(pose, offset, 0.)
        for name in pending:
            if self.attempts[name] >= MAX_TRIES:
                if not any(math.hypot(pose.x-x, pose.y-y)<0.08 for x,y in self.exhausted[name]):
                    self.exhausted[name].append((pose.x, pose.y))
            if watch.views[name].needs_parallax:
                references[name] = (watch.views[name].measurements or {}).get("parallax_origin", origin[:2])
        self.action = joint
        if joint.observation_yaw is not None:
            self.observation_goal = joint
            self.look_key = self.look_started = None
            self._advance_observation()
        self.blocked_until = now + 5.0
        self.look_key = self.look_started = None
        self.message = "换到安全观察点，只查看尚未看清的塔" if self.action.kind == "move" else "缺少安全有效观察点，任务保留，等待条件变化"

    def lines(self):
        watch = self.watch
        rows = []
        for name in sorted(DISPLAY, key=lambda n: TOWER_NO[n]):
            vote = watch.votes[name]
            state = vote.locked or "未确认"
            status = "已识别" if vote.locked is not None else "尚未看清"
            rows.append(f"{TOWER_NO[name]}号：{state}；{status}")
        rows.append(self.message)
        action = self.action
        if self.stop_until is not None:
            return rows
        if action is not None and action.kind == "wait":
            rows.append("停止！保持车身和相机不动，正在观察")
        elif action is not None and action.kind == "blocked":
            rows.append("停止！暂无安全有效观察点，等待后续指令")
        if action is not None and action.kind in ("move", "yaw", "pitch", "wait"):
            pose = watch.last_pose
            x = action.x if action.x is not None else pose.x
            y = action.y if action.y is not None else pose.y
            yaw = action.yaw if action.yaw is not None else pose.yaw
            pitch = action.pitch if action.pitch is not None else watch.pitch
            target_names = action.cells or ((action.cell,) if action.cell else ())
            # 移动阶段先给出下一安全折点，再给出最终观察姿态。
            rows.append(f"下一步（场地原点/右前角）：x={x:+.3f} m y={y:+.3f} m yaw={math.degrees(yaw):+.1f}° 俯仰={pitch:+.1f}°")
            if action.kind == "yaw":
                turn = math.degrees(math.atan2(math.sin(yaw-pose.yaw), math.cos(yaw-pose.yaw)))
                if abs(turn) <= 3.0:
                    rows.append("可以了，停止！车头已对准（允许误差±3°）")
                else:
                    rows.append(f"现在向{'左' if turn > 0 else '右'}转{abs(turn):.0f}°，车头转到{heading_words(math.degrees(yaw))}")
            if action.kind == "pitch":
                rows.append(f"相机{format_pitch(pitch)}，车身位置不动")
            if action.cell and action.kind != "move" and self.observation_goal is None:
                center = self.heading_center(action.cell)
                targets = self._targets(action.cell)
                headings = '、'.join(heading_words(center+goal) for goal in dict.fromkeys(t[0] for t in targets))
                rows.append(f"{TOWER_NO[action.cell]}号原地候选朝向：{headings}")
                pitches = '、'.join(format_pitch(pitch) for pitch in dict.fromkeys(t[1] for t in targets))
                rows.append(f"候选相机俯仰（按需调整）：{pitches}")
                rows.append(f"当前第{action.attempt+1}/{len(targets)}个候选；目标：{format_pitch(targets[action.attempt][1])}")
                rows.append("获得有效识别结果即记住；仅未看清时继续调整视角")
            if action.kind == "move" and action.final_x is not None:
                stood = CarPose(action.final_x, action.final_y, pose.yaw)
                rows.append(f"观察落点：x={stood.x:+.3f} m y={stood.y:+.3f} m")
                if self.observation_goal is None:
                    for look in _looks(stood, watch.setup.center_to_corner, watch.intr, target_names):
                        rows.append(f"{look.number}号：yaw={look.yaw_deg:+.1f}° 俯仰={look.pitch_deg:+.1f}°")
            if self.observation_goal is not None:
                goal = self.observation_goal
                rows.append(f"本次完整观测姿态：x={goal.observation_x:+.3f} m y={goal.observation_y:+.3f} m "
                            f"yaw={math.degrees(goal.observation_yaw):+.1f}° 俯仰={goal.observation_pitch:+.1f}°；全部到位后计一次观测")
            if target_names:
                rows.append("观察任务：" + "、".join(str(TOWER_NO[n])+"号" for n in target_names))
        if self.complete:
            rows.append("最终结果")
            rows.extend(f"{TOWER_NO[n]} {watch.votes[n].locked}" for n in sorted(DISPLAY, key=lambda n: TOWER_NO[n]))
        return rows

    def route_display(self):
        if self.awaiting_start:
            return dict(points=[],line=[],active=None,index=0,complete=False,kinds=[],
                        reserve_points=[],reserve_index=0)
        initial = self.initial_pose or self.watch.last_pose
        points = [(initial.x,initial.y)] if initial is not None else []
        kinds = ['初始点']
        if self.primary_stops:
            points = [(s.x,s.y) for s in self.primary_stops]
            kinds = ['初始点'] + ['补充点']*(len(points)-1)
            points.extend((s.x,s.y) for s in self.additional_stops)
            kinds.extend('追加点' for _ in self.additional_stops)
        elif self.observation_goal is not None:
            goal = self.observation_goal
            points.append((goal.observation_x,goal.observation_y))
            kinds.append('补充点')
        active = None
        if self.action is not None and self.action.kind == 'move':
            a = self.action
            active = [(self.watch.last_pose.x,self.watch.last_pose.y),(a.x,a.y)]
        return dict(points=points, line=self.primary_line, active=active,
                    index=min(self.primary_index+1,len(points)) if self.initial_inflight else len(points), complete=self.complete, kinds=kinds,
                    reserve_points=[], reserve_index=0)

    def snapshot(self):
        return dict(coordinates="场地原点，右前角x/y单位米；action.yaw为弧度，history.yaw_deg为角度；正yaw向左，正pitch抬头",
                    awaiting_start=self.awaiting_start,
                    camera_stability=(self.watch.camera_stability.summary()
                                      if hasattr(self.watch,"camera_stability") else None),
                    camera_ready=self._camera_ready(),
                    initial_pose=asdict(self.initial_pose) if self.initial_pose is not None else None,
                    initial_pitch=self.initial_pitch if self.confirmed_start else None,
                    next_action=asdict(self.action) if self.action is not None else None,
                    pending=[n for n,v in self.watch.votes.items() if v.locked is None],
                    complete=self.complete, results={n: v.locked for n,v in self.watch.votes.items()},
                    route_plan=dict(primary=[asdict(s) for s in self.primary_stops or []],
                                    primary_index=self.primary_index,
                                    additional=[asdict(s) for s in self.additional_stops],
                                    reserve=[asdict(s) for s in self.reserve_stops],
                                    reserve_index=self.reserve_index),
                    observation_goal=asdict(self.observation_goal) if self.observation_goal else None,
                    metrics=self.metrics(),
                    pitch_checks=self.pitch_checks,
                    conflicts=[n for n,v in self.watch.votes.items() if v.recheck],
                    conflict_results=self.conflict_results, history=self.history,
                    instructions=self.lines())

    def save(self, path):
        import json
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
