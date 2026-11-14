"""按场地塔位记忆结果，自动长时确认并调度局部复核。"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field, replace
import math
import time

from car_pose import (CarPose, camera_position, observation_moved,
                      OBS_STABLE_FRAMES, OBS_ANGLE_JITTER_DEG)
from field_map import DISPLAY, TOWER_NO, tower_cells
from motion import (Action, MAX_TRIES, GOAL_BEARING, PITCH_OFFSET, heading_words, format_pitch, _aim_cell, _looks,
                    fits_when_aimed, fits_rgb_when_aimed, nearest_observation, DEPTH_FAR_M, _slant, primary_route, _travel, _move_toward)

WINDOW_SECONDS = 2.0
CONFIRM_WINDOWS = 3
MIN_SAMPLES = 6
MIN_AGREEMENT = 0.7
MAX_FRAME_GAP = 1.5
LOOK_SECONDS = 30.0
NO_PROGRESS_SECONDS = 10.0
STATION_RESET_M = 0.10
PITCH_RECHECK_DEG = 4.0


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
    def __init__(self, watch, clock=time.monotonic):
        self.watch = watch
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
        self.last_update = None
        self.message = "等待相机稳定"
        self.blocked_until = 0.0
        self.primary_stops = None
        self.primary_line = []
        self.primary_index = 1

    @property
    def complete(self):
        return all(v.locked is not None and not v.recheck for v in self.watch.votes.values())

    def pause(self):
        self.pitch_checks = {name: [] for name in DISPLAY}
        for evidence in self.evidence.values():
            evidence.clear()
        self.look_started = None
        self.look_key = None
        self.anchor = None
        self.action = None
        self.station_pose = None
        self.heading_centers.clear()
        self.attempts = {name: 0 for name in DISPLAY}

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
        self.heading_centers = {name: math.degrees(pose.yaw) - view.bearing_deg
                                for name, view in self.watch.views.items()}

    def heading_center(self, name):
        if name not in self.heading_centers:
            self.heading_centers[name] = (math.degrees(self.watch.last_pose.yaw)
                                          - self.watch.views[name].bearing_deg)
        return self.heading_centers[name]

    def _aim(self, name):
        watch = self.watch
        center = self.heading_center(name)
        bearing = math.degrees(watch.last_pose.yaw) - center
        view = replace(watch.views[name], bearing_deg=bearing)
        action = _aim_cell(name, view, watch.last_pose, watch.setup.center_to_corner,
                           watch.pitch, True, watch.intr, self.attempts[name])
        if action is not None and action.kind in ('yaw', 'pitch', 'wait'):
            target = math.radians(center + GOAL_BEARING[action.attempt])
            action = replace(action, yaw=math.atan2(math.sin(target), math.cos(target)))
        return action

    def required_checks(self, name):
        cell = next(c for c in tower_cells() if c.name == name)
        distance = _slant(cell,camera_position(self.watch.last_pose,self.watch.setup.center_to_corner,0.))
        return 3 if distance > DEPTH_FAR_M else 2

    def _record(self, name, event, result, now, previous=None):
        watch = self.watch
        pose = watch.last_pose
        self.history[name].append(dict(event=event, result=result, previous=previous,
            elapsed_s=now-self.started, x=pose.x, y=pose.y,
            yaw_deg=math.degrees(pose.yaw), pitch_deg=watch.pitch))

    def _changed_view(self, old):
        watch = self.watch
        pose = watch.last_pose
        turn = math.degrees(math.atan2(math.sin(pose.yaw-old[2]), math.cos(pose.yaw-old[2])))
        return (math.hypot(pose.x-old[0], pose.y-old[1]) >= 0.02
                or abs(turn) >= 5 or abs(watch.pitch-old[3]) >= 3)

    def update(self):
        watch = self.watch
        now = self.clock()
        pose = watch.last_pose
        if self.last_update is not None and now-self.last_update > MAX_FRAME_GAP:
            for evidence in self.evidence.values():
                evidence.clear()
            self.anchor = None
        self.last_update = now
        if self.station_pose is None:
            self.station_pose = pose
            self._freeze_headings(pose)
        if self.action is not None and self.action.kind == 'move':
            if not watch._arrived(self.action) or watch.stable_frames < OBS_STABLE_FRAMES:
                for evidence in self.evidence.values():
                    evidence.clear()
                self.message = '执行换点指令，到位稳定后重新扫描'
                return
            self._reset_station(pose)
        distance = math.hypot(pose.x-self.station_pose.x, pose.y-self.station_pose.y)
        if distance > STATION_RESET_M:
            if watch.stable_frames < OBS_STABLE_FRAMES:
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
        if not watch.have_color or watch.stable_frames < OBS_STABLE_FRAMES:
            for evidence in self.evidence.values():
                evidence.clear()
            self.message = "等待相机稳定"
            # 暂停证据采样，但同一观察档的失败时限仍然有效。
            if self._expire_look(now):
                self._schedule(now)
            return
        for name, view in watch.views.items():
            evidence = self.evidence[name]
            vote = watch.votes[name]
            if not view.in_view or view.needs_yaw or view.needs_pitch:
                evidence.clear()
                continue
            if name in self.conflicts and not self._changed_view(self.conflicts[name]):
                evidence.clear()
                continue
            value = None if view.skip_vote else view.conclusion
            # 远距试读只采信独立顶块；空点/白盒仍用原有近距规划确认。
            cell = next(c for c in tower_cells() if c.name == name)
            far = _slant(cell, camera_position(pose, watch.setup.center_to_corner, 0.)) > DEPTH_FAR_M
            if far and (value not in ('红', '蓝') or view.method != 'RGB顶块'):
                value = None
            result = evidence.see(value, now)
            if result is None:
                continue
            if (vote.locked is None or vote.recheck or result != vote.locked):
                # 新证据改变任务集后，立即作废“无可用点”的旧规划缓存。
                if self.action is not None and self.action.kind == "blocked":
                    self.action = None
                    self.blocked_until = 0.0
            if vote.locked is not None and not vote.recheck and result == vote.locked:
                continue
            if vote.locked is not None and not vote.recheck:
                vote.recheck = True
                self.pitch_checks[name] = []
                self.conflicts[name] = (pose.x, pose.y, pose.yaw, watch.pitch)
                self.conflict_results[name] = dict(previous=vote.locked, detected=result, candidate=result)
                self._record(name, "conflict", result, now, vote.locked)
                # 冲突任务优先复核，停止原任务动作和观察计时。
                self.action = None
                self.look_key = self.look_started = None
            if vote.recheck and name in self.conflict_results:
                self.conflict_results[name]['candidate'] = result
            checks = self.pitch_checks[name]
            # 新姿态的稳定结果不同，作废之前的交叉证据，不能跨冲突凑满两档。
            if checks and checks[0]['result'] != result:
                checks.clear()
            distinct = not any(abs(watch.pitch-item['pitch']) < PITCH_RECHECK_DEG for item in checks)
            if distinct:
                checks.append(dict(result=result, pitch=watch.pitch))
                self._record(name, "pitch_checked", result, now)
            if len(checks) < self.required_checks(name):
                if distinct:
                    self.attempts[name] = min(MAX_TRIES, self.attempts[name] + 1)
                    if self.action is not None and self.action.cell == name:
                        self.action = None
                        self.look_key = self.look_started = None
                continue
            previous = vote.locked
            vote.locked, vote.recheck, vote.ready = result, False, False
            self.conflicts.pop(name, None)
            self.conflict_results.pop(name, None)
            self._record(name, "confirmed" if previous is None else "resolved", result, now, previous)
            self.attempts[name] = 0
        self._schedule(now)

    def _expire_look(self, now):
        if self.look_key is None or self.look_started is None:
            return False
        name, attempt = self.look_key
        elapsed = now - self.look_started
        progress = self.evidence[name].windows
        if elapsed < LOOK_SECONDS and (elapsed < NO_PROGRESS_SECONDS or progress > 0):
            return False
        self.attempts[name] = max(self.attempts[name], attempt + 1)
        self._record(name, "observation_timeout", self.watch.views[name].conclusion, now)
        self.evidence[name].clear()
        self.look_key = self.look_started = None
        self.action = None
        return True

    def _schedule(self, now):
        watch = self.watch
        pose, offset, intr = watch.last_pose, watch.setup.center_to_corner, watch.intr
        self._expire_look(now)
        if self.action is not None and self.action.cells and all(
            watch.votes[n].locked is not None and not watch.votes[n].recheck for n in self.action.cells
        ):
            self.action = None
        if self.action is not None and self.action.kind == "blocked" and now < self.blocked_until:
            return
        if self.action is not None and self.action.kind in ("move", "yaw", "pitch"):
            if not watch._arrived(self.action):
                self.message = "执行观察指令，到位稳定后自动检测"
                return
            self.action = None
        if self.primary_stops is None:
            self.primary_stops, self.primary_line = primary_route(pose,offset,intr)
        while self.primary_index < len(self.primary_stops):
            stop = self.primary_stops[self.primary_index]
            if math.hypot(stop.x-pose.x,stop.y-pose.y) > .08:
                break
            self.primary_index += 1
        pending = [n for n in DISPLAY if watch.votes[n].locked is None or watch.votes[n].recheck]
        if not pending:
            self.action = Action("stop", "全部区域已自动确认")
            self.message = "全部区域已自动确认"
            return
        cells = {cell.name: cell for cell in tower_cells()}
        # 保持正在积累的观察段；其余任务比较下一可执行姿态的实际转角。
        def turn_cost(name):
            if self.attempts[name] >= MAX_TRIES:
                return math.inf
            candidate = self._aim(name)
            if candidate is None:
                return math.inf
            if candidate.yaw is None:
                return 0.0
            return abs(math.degrees(math.atan2(math.sin(candidate.yaw-pose.yaw),
                                               math.cos(candidate.yaw-pose.yaw))))
        active = self.look_key[0] if self.look_key is not None else (
            self.action.cell if self.action is not None and self.action.kind == 'wait' else None)
        pending.sort(key=lambda n: (not watch.votes[n].recheck, n != active, turn_cost(n)))
        for name in pending:
            view = watch.views[name]
            if self.attempts[name] >= MAX_TRIES:
                continue
            if not (fits_when_aimed(cells[name], pose, offset, intr)
                    or fits_rgb_when_aimed(cells[name], pose, offset, intr)
                    or view.conclusion is not None and view.in_view):
                continue
            action = self._aim(name)
            if action is None:
                self.attempts[name] = MAX_TRIES
                continue
            self.attempts[name] = action.attempt
            if action.kind == "wait" and name in self.conflicts and not self._changed_view(self.conflicts[name]):
                self.attempts[name] += 1
                return self._schedule(now)
            if self.look_key is not None and self.look_key != (name, action.attempt):
                self.look_key = self.look_started = None
            self.action = action
            if action.kind == "wait":
                key = (name, self.attempts[name])
                if key != self.look_key:
                    self.look_key, self.look_started = key, now
                progress = self.evidence[name].windows
                self.message = f"{TOWER_NO[name]}号：俯仰复核{len(self.pitch_checks[name])}/{self.required_checks(name)}档；本档稳定{progress}/{CONFIRM_WINDOWS}段（每段至少{WINDOW_SECONDS:g}秒）"
                if view.conclusion is None:
                    reason = view.reason or "尚无通过校验的识别结果"
                    rgb = (view.measurements or {}).get("rgb_cap")
                    if rgb is not None:
                        reason += f"；独立RGB顶块合格候选{rgb['candidates']}个（需唯一候选）"
                    self.message += f"；{reason}；本档已等{now-self.look_started:.0f}秒，无进展{NO_PROGRESS_SECONDS:g}秒换档"
                return
            self.message = f"原地调整{TOWER_NO[name]}号观察角度（{action.attempt+1}/{MAX_TRIES}）"
            return
        # 当前点看够后先走近处主线，沿途点重新扫描远塔；只有主线用尽才去兜底。
        while self.primary_index < len(self.primary_stops):
            stop = self.primary_stops[self.primary_index]
            useful = tuple(n for n in pending if fits_rgb_when_aimed(cells[n],
                CarPose(stop.x,stop.y,pose.yaw),offset,intr))
            if not useful:
                self.primary_index += 1
                continue
            trip = _travel(pose.x,pose.y,stop.x,stop.y,[],pose.yaw)
            if trip is None:
                self.primary_index += 1
                continue
            self.action = _move_toward(pose,stop.x,stop.y,trip[1],'primary',tuple(pending))
            if self.action is not None:
                self.message = f"主路线→{stop.kind}；停稳后复看远处顶块"
                self.look_key = self.look_started = None
                return
            self.primary_index += 1
        # 主线用尽仍未确认的塔，才规划近距兜底。
        references = {}
        origin = camera_position(pose, offset, 0.)
        for name in pending:
            if self.attempts[name] >= MAX_TRIES:
                if not any(math.hypot(pose.x-x, pose.y-y)<0.08 for x,y in self.exhausted[name]):
                    self.exhausted[name].append((pose.x, pose.y))
            if watch.views[name].needs_parallax or self.attempts[name] >= MAX_TRIES:
                references[name] = (watch.views[name].measurements or {}).get("parallax_origin", origin[:2])
        self.action = nearest_observation(pose, offset, intr, pending, self.exhausted, references)
        self.blocked_until = now + 5.0
        self.look_key = self.look_started = None
        self.message = "优先短距离换位复核" if self.action.kind == "move" else "缺少安全有效观察点，任务保留，等待条件变化"

    def lines(self):
        watch = self.watch
        rows = []
        for name in sorted(DISPLAY, key=lambda n: TOWER_NO[n]):
            vote = watch.votes[name]
            state = vote.locked or "未确认"
            status = "冲突待复核，旧结果暂留" if vote.recheck else ("已自动确认" if vote.locked else "检测中")
            rows.append(f"{TOWER_NO[name]}号：{state}；{status}")
        words = {'0': '空塔位', '1': '一层', '2': '两层', '红': '红顶', '蓝': '蓝顶'}
        for name in DISPLAY:
            vote = watch.votes[name]
            if not vote.recheck:
                continue
            detail = self.conflict_results.get(name, {})
            old = words.get(vote.locked, vote.locked or '未知')
            new = words.get(detail.get('detected'), detail.get('detected') or '未知')
            candidate = words.get(detail.get('candidate'), detail.get('candidate') or '待采样')
            rows.append(f"【{TOWER_NO[name]}号冲突】原确认：{old}；新稳定识别：{new}；当前复核：{candidate}")
            checks = len(self.pitch_checks[name])
            progress = self.evidence[name].windows
            rows.append(f"流程改为优先复核：旧结果暂留，不计完成；换姿态后重新采样，俯仰{checks}/{self.required_checks(name)}档、本档{progress}/{CONFIRM_WINDOWS}段")
            action = self.action
            if action is not None and action.kind == 'move':
                rows.append("冲突后续：原地无可用复核姿态，换到新观察点，重新累计两档一致结果")
            elif action is not None and action.kind == 'blocked':
                rows.append("冲突后续：暂无安全有效观察点，保留冲突，等待条件变化后重试")
            else:
                rows.append("冲突后续：先原地换俯仰/朝向；两档一致才写入结果，原地用尽再换位")
        if not any(v.recheck for v in watch.votes.values()):
            resolved = [(name, items[-1]) for name, items in self.history.items()
                        if items and items[-1]['event'] == 'resolved']
            if resolved:
                name, last = max(resolved, key=lambda item: item[1]['elapsed_s'])
                rows.append(f"最近{TOWER_NO[name]}号冲突已解决：{words.get(last['previous'], last['previous'])}→{words.get(last['result'], last['result'])}，两档一致，已写入结果")
        rows.append(self.message)
        action = self.action
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
                    rows.append("车头已对准，停稳后测量（允许误差±3°）")
                else:
                    rows.append(f"现在向{'左' if turn > 0 else '右'}转{abs(turn):.0f}°，车头转到{heading_words(math.degrees(yaw))}")
            if action.kind == "pitch":
                rows.append(f"相机{format_pitch(pitch)}，车身位置不动")
            if action.cell and action.kind != "move":
                center = self.heading_center(action.cell)
                headings = '、'.join(heading_words(center+delta) for delta in dict.fromkeys(GOAL_BEARING))
                rows.append(f"{TOWER_NO[action.cell]}号原地候选朝向：{headings}")
                base_pitch = watch.views[action.cell].pitch_target_deg
                pitches = '、'.join(format_pitch(base_pitch+d) for d in (0.,8.,-8.))
                rows.append(f"每个朝向的相机姿态：{pitches}")
                rows.append(f"当前第{action.attempt+1}/{MAX_TRIES}档；目标：{format_pitch(base_pitch+PITCH_OFFSET[action.attempt])}")
                rows.append(f"俯仰复核{len(self.pitch_checks[action.cell])}/{self.required_checks(action.cell)}档：不同俯仰稳定结果一致才确认；出画档跳过")
            if action.kind == "move" and action.final_x is not None:
                stood = CarPose(action.final_x, action.final_y, pose.yaw)
                rows.append(f"观察落点：x={stood.x:+.3f} m y={stood.y:+.3f} m")
                for look in _looks(stood, watch.setup.center_to_corner, watch.intr, target_names):
                    rows.append(f"{look.number}号：yaw={look.yaw_deg:+.1f}° 俯仰={look.pitch_deg:+.1f}°")
            if target_names:
                rows.append("观察任务：" + "、".join(str(TOWER_NO[n])+"号" for n in target_names))
        if self.complete:
            rows.append("最终结果")
            rows.extend(f"{TOWER_NO[n]} {watch.votes[n].locked}" for n in sorted(DISPLAY, key=lambda n: TOWER_NO[n]))
        return rows

    def route_display(self):
        points = [(s.x,s.y) for s in self.primary_stops or []]
        active = None
        if self.action is not None and self.action.kind == 'move':
            a = self.action
            active = [(self.watch.last_pose.x,self.watch.last_pose.y),(a.x,a.y)]
        return dict(points=points, line=self.primary_line, active=active,
                    index=self.primary_index, complete=self.complete,
                    kinds=[s.kind for s in self.primary_stops or []])

    def snapshot(self):
        return dict(coordinates="场地原点，右前角x/y单位米；action.yaw为弧度，history.yaw_deg为角度；正yaw向左，正pitch抬头",
                    next_action=asdict(self.action) if self.action is not None else None,
                    pending=[n for n,v in self.watch.votes.items() if v.locked is None or v.recheck],
                    complete=self.complete, results={n: v.locked for n,v in self.watch.votes.items()},
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
