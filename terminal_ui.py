"""终端与图形窗口共用按键；无界面模式不初始化图形后端。"""
from car_pose import display_yaw_deg
import select
import sys


class LiveUI:
    def __init__(self, backend, headless=False, input_stream=None):
        self.backend=backend
        self.headless=headless
        self.input_stream=sys.stdin if input_stream is None else input_stream
        self.saved_terminal=None
        self.owned_terminal=None
        self.input_status="终端输入尚未初始化"

    def start_input(self):
        if not self.input_stream.isatty() and self.input_stream is sys.stdin:
            # SSH/启动脚本可能重定向stdin；从当前会话的控制终端接收按键。
            try:
                self.owned_terminal=open('/dev/tty', 'r')
                self.input_stream=self.owned_terminal
            except OSError:
                self.input_status='未连接交互终端：SSH请使用 ssh -tt，启动时不要使用 ssh -n 或后台运行'
                return
        if not self.input_stream.isatty():
            self.input_status='输入为管道：命令需要回车'
            return
        import termios
        import tty
        self.saved_terminal=termios.tcgetattr(self.input_stream.fileno())
        tty.setcbreak(self.input_stream.fileno())
        settings=termios.tcgetattr(self.input_stream.fileno())
        settings[3] &= ~termios.ECHO
        termios.tcsetattr(self.input_stream.fileno(),termios.TCSANOW,settings)
        import os
        self.input_status=f'终端按键已启用（{os.ttyname(self.input_stream.fileno())}）：b/c/d/n/s/1-5/q，无需回车'

    def close_input(self):
        if self.saved_terminal is not None:
            import termios
            termios.tcsetattr(self.input_stream.fileno(),termios.TCSANOW,self.saved_terminal)
            self.saved_terminal=None
        if self.owned_terminal is not None:
            self.owned_terminal.close()
            self.owned_terminal=None

    def waitKey(self, delay):
        if not self.headless:
            # 每轮仍泵送OpenCV事件；图形模式同时读取已启用的真实终端。
            key = self.backend.waitKey(delay)
            if key & 0xFF != 0xFF:
                return key
            if self.saved_terminal is None:
                return -1
        timeout = max(0,delay)/1000 if self.headless else 0
        if not select.select([self.input_stream],[],[],timeout)[0]:
            return -1
        if self.saved_terminal is not None:
            import os
            command=os.read(self.input_stream.fileno(),1).decode(errors="ignore")
            return ord(command.lower()) if command and command not in "\r\n" else -1
        command=self.input_stream.readline()
        if not command:
            return ord('q')
        command=command.strip().lower()
        return ord(command) if len(command)==1 else -1

    def __getattr__(self, name):
        if self.headless and name in ('namedWindow','setMouseCallback','imshow',
                                      'destroyWindow','destroyAllWindows'):
            return lambda *args,**kwargs: None
        return getattr(self.backend,name)


def terminal_results(watch):
    from field_map import DISPLAY
    rows=[]
    for number,name in enumerate(DISPLAY,1):
        view=(getattr(watch,'views',None) or {}).get(name)
        state=None if view is None else view.conclusion
        vote=watch.votes[name]
        confirmed=vote.locked if not vote.recheck else None
        rows.append(f'{number}号：本帧={state if state is not None else "未知"}；'
                    f'已确认={confirmed if confirmed is not None else "未确认"}')
    return rows


def _pose_text(pose,pitch):
    import math
    if pose is None:
        return 'x=-- m  y=-- m  yaw=--°  俯仰=--（等待定位恢复）'
    tilt='--' if pitch is None else f'{pitch:+.1f}°'
    yaw=display_yaw_deg(pose.yaw)
    return f'x={pose.x:+.3f} m  y={pose.y:+.3f} m  yaw={yaw:+.1f}°  俯仰={tilt}'


def terminal_status(watch,pose,pitch):
    """按当前位置与下一位置展示观测目标，保留安全折点指令。"""
    from car_pose import CarPose,turned_pose
    from field_map import DISPLAY,TOWER_NO
    survey=watch.autonomy
    action=getattr(survey,'action',None)
    goal=getattr(survey,'observation_goal',None)
    complete=bool(getattr(survey,'complete',False))
    awaiting=bool(getattr(survey,'awaiting_start',False))
    holding=getattr(survey,'stop_until',None) is not None
    views=getattr(watch,'views',None) or {}
    visible=tuple(n for n,v in views.items()
                  if v.in_view and not v.needs_yaw and not v.needs_pitch)
    initial_idle=(survey is not None and not awaiting and not complete
                  and action is None and goal is None and not holding)
    initial_view_ready=(bool(getattr(watch,'have_color',False))
                        and all(n in views for n in DISPLAY))
    initial_empty=initial_idle and initial_view_ready and not visible
    initial_unknown=initial_idle and not initial_view_ready
    camera_ready=(getattr(watch,'camera_settled',False)
                  and getattr(watch,'confirmation_allowed',False))
    if survey is None:
        phase='固定视角识别：保持相机不动'
    elif pose is None:
        phase='定位无效：停止移动，暂停识别和确认'
    elif awaiting:
        phase='待开始：调整车位和相机后按 b 确认起点'
    elif complete:
        phase=('本轮输出完成（含兜底推定，不能算全部识别成功）' if getattr(survey,'fallback_zero',{})
               else '识别全部完成，停止！可按 q 退出')
    elif holding:
        phase={'move':'本段移动已到位','yaw':'本次旋转已到位','pitch':'本次俯仰调整已到位'}.get(getattr(action,'kind',None),'本步已到位')+'，短暂停顿后继续下一步'
    elif action is not None:
        phase={'move':'执行移动','yaw':'调整车头朝向','pitch':'调整相机俯仰',
               'wait':('观测姿态已到位，保持不动，正在识别' if camera_ready else '观测姿态已到位，保持不动，等待相机停稳及有效数据'),'blocked':'暂无安全观察点，停止并保留未确认目标',
               'stop':'停止'}.get(action.kind,action.kind)
    elif initial_empty:
        phase='起点无可观测塔，正在规划下一位置'
    elif initial_unknown:
        phase='起点已确认，等待有效视野数据判断观察目标'
    else:
        phase='起点已确认，保持不动，等待停稳识别'
    rows=['任务三 · 实时状态（坐标：场地原点 / 车身右前角；yaw：车头朝向）',f'状态：{phase}',
          '当前位置：'+_pose_text(pose,pitch)]
    command='等待当前点识别完成，程序自动给出下一步'
    target=None
    target_pitch=pitch
    if not awaiting and not complete and action is not None and action.kind in ('move','yaw','pitch','wait'):
        # move.x/y是安全折点；完整观测目标另列，禁止直接穿障前往最终落点。
        basis=pose or watch.last_pose
        target=CarPose(action.x if action.x is not None else basis.x,
                       action.y if action.y is not None else basis.y,
                       action.yaw if action.yaw is not None else basis.yaw)
        target_pitch=action.pitch if action.pitch is not None else watch.pitch
        if action.kind=='yaw':
            target=turned_pose(basis,action.yaw)
            target_pitch=pitch  # 本步只转向；未来俯仰仅在最终姿态预览中显示。
        elif action.kind=='pitch':
            target=CarPose(basis.x,basis.y,basis.yaw)
        elif action.kind=='wait' and goal is not None:
            target=CarPose(goal.observation_x,goal.observation_y,goal.observation_yaw)
            target_pitch=goal.observation_pitch
        command={'move':'移动到本步要求的 x/y；按本步 yaw 保持车头朝向',
                 'yaw':'绕车身中心原地转至本步 yaw；右前角 x/y 会随转动变化',
                 'pitch':'相机调至本步要求的俯仰，车身保持不动',
                 'wait':'停止调整，保持车身和相机不动，等待识别'}.get(action.kind,command)
    elif action is None and not awaiting and not complete and getattr(survey,'initial_pose',None) is not None:
        target=survey.initial_pose
        target_pitch=getattr(survey,'initial_pitch',watch.pitch)
    if pose is None:
        target=None
        command='停止！等待定位恢复，暂不执行目标'
    elif awaiting:
        command='按 b 锁定当前 x/y/yaw 和实测俯仰，建立初始路线'
    elif complete:
        command='停止！保持车身和相机不动'
    elif holding:
        command='结束本次调整，短暂停顿；程序随后给出下一步，尚未开始识别'
    elif action is not None and action.kind=='blocked':
        command='停止！'+getattr(survey,'message','等待安全观察点')
    elif initial_empty or initial_unknown:
        target=None
        command=('起点无可观测塔，跳过起点识别；正在规划下一位置，请勿自行移动'
                 if initial_empty else '保持不动，等待有效视野数据；暂不开始识别')
    angle_rows=[]
    angle_key=None
    if target is not None and pose is not None and not awaiting and not complete and not holding:
        import math
        from motion import YAW_TOL_DEG,PITCH_TOL_DEG
        yaw_error=math.degrees(math.atan2(math.sin(target.yaw-pose.yaw),math.cos(target.yaw-pose.yaw)))
        yaw_ok=abs(yaw_error)<=YAW_TOL_DEG+1e-9
        pitch_error=None if pitch is None or target_pitch is None else target_pitch-pitch
        pitch_ok=pitch_error is not None and abs(pitch_error)<=PITCH_TOL_DEG+1e-9
        angle_rows.append('旋转：'+('可以了，保持朝向（误差不超过3°）' if yaw_ok else f'还需调整 {yaw_error:+.1f}°（目标减实测）'))
        angle_rows.append('俯仰：'+('实测无效，等待反馈' if pitch_error is None else
                          '可以了，保持俯仰（误差不超过3°）' if pitch_ok else
                          f'还需调整 {pitch_error:+.1f}°（正数抬头，负数低头）'))
        angle_key=(yaw_ok,pitch_ok,pitch_error is None)
        if not holding and action is not None:
            if action.kind=='yaw' and yaw_ok:
                command='旋转可以了，停止转动，等待停稳后进入下一步'
            elif action.kind=='pitch' and pitch_ok:
                command='俯仰可以了，停止调整，等待停稳后进入下一步'
    inferred=getattr(survey,'fallback_zero',{})
    def observation_text(names):
        return '、'.join(TOWER_NO[n]+'号（'+('已完成' if watch.votes[n].locked is not None
            and not watch.votes[n].recheck and n not in inferred else '未完成')+'）'
            for n in DISPLAY if n in names) or '无'
    if pose is None:
        current_observation='定位无效，暂停观察'
    elif awaiting:
        current_observation='待按 b 确认起点；当前视野：'+observation_text(visible)
    elif initial_unknown:
        current_observation='待有效视野数据确定'
    elif complete:
        current_observation='本轮已结束'
    elif action is not None and action.kind in ('move','yaw','pitch','blocked'):
        current_observation='无（尚未到达观测姿态）'
    else:
        current_names=tuple(getattr(goal,'cells',())) or tuple(getattr(action,'cells',())) or visible
        current_observation=observation_text(current_names)
    rows.append('当前位置观察目标：'+current_observation)
    rows.append('下一位置：'+(_pose_text(target,target_pitch) if target is not None else
                            '暂无移动要求' if pose is None else
                            '等待起点确认' if awaiting else
                            '正在规划' if initial_empty else
                            '待有效视野数据后确定' if initial_unknown else
                            '本轮已结束，无后续观测位置' if complete else '暂无移动要求'))
    next_names=tuple(getattr(goal,'cells',())) or tuple(getattr(action,'cells',()))
    next_observation=('等待起点确认后规划' if awaiting else
                      '规划后给出' if initial_empty else '待有效视野数据后确定' if initial_unknown else '无' if complete or target is None else
                      '当前点继续观察，无需移动' if action is None or action.kind=='wait' else
                      observation_text(next_names))
    if target is not None and action is not None and action.kind in ('move','yaw','pitch') and not awaiting and not complete:
        next_observation+='（完成后续姿态调整后观察）'
    rows += ['下一位置观察目标：'+next_observation,'本步操作：'+command]
    rows.extend(angle_rows)
    if target is not None and pose is not None and not awaiting and not complete and not holding:
        import math
        turn=math.degrees(math.atan2(math.sin(target.yaw-pose.yaw),math.cos(target.yaw-pose.yaw)))
        rows.append(f'距本步目标：Δx={target.x-pose.x:+.3f} m  Δy={target.y-pose.y:+.3f} m  Δyaw={turn:+.1f}°')
        rows.append('差值说明：目标减实测；x/y 沿场地坐标轴，yaw 按最短角度')
    if goal is not None and target is not None and not awaiting and not complete:
        final=CarPose(goal.observation_x,goal.observation_y,goal.observation_yaw)
        # 下一位置可能是避障折点，不能把最终观测点当作直接移动指令。
        if action is not None and action.kind in ('move','yaw','pitch'):
            rows.append('下一位置用途：安全过渡，停稳后按后续指令到达观测姿态')
            rows.append('该目标最终观测姿态（预览，不可直接前往）：'+_pose_text(final,goal.observation_pitch))
    inferred=getattr(survey,'fallback_zero',{})
    confirmed={n:v.locked for n,v in watch.votes.items() if v.locked is not None and not v.recheck and n not in inferred}
    rows.append('已确认：'+('；'.join(TOWER_NO[n]+'号='+str(confirmed[n]) for n in DISPLAY if n in confirmed) or '无'))
    inferred=getattr(survey,'fallback_zero',{})
    if inferred:
        rows.append('兜底推定0层（非识别确认）：'+'、'.join(TOWER_NO[n]+'号' for n in DISPLAY if n in inferred))
    rows.append('待确认：'+('、'.join(TOWER_NO[n]+'号' for n in DISPLAY if n not in confirmed) or '无'))
    if awaiting or (action is not None and action.kind=='blocked'):
        rows.append(('开始条件：' if awaiting else '停止原因：')+getattr(survey,'message','等待观察任务'))
    gate=getattr(watch,'camera_stability',None)
    if gate is not None:
        state=gate.summary()
        rows.append(f"停稳检查：{state['samples']}/3次；{state['duration_s']:.1f}/0.6秒；"
                    +getattr(watch,'stability_reason',state['reason']))
    if survey is not None:
        current_result=getattr(survey,'observation_status','尚无已确认结果')
        if pose is None:
            current_result='定位无效，暂停确认'
        elif awaiting:
            current_result='尚未开始，按 b 确认起点'
        elif complete:
            current_result='本轮输出结束；详见最终汇总'
        elif holding:
            current_result='本步调整完成，等待进入下一步，尚未开始识别'
        elif action is not None and action.kind=='blocked':
            current_result='无安全观察指令，暂停识别并保留未确认目标'
        elif action is not None and action.kind in ('move','yaw','pitch'):
            current_result='本点尚未识别：请先完成上方本步操作'
        elif initial_empty:
            current_result='起点无可观测塔，跳过识别，规划后续观察点'
        elif initial_unknown:
            current_result='等待有效视野数据，尚未开始识别'
        elif holding:
            current_result='本步调整完成，等待进入下一步，尚未开始识别'
        elif action is not None and action.kind=='wait':
            current_result=('正在识别；未列入已确认的目标均未成功确认' if camera_ready
                            else '等待相机停稳及有效数据，尚未开始识别')
        rows.append('当前观测结果：'+current_result)
        rows.append('最近观测结论：'+getattr(survey,'last_observation_outcome','尚无完成的观测'))
    # 只展示本步目标的阻止原因，避免六个本帧候选与当前命令混在一起。
    names = tuple(getattr(action,'cells',())) or tuple(getattr(goal,'cells',()))
    if survey is not None and hasattr(survey,'confirmation_reason'):
        for name in names:
            reason=survey.confirmation_reason(name)
            if reason:
                rows.append(f'{TOWER_NO[name]}号确认状态：{reason}')
    rows.append('命令：b 开始 | c 确认当前结果 | 1–5 光线 | n 切塔 | d 诊断 | s 保存 | q 退出')
    if complete:
        rows += ['最终汇总：']+[f'{TOWER_NO[n]}号：'+
            ('兜底推定0层（非识别确认）' if n in inferred else str(confirmed.get(n,'未确认'))) for n in DISPLAY]
    key=(phase,angle_key,getattr(action,'kind',None),
         tuple(getattr(action,k,None) for k in ('x','y','yaw','pitch')),tuple(confirmed.items()),
         getattr(survey,'observation_status',None),
         getattr(goal,'observation_x',None),getattr(goal,'observation_y',None),
         getattr(watch,'confirmation_allowed',None),
         tuple((n,getattr(v,'conclusion',None)) for n,v in (getattr(watch,'views',None) or {}).items()))
    return rows,key,confirmed


class TerminalDashboard:
    def __init__(self,stream=None,interactive=None):
        self.stream=sys.stdout if stream is None else stream
        self.interactive=self.stream.isatty() if interactive is None else interactive
        self.active=False
        self.last_key=None
        self.known={}
        self.event=''
        self.notice=''
        self.input_status=''
        self.completed_here=None

    def show(self,watch,pose,pitch,notice=None):
        rows,key,confirmed=terminal_status(watch,pose,pitch)
        newly=[n for n in confirmed if self.known.get(n)!=confirmed[n]]
        if newly:
            self.completed_here=(pose,pitch,tuple(newly))
            from field_map import TOWER_NO
            self.event='识别好了：'+'；'.join(TOWER_NO[n]+'号='+str(confirmed[n]) for n in newly)+'；请查看本步操作'
        if self.known and not confirmed:  # 新一轮开始。
            self.event=''
            self.completed_here=None
        if getattr(watch.autonomy,'awaiting_start',False):
            self.completed_here=None
        if self.completed_here is not None and pose is not None:
            import math
            from field_map import TOWER_NO
            observed,tilt,names=self.completed_here
            same=(observed is not None and math.hypot(pose.x-observed.x,pose.y-observed.y)<=.025
                  and abs(math.degrees(math.atan2(math.sin(pose.yaw-observed.yaw),math.cos(pose.yaw-observed.yaw))))<=3
                  and pitch is not None and tilt is not None and abs(pitch-tilt)<=3)
            if same:
                index=next(i for i,row in enumerate(rows) if row.startswith('当前位置观察目标：'))
                rows[index]='当前位置观察目标：'+'、'.join(TOWER_NO[n]+'号（已完成）' for n in names if n in confirmed)
            else:
                self.completed_here=None
        self.known=confirmed
        if notice is not None:
            self.notice=notice
        elif self.last_key is not None and key!=self.last_key[0]:
            self.notice=''  # 任务变化后不继续显示上一阶段的操作反馈。
        at=next(i for i,row in enumerate(rows) if row.startswith('命令：'))
        if self.input_status:
            rows.insert(at, '输入通道：'+self.input_status)
            at += 1
        rows[at:at]=['最近完成：'+(self.event or '尚无已确认结果'),
                     '操作反馈：'+(self.notice or '按本步操作执行；无需额外确认')]
        key=(key,self.event,self.notice)
        if self.interactive:
            if not self.active:
                self.stream.write('\x1b[?1049h\x1b[?25l')
                self.active=True
            self.stream.write('\x1b[H\x1b[J'+'\n'.join(rows)+'\n')
            self.stream.flush()
        elif key!=self.last_key:
            # 重定向到文件时仅写状态变化，避免坐标抖动生成大量日志。
            self.stream.write('\n'.join(rows)+'\n')
            self.stream.flush()
        self.last_key=key

    def finish(self,watch):
        from field_map import DISPLAY,TOWER_NO
        if self.active:
            self.stream.write('\x1b[?25h\x1b[?1049l')
            self.active=False
        inferred=getattr(getattr(watch,'autonomy',None),'fallback_zero',{})
        confirmed=[n for n in DISPLAY if watch.votes[n].locked is not None and not watch.votes[n].recheck and n not in inferred]
        self.stream.write('\n最终汇总：'+('全部识别完成' if len(confirmed)==len(DISPLAY) else '本轮已退出，保留未确认目标')+'\n')
        for n in DISPLAY:
            result='兜底推定0层（非识别确认）' if n in inferred else watch.votes[n].locked if n in confirmed else '未确认'
            self.stream.write(f'{TOWER_NO[n]}号：{result}\n')
        survey=getattr(watch,'autonomy',None)
        if survey is not None:
            metrics=survey.metrics()
            self.stream.write(f"观测{metrics['observation_count']}次；累计路程{metrics['distance_m']:.3f}米\n")
        self.stream.flush()
