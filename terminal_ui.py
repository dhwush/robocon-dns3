"""SSH 终端输入；不初始化 OpenCV 的图形后端。"""
import select
import sys


class LiveUI:
    def __init__(self, backend, headless=False, input_stream=None):
        self.backend=backend
        self.headless=headless
        self.input_stream=sys.stdin if input_stream is None else input_stream
        self.saved_terminal=None

    def start_input(self):
        if not self.headless or not self.input_stream.isatty():
            return
        import termios
        import tty
        self.saved_terminal=termios.tcgetattr(self.input_stream.fileno())
        tty.setcbreak(self.input_stream.fileno())
        settings=termios.tcgetattr(self.input_stream.fileno())
        settings[3] &= ~termios.ECHO
        termios.tcsetattr(self.input_stream.fileno(),termios.TCSANOW,settings)

    def close_input(self):
        if self.saved_terminal is not None:
            import termios
            termios.tcsetattr(self.input_stream.fileno(),termios.TCSANOW,self.saved_terminal)
            self.saved_terminal=None

    def waitKey(self, delay):
        if not self.headless:
            return self.backend.waitKey(delay)
        if not select.select([self.input_stream],[],[],max(0,delay)/1000)[0]:
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
        view=(watch.views or {}).get(name)
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
    return f'x={pose.x:+.3f} m  y={pose.y:+.3f} m  yaw={math.degrees(pose.yaw):+.1f}°  俯仰={tilt}'


def terminal_status(watch,pose,pitch):
    """当前反馈、立即执行的安全指令和最终观测姿态分别显示。"""
    from car_pose import CarPose,turned_pose
    from field_map import DISPLAY,TOWER_NO
    survey=watch.autonomy
    action=getattr(survey,'action',None)
    goal=getattr(survey,'observation_goal',None)
    complete=bool(getattr(survey,'complete',False))
    awaiting=bool(getattr(survey,'awaiting_start',False))
    holding=getattr(survey,'stop_until',None) is not None
    if survey is None:
        phase='固定视角识别：保持相机不动'
    elif awaiting:
        phase='待开始：调整车位和相机后按 b 确认起点'
    elif complete:
        phase='识别全部完成，停止！可按 q 退出'
    elif pose is None:
        phase='实时位姿失效：暂停移动和确认，等待定位恢复'
    elif holding:
        phase='已到位，停止！保持车身和相机不动，等待停稳'
    elif action is not None:
        phase={'move':'执行移动','yaw':'调整车头朝向','pitch':'调整相机俯仰',
               'wait':'已到位，保持不动，正在识别','blocked':'暂无安全观察点，停止并保留未确认目标',
               'stop':'停止'}.get(action.kind,action.kind)
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
        elif action.kind=='wait' and goal is not None:
            target=CarPose(goal.observation_x,goal.observation_y,goal.observation_yaw)
            target_pitch=goal.observation_pitch
        command={'move':action.text or '移动至下列安全点',
                 'yaw':'车头转至下列 yaw，位置保持不动',
                 'pitch':'相机调至下列俯仰，车身保持不动',
                 'wait':'停止调整，保持车身和相机不动，等待识别'}.get(action.kind,command)
    elif action is None and not awaiting and not complete and getattr(survey,'initial_pose',None) is not None:
        target=survey.initial_pose
        target_pitch=getattr(survey,'initial_pitch',watch.pitch)
    if awaiting:
        command='按 b 锁定当前 x/y/yaw 和实测俯仰，建立初始路线'
    elif complete or holding:
        command='停止！保持车身和相机不动'
    elif pose is None:
        target=None
        command='停止！等待有效定位，暂不执行下一步'
    elif action is not None and action.kind=='blocked':
        command='停止！'+getattr(survey,'message','等待安全观察点')
    rows += ['下一步命令：'+command,
             '当前要求：'+(_pose_text(target,target_pitch) if target is not None else '暂无移动要求')]
    if goal is not None:
        final=CarPose(goal.observation_x,goal.observation_y,goal.observation_yaw)
        rows.append('最终观测位置（按下一步命令逐步到达）：'+_pose_text(final,goal.observation_pitch))
        rows.append('本站观察：'+'、'.join(TOWER_NO[n]+'号' for n in goal.cells))
    else:
        rows += ['最终观测位置：等待起点确认' if awaiting else '最终观测位置：当前起点' ,'本站观察：仅尚未确认的塔']
    confirmed={n:v.locked for n,v in watch.votes.items() if v.locked is not None and not v.recheck}
    rows.append('已确认：'+('；'.join(TOWER_NO[n]+'号='+str(confirmed[n]) for n in DISPLAY if n in confirmed) or '无'))
    rows.append('待确认：'+('、'.join(TOWER_NO[n]+'号' for n in DISPLAY if n not in confirmed) or '无'))
    rows.append('提示：'+getattr(survey,'message','等待观察任务'))
    rows.append('命令：b 开始 | c 确认当前结果 | 1–5 光线 | n 切塔 | d 诊断 | s 保存 | q 退出')
    if complete:
        rows += ['最终汇总：']+[f'{TOWER_NO[n]}号：{confirmed.get(n,"未确认")}' for n in DISPLAY]
    key=(phase,getattr(action,'kind',None),tuple(confirmed.items()),
         getattr(goal,'observation_x',None),getattr(goal,'observation_y',None))
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

    def show(self,watch,pose,pitch,notice=None):
        rows,key,confirmed=terminal_status(watch,pose,pitch)
        newly=[n for n in confirmed if self.known.get(n)!=confirmed[n]]
        if newly:
            from field_map import TOWER_NO
            self.event='识别好了：'+'；'.join(TOWER_NO[n]+'号='+str(confirmed[n]) for n in newly)+'；下一步请执行上方命令'
        if self.known and not confirmed:  # 新一轮开始。
            self.event=''
        self.known=confirmed
        if notice is not None:
            self.notice=notice
        at=next(i for i,row in enumerate(rows) if row.startswith('命令：'))
        rows[at:at]=['最近完成：'+(self.event or '尚无已确认结果'),
                     '操作反馈：'+(self.notice or '等待命令')]
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
        confirmed=[n for n in DISPLAY if watch.votes[n].locked is not None and not watch.votes[n].recheck]
        self.stream.write('\n最终汇总：'+('全部识别完成' if len(confirmed)==len(DISPLAY) else '本轮已退出，保留未确认目标')+'\n')
        for n in DISPLAY:
            self.stream.write(f'{TOWER_NO[n]}号：{watch.votes[n].locked if n in confirmed else "未确认"}\n')
        survey=getattr(watch,'autonomy',None)
        if survey is not None:
            metrics=survey.metrics()
            self.stream.write(f"观测{metrics['observation_count']}次；累计路程{metrics['distance_m']:.3f}米\n")
        self.stream.flush()
