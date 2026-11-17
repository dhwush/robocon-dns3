"""离线合成RGB-D场地 + 真实识别/自主规划/终端；理想执行器，不连接硬件。"""
import io,json,math,sys
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from car_pose import CarPose,Setup,OPTICAL_FROM_CORNER,load_intrinsics,Intrinsics,camera_frame,turned_pose
from field_map import tower_cells,all_cells,DISPLAY
from report import FieldWatch
from terminal_ui import TerminalDashboard,terminal_status
HERE=Path(__file__).resolve().parent
TRUTH=['1','2','红','蓝','0','1']
base=load_intrinsics(str(ROOT/'config/intrinsics.yaml'))
INTR=Intrinsics(640,360,base.fx/2,base.fy/2,base.cx/2,base.cy/2)

class Clock:
    now=1000.
    def __call__(self):return self.now

def render(pose,pitch):
    origin,right,down,ahead=camera_frame(pose,OPTICAL_FROM_CORNER,pitch)
    yy,xx=np.indices((INTR.height,INTR.width))
    rays=((xx-INTR.cx)/INTR.fx)[...,None]*right+((yy-INTR.cy)/INTR.fy)[...,None]*down+ahead
    color=np.full((INTR.height,INTR.width,3),100,np.uint8)
    depth=np.full(color.shape[:2],8.,np.float32)
    def box(bounds,z0,z1,ink):
        near=np.full(depth.shape,-np.inf);far=np.full(depth.shape,np.inf)
        for axis,lo,hi in [(0,*bounds[:2]),(1,*bounds[2:]),(2,z0,z1)]:
            v=rays[...,axis];moving=np.abs(v)>1e-9
            inv=np.divide(1.,v,out=np.zeros_like(v),where=moving)
            a,b=(lo-origin[axis])*inv,(hi-origin[axis])*inv
            inside=lo<=origin[axis]<=hi
            near=np.maximum(near,np.where(moving,np.minimum(a,b),-np.inf if inside else np.inf))
            far=np.minimum(far,np.where(moving,np.maximum(a,b),np.inf if inside else -np.inf))
        hit=(far>=near)&(near>.3)&(near<depth)
        color[hit]=ink;depth[hit]=near[hit]
    box((-6.,6.,-3.,8.),-.05,0.,(70,100,130))
    for c in all_cells():box(c.bounds(),0,c.height,(45,135,45))
    for c,state in zip(tower_cells(),TRUTH):
        layers=int(state) if state.isdigit() else 2
        b=(c.x-.175,c.x+.175,c.y-.175,c.y+.175)
        for n in range(layers):box(b,c.base_z+n*.35,c.base_z+(n+1)*.35,(220,220,220))
        if state in ('红','蓝'):
            b=(c.x-.1,c.x+.1,c.y-.1,c.y+.1)
            low,high=((220,0,0),(0,0,220)) if state=='红' else ((0,0,220),(220,0,0))
            box(b,c.two_z,c.two_z+.1,low);box(b,c.two_z+.1,c.cap_z,high)
    return color,depth

def run():
    clock=Clock();pose=CarPose(.2,0.,0.,clock.now)
    watch=FieldWatch(Setup(True,OPTICAL_FROM_CORNER,0.,'', 'rad'),INTR)
    watch.last_pose=pose;watch.enable_autonomy(clock)
    survey=watch.autonomy;survey.wait_for_start()
    stream=io.StringIO();ui=TerminalDashboard(stream,interactive=False)
    snapshots=[];last=None;cache={}
    def capture(label):
        ui.show(watch,pose,watch.pitch,notice=label)
        rows,_,_=terminal_status(watch,pose,watch.pitch)
        snapshots.append(dict(time=round(clock.now-1000,2),label=label,rows=rows))
        (HERE/'snapshots.json').write_text(json.dumps(snapshots,ensure_ascii=False,indent=2))
        (HERE/'terminal.txt').write_text('离线模拟：合成RGB-D / 理想执行器；非现场运行\n\n'+stream.getvalue())
    capture('离线模拟：合成RGB-D；理想执行器；未连接雷达、相机或机器人')
    assert survey.confirm_start(pose,0.)
    capture('模拟按 b：确认起点')
    for tick in range(900):
        action=survey.action
        if action and survey.stop_until is None:
            if action.kind=='move':pose=CarPose(action.x,action.y,pose.yaw,clock.now)
            elif action.kind=='yaw':pose=turned_pose(pose,action.yaw)
            elif action.kind=='pitch':watch.follow_pitch(action.pitch)
        clock.now+=.2
        pose=CarPose(pose.x,pose.y,pose.yaw,clock.now)
        key=tuple(round(v,5) for v in (pose.x,pose.y,pose.yaw,watch.pitch))
        if key not in cache:cache[key]=render(pose,watch.pitch)
        color,depth=cache[key]
        watch.update(pose,color,depth)
        action=survey.action
        signature=(getattr(action,'kind',None),tuple(v.locked for v in watch.votes.values()),
            getattr(action,'x',None),getattr(action,'y',None),getattr(action,'yaw',None),getattr(action,'pitch',None),survey.stop_until is not None)
        if signature!=last:
            capture('模拟执行器已按指令到位；停稳门槛由原程序检查')
            last=signature
        if survey.complete:break
        if tick%100==0:print('progress',tick,[v.locked for v in watch.votes.values()],getattr(survey.action,'kind',None),pose,flush=True)
    capture('模拟结束：'+('六塔全部确认' if survey.complete else '仍有未确认目标，未强行填充结果'))
    results=[watch.votes[n].locked for n in DISPLAY]
    (HERE/'terminal.txt').write_text('离线模拟：合成RGB-D / 理想执行器；非现场运行\n\n'+stream.getvalue())
    (HERE/'snapshots.json').write_text(json.dumps(snapshots,ensure_ascii=False,indent=2))
    survey.save(HERE/'survey.json')
    summary=dict(complete=survey.complete,truth=TRUTH,results=results,matching=results==TRUTH,
        simulated_seconds=round(clock.now-1000,2),snapshots=len(snapshots),metrics=survey.metrics())
    (HERE/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    cv2.imwrite(str(HERE/'last_rgb.png'),color)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    assert survey.complete and results==TRUTH,summary
if __name__=='__main__':run()
