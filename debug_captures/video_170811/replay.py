"""真实录屏RGB复算；受控深度是几何合成，不是录像原始深度。"""
import json
import math
from pathlib import Path
import sys
import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from classify import inspect_frame
from car_pose import CarPose,OPTICAL_FROM_CORNER,load_intrinsics,camera_frame
from field_map import tower_cells,DISPLAY

CASES=json.loads((HERE/'cases.json').read_text())
INTR=load_intrinsics(str(ROOT/'config/intrinsics.yaml'))

def scene(case,mode):
    color=cv2.imread(str(HERE/f"{case['second']}.png"))
    x,y,yaw,pitch=case['pose']
    pose=CarPose(x,y,math.radians(yaw))
    if mode=='rgb':
        return pose,pitch,color,None
    depth=np.full(color.shape[:2],{'missing':np.nan,'background':8.,'foreground':.5}.get(mode,8.),np.float32)
    if mode=='owned':
        # 只在真实白盒的大致图像区域提供合成塔体交点，背景保留8m。
        cell=tower_cells()[case['tower']-1]
        origin,right,down,ahead=camera_frame(pose,OPTICAL_FROM_CORNER,pitch)
        yy,xx=np.indices(depth.shape)
        rays=((xx-INTR.cx)/INTR.fx)[...,None]*right+((yy-INTR.cy)/INTR.fy)[...,None]*down+ahead
        level=cell.one_z if case['tower']==2 else cell.two_z
        near=np.full(depth.shape,-np.inf);far=np.full(depth.shape,np.inf)
        for axis,lo,hi in [(0,cell.x-.23,cell.x+.23),(1,cell.y-.23,cell.y+.23),(2,cell.base_z,level)]:
            direction=rays[...,axis]
            inv=np.divide(1.,direction,out=np.zeros_like(direction),where=np.abs(direction)>1e-9)
            a,b=(lo-origin[axis])*inv,(hi-origin[axis])*inv
            near=np.maximum(near,np.minimum(a,b));far=np.minimum(far,np.maximum(a,b))
        roi=(xx>=200)&(xx<=710)&(yy>=560) if case['tower']==2 else (xx>=440)&(xx<=580)&(yy>=430)&(yy<=640)
        hit=(far>=near)&(near>.3)&roi
        depth[hit]=near[hit]
    return pose,pitch,color,depth

def run():
    records=[]
    for case in CASES:
        modes=['rgb','missing','background','foreground']
        if case['second'] in (60,163): modes.append('owned')
        for mode in modes:
            pose,pitch,color,depth=scene(case,mode)
            view=inspect_frame(pose,OPTICAL_FROM_CORNER,pitch,INTR,color,depth,0)[DISPLAY[case['tower']-1]]
            record=dict(second=case['second'],tower=case['tower'],mode=mode,result=view.conclusion,
                method=view.method,reason=view.reason,box=None if view.piece is None else cv2.boundingRect(view.piece),diagnostic=view.measurements)
            records.append(record)
            print(case['second'],mode,view.conclusion,view.method,view.reason,flush=True)
    (HERE/'results.json').write_text(json.dumps(records,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n')
    return records

if __name__=='__main__':run()
