"""两段录屏RGB复算；合成深度只用于验证分支，不代表现场RGB-D重放。"""
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import types

import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
import classify
from car_pose import CarPose,OPTICAL_FROM_CORNER,load_intrinsics,to_camera
from field_map import tower_cells,DISPLAY


def baseline():
    module=types.ModuleType('classify_89e0b6b')
    sys.modules[module.__name__]=module
    source=subprocess.check_output(['git','show','89e0b6b:classify.py'],cwd=ROOT,text=True)
    exec(compile(source,'classify_89e0b6b.py','exec'),module.__dict__)
    return module


def run():
    intr=load_intrinsics(str(ROOT/'config/intrinsics.yaml'))
    old=baseline()
    cell=tower_cells()[2]
    results=[]
    for case in json.loads((HERE/'cases.json').read_text()):
        color=cv2.imread(str(HERE/(case['id']+'.png')))
        x,y,yaw,pitch=case['pose']
        pose=CarPose(x,y,math.radians(yaw))
        record=dict(case=case,variants={})
        for label,module in [('before',old),('after',classify)]:
            variants={}
            for mode in ['rgb','missing','background','foreground']:
                depth=None if mode=='rgb' else np.full(color.shape[:2],
                    {'missing':np.nan,'background':8.,'foreground':2.}[mode],np.float32)
                info={}
                result=module._rgb_white_top(cell,color,module._white_mask(color),
                    module._mask(cell,pose,OPTICAL_FROM_CORNER,pitch,intr),pose,
                    OPTICAL_FROM_CORNER,pitch,intr,
                    float(to_camera(cell.center(cell.one_z),pose,OPTICAL_FROM_CORNER,pitch)[2]),
                    diagnostic=info,depth=depth)
                view=module.inspect_frame(pose,OPTICAL_FROM_CORNER,pitch,intr,color,depth,0)[DISPLAY[2]]
                variants[mode]=dict(result=view.conclusion,method=view.method,reason=view.reason,
                    box=None if view.piece is None else cv2.boundingRect(view.piece),
                    local_top=None if result is None else result[0],
                    foreground=view.measurements.get('foreground_check'),
                    diagnostic=info)
                print(case['id'],label,mode,view.conclusion,variants[mode]['local_top'],flush=True)
            record['variants'][label]=variants
        results.append(record)
    (HERE/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,
        default=lambda a:a.tolist() if hasattr(a,'tolist') else str(a))+'\n')


if __name__=='__main__':
    run()
