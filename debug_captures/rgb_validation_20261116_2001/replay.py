"""重放录屏 RGB：保留原有叠字；无原始深度，roll假定0，不能代替现场RGB-D测试。"""
import importlib.util
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import classify
from car_pose import CarPose,OPTICAL_FROM_CORNER,load_intrinsics
from field_map import DISPLAY

POSES={38:(0,.042,.4,-1),42:(-.063,.068,22.5,-1),45:(-.075,.069,28.8,0),
       46:(-.088,.059,34,-1),155:(-1.857,.353,-8.2,-1)}
INTR=load_intrinsics(str(ROOT/'config/intrinsics.yaml'))
BASELINE='be993d2'


def replay(module,pitch_offsets=(0,)):
    results=[]
    for second,pose in POSES.items():
        color=cv2.imread(str(HERE/f'rgb_{second}.png'))
        x,y,yaw,pitch=pose
        for delta in pitch_offsets:
            view=module.inspect_frame(CarPose(x,y,math.radians(yaw)),OPTICAL_FROM_CORNER,
                    pitch+delta,INTR,color,None,0)[DISPLAY[2]]
            results.append(dict(second=second,pose=pose,pitch_delta=delta,
                result=view.conclusion,method=view.method,reason=view.reason,
                box=None if view.piece is None else cv2.boundingRect(view.piece),
                diagnostics=view.measurements))
            print(second,delta,view.conclusion,view.method,flush=True)
    return results


def main():
    with tempfile.TemporaryDirectory() as directory:
        source=Path(directory)/'baseline_classify.py'
        saved=subprocess.run(['git','show',f'{BASELINE}:classify.py'],cwd=ROOT,
                             check=True,capture_output=True,text=True).stdout
        source.write_text(saved)
        spec=importlib.util.spec_from_file_location('baseline_classify',source)
        baseline=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=baseline
        spec.loader.exec_module(baseline)
        before=replay(baseline)
    after=replay(classify)
    perturbations=replay(classify,(-.5,0,.5))
    result=dict(video='/home/dong/2026-11-16 20-01-51.mkv',baseline_commit=BASELINE,
                code_sha256=hashlib.sha256((ROOT/'classify.py').read_bytes()).hexdigest(),
                limitations=['带检测叠字的录屏RGB','无原始深度','HUD俯仰取整','roll假定0'],
                before=before,after=after,pitch_perturbations=perturbations)
    (HERE/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,
        default=lambda value:value.tolist() if hasattr(value,'tolist') else str(value)))
    panels=[]
    for old,new in zip(before,after):
        second=new['second'];image=cv2.imread(str(HERE/f'rgb_{second}.png'))
        if second==155:roi=(610,215,855,445)
        else:
            x,y,w,h=new['box'];roi=(max(0,x-20),max(0,y-20),min(1280,x+w+20),min(720,y+120))
        a,b,c,d=roi
        row=[]
        for label,item in [('BEFORE',old),('AFTER',new)]:
            panel=image.copy()
            if item['box']:
                x,y,w,h=item['box'];cv2.rectangle(panel,(x,y),(x+w,y+h),(0,255,0),2)
            panel=cv2.resize(panel[b:d,a:c],(400,240))
            cv2.rectangle(panel,(0,0),(400,28),(0,0,0),-1)
            cv2.putText(panel,f'{second}s {label}: {item["result"] or "unknown"}',(8,20),
                        cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
            row.append(panel)
        panels.append(cv2.hconcat(row))
    cv2.imwrite(str(HERE/'comparison.jpg'),cv2.vconcat(panels))


if __name__=='__main__':main()
