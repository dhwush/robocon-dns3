"""复算视频RGB，不使用伪造深度；--video 可重新抽取完整时间点。"""
import argparse
import json
import math
from pathlib import Path
import sys

import cv2

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from car_pose import CarPose,OPTICAL_FROM_CORNER,load_intrinsics
from classify import inspect_frame
from field_map import DISPLAY


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path)
    parser.add_argument('--output',type=Path,default=HERE/'after_issue1.json')
    args=parser.parse_args()
    samples=json.loads((HERE/'before_issue1.json').read_text())
    intr=load_intrinsics(str(ROOT/'config/intrinsics.yaml'))
    capture=cv2.VideoCapture(str(args.video)) if args.video else None
    results=[]
    for sample in samples:
        t=sample['time']
        if capture is not None:
            capture.set(cv2.CAP_PROP_POS_MSEC,t*1000)
            ok,frame=capture.read()
            if not ok:
                raise RuntimeError(f'无法抽取 {t}s')
            color=cv2.resize(frame[432:1872,50:2610],(1280,720),interpolation=cv2.INTER_AREA)
        else:
            path=HERE/f'rgb_{t}.png'
            if not path.exists():
                continue
            color=cv2.imread(str(path))
        x,y,yaw,pitch=sample['pose']
        view=inspect_frame(CarPose(x,y,math.radians(yaw)),OPTICAL_FROM_CORNER,pitch,
                           intr,color,None,0)[DISPLAY[2]]
        results.append(dict(time=t,pose=sample['pose'],state=view.conclusion,
            method=view.method,reason=view.reason,
            box=None if view.piece is None else cv2.boundingRect(view.piece),
            seam_rejected=view.measurements.get('rgb_single_seam_rejected',False)))
        print(t,view.conclusion,view.method,flush=True)
    if capture is not None:
        capture.release()
    args.output.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
