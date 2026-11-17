"""录屏RGB回放；位姿来自界面，无原始同步深度。"""
import json
import math
import sys
from pathlib import Path
import cv2
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from classify import inspect_frame
from car_pose import CarPose, OPTICAL_FROM_CORNER, load_intrinsics
from field_map import DISPLAY
from view_field import paint_window

HERE=Path(__file__).resolve().parent

def run():
    intr=load_intrinsics(str(ROOT/'config/intrinsics.yaml'))
    results=[]
    for case in json.loads((HERE/'cases.json').read_text()):
        x,y,yaw,pitch=case['pose']
        color=cv2.imread(str(HERE/f"{case['second']}.png"))
        views=inspect_frame(CarPose(x,y,math.radians(yaw)),OPTICAL_FROM_CORNER,pitch,intr,color,None,0)
        view=views[DISPLAY[case['tower']-1]]
        risk=(view.measurements or {}).get('view_risk')
        results.append(dict(**case,result=view.conclusion,reason=view.reason,risk=risk))
        board=paint_window(color,views,{},[],True,panels=dict(left=[f"录像 {case['second']} 秒 · RGB回放（无原始深度）"],right=[f"重算{case['tower']}号：{view.conclusion or '未知'}", '其余原录屏文字非重算结果'],bottom=[view.reason or '未触发背景限制']))
        cv2.imwrite(str(HERE/f"risk_{case['second']}.jpg"),board)
    (HERE/'risk_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    return results
if __name__=='__main__':
    print(json.dumps(run(),ensure_ascii=False,indent=2))
