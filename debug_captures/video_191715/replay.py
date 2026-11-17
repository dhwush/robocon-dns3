"""19:17录像后半段截图；无原始深度，只核对本次场地策略。"""
import json
import math
import sys
from pathlib import Path
import cv2
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import classify
from car_pose import CarPose,OPTICAL_FROM_CORNER,load_intrinsics
from field_map import DISPLAY,tower_cells
from view_field import paint_window
HERE=Path(__file__).resolve().parent
INTR=load_intrinsics(str(ROOT/'config/intrinsics.yaml'))
POSE=CarPose(2.044,2.858,math.radians(105.7))
PITCH=2.

def run():
    color=cv2.imread(str(HERE/'370.png'))
    # 对比不加本次背景策略的原有判别；不关闭原有可见性/归属校验。
    policy=classify.observation_risk
    try:
        classify.observation_risk=lambda *args:None
        baseline=classify.inspect_frame(POSE,OPTICAL_FROM_CORNER,PITCH,INTR,color,None)
    finally:
        classify.observation_risk=policy
    views=classify.inspect_frame(POSE,OPTICAL_FROM_CORNER,PITCH,INTR,color,None)
    results=[]
    for n in (0,1,2):
        name=DISPLAY[n]
        v=views[name]
        assert not v.measurements.get('view_risk')
        assert (v.conclusion,v.method)==(baseline[name].conclusion,baseline[name].method)
        results.append(dict(tower=n+1,result=v.conclusion,method=v.method,risk=None))
    (HERE/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    # 原录屏叠字仍在；顶部用实底写清本次政策核验结果。
    board=paint_window(color,views,{},[],True,panels=dict(left=[],right=[],bottom=[]))
    from PIL import Image,ImageDraw,ImageFont
    from view_field import _FONT
    panel=Image.fromarray(cv2.cvtColor(board,cv2.COLOR_BGR2RGB))
    draw=ImageDraw.Draw(panel);font=ImageFont.truetype(_FONT,22)
    draw.rectangle((0,0,1280,85),fill=(20,20,20))
    draw.text((12,8),'新规则回放：370秒，y=2.858m < 6号中线3.964m',font=font,fill='white')
    draw.text((12,40),'1/2/3号均不触发背景限制；保留原识别。图中其他文字来自原录屏。',font=font,fill='white')
    cv2.imwrite(str(HERE/'verified_370.jpg'),cv2.cvtColor(__import__('numpy').asarray(panel),cv2.COLOR_RGB2BGR))
    return results
if __name__=='__main__':print(run())
