import sys,math,json,cv2,numpy as np
from pathlib import Path
ARTIFACTS=Path(__file__).resolve().parent
VIDEO=Path("/home/dong/2026-11-16 13-05-08.mkv")
cap=cv2.VideoCapture(str(VIDEO))
if not cap.isOpened():
 raise RuntimeError(f"无法打开录像: {VIDEO}")
sys.path.insert(0,'/home/dong/robocon/robocon3')
from classify import inspect_frame
from car_pose import CarPose,OPTICAL_FROM_CORNER,load_intrinsics
from field_map import DISPLAY
intr=load_intrinsics('config/intrinsics.yaml');out=[]
for t,vals,numbers in [(29,(-.010,.021,-.5,-1),[3]),(133,(-.248,.191,-46.9,-1),[4,5,6]),(135,(-.243,.195,-46.9,0),[4,5,6])]:
 cap.set(cv2.CAP_PROP_POS_MSEC,t*1000)
 ok,f=cap.read()
 if not ok: raise RuntimeError(f'无法读取{t}秒帧')
 c=cv2.resize(f[188:1628,352:2912],(1280,720),interpolation=cv2.INTER_AREA)
 cv2.imwrite(str(ARTIFACTS/f'rgb_{t}.png'),c)
 x,y,yaw,pitch=vals;vs=inspect_frame(CarPose(x,y,math.radians(yaw)),OPTICAL_FROM_CORNER,pitch,intr,c,None,0)
 r={'time_s':t,'depth':None,'roll_assumed':0,'pose':vals,'targets':{}}
 for number in numbers:
  v=vs[DISPLAY[number-1]];r['targets'][number]={'result':v.conclusion,'reason':v.reason,'method':v.method,'diagnostic':v.measurements}
  print(t,number,v.conclusion,v.reason,v.measurements.get('rgb_white_top'),v.measurements.get('rgb_white_edges'))
 out.append(r)
json.dump(out,open(ARTIFACTS/'results.json','w'),ensure_ascii=False,indent=2,default=lambda a:a.tolist() if hasattr(a,'tolist') else str(a))

# 将本轮真正采用的盒面画为绿色；未知结果不沿用旧候选框。
rows=[]
for record in out:
 color=cv2.imread(str(ARTIFACTS/f"rgb_{record['time_s']}.png"))
 for number,view in record['targets'].items():
  if view['result'] is None:
   cv2.putText(color,f"#{number}: unknown",(900,610),cv2.FONT_HERSHEY_SIMPLEX,.75,(0,165,255),2)
   continue
  piece=view['diagnostic'].get('white_piece')
  if piece is not None:
   piece=np.asarray(piece,np.int32);cv2.polylines(color,[piece],True,(0,255,0),3)
   x,y,w,h=cv2.boundingRect(piece)
   cv2.putText(color,f"#{number}: {view['result']} layer",(x,max(24,y-10)),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,0),2)
 cv2.putText(color,f"RGB replay {record['time_s']}s; depth=None",(20,705),cv2.FONT_HERSHEY_SIMPLEX,.75,(0,255,0),2)
 rows.append(cv2.resize(color,(960,540)))
cv2.imwrite(str(ARTIFACTS/'comparison.jpg'),np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,90])
cap.release()
