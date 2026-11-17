"""录屏RGB前后复算。无原始深度；使用HUD姿态、roll=0。"""
import importlib.util,json,math,subprocess,sys,tempfile
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import classify
from car_pose import CarPose,OPTICAL_FROM_CORNER,load_intrinsics
from field_map import DISPLAY
INTR=load_intrinsics(str(ROOT/'config/intrinsics.yaml'))
BASELINE='febe31d'
POSES={134:(-.534,.328,-59.4),136:(-.530,.331,-59.3),137:(-.531,.324,-59.3),138:(-.532,.330,-59.3),145:(-.536,.327,-59.3)}
VIDEO='/home/dong/2026-11-16 22-30-52.mkv'

def main():
    with tempfile.TemporaryDirectory() as directory:
        source=Path(directory)/'baseline.py'
        source.write_bytes(subprocess.check_output(['git','show',f'{BASELINE}:classify.py'],cwd=ROOT))
        spec=importlib.util.spec_from_file_location('baseline_223052',source)
        before=importlib.util.module_from_spec(spec);sys.modules[spec.name]=before;spec.loader.exec_module(before)
        records=[];panels=[]
        for t,(x,y,yaw) in POSES.items():
            path=HERE/f'rgb_{t}.png'
            if not path.exists():
                full=Path(directory)/'full.png'
                subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-ss',str(t),'-i',VIDEO,'-frames:v','1',str(full)],check=True)
                frame=cv2.imread(str(full));cv2.imwrite(str(path),cv2.resize(frame[176:1616,50:2610],(1280,720)))
            rgb=cv2.imread(str(path));row=[]
            for label,module in [('BEFORE',before),('AFTER',classify)]:
                center=None
                for delta in (-.5,0,.5):
                    views=module.inspect_frame(CarPose(x,y,math.radians(yaw)),OPTICAL_FROM_CORNER,-2+delta,INTR,rgb,None,0)
                    if delta==0:center=views
                    for no in (5,6):
                        v=views[DISPLAY[no-1]]
                        records.append(dict(second=t,version=label,no=no,pitch_delta=delta,result=v.conclusion,method=v.method,reason=v.reason,
                            box=None if v.piece is None else cv2.boundingRect(v.piece)))
                panel=rgb.copy()
                for no in (5,6):
                    v=center[DISPLAY[no-1]]
                    if v.piece is not None:cv2.drawContours(panel,[v.piece],-1,(0,255,0),2)
                panel=cv2.resize(panel[245:470,230:1000],(770,225))
                canvas=np.zeros((255,770,3),np.uint8);canvas[30:]=panel
                states=' '.join(f'{no}={ {"红":"RED","蓝":"BLUE"}.get(center[DISPLAY[no-1]].conclusion, center[DISPLAY[no-1]].conclusion or "unknown")}' for no in (5,6))
                cv2.putText(canvas,f'{t}s {label}: {states}',(10,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
                row.append(canvas)
            panels.append(cv2.hconcat(row))
            print(t,[{k:r[k] for k in ('version','no','result')} for r in records if r['second']==t and r['pitch_delta']==0],flush=True)
        assert all(r['result']=='红' for r in records if r['version']=='AFTER' and r['no']==5)
        assert all(r['result']=='2' for r in records if r['version']=='AFTER' and r['no']==6 and r['second'] in (137,145))
        (HERE/'results.json').write_text(json.dumps(dict(baseline=BASELINE,depth=False,roll=0,pitch=-2,records=records),ensure_ascii=False,indent=2))
        cv2.imwrite(str(HERE/'comparison.jpg'),cv2.vconcat(panels))
if __name__=='__main__':main()
