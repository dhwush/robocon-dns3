"""把真实终端状态输出排版为图片；不是现场屏幕截图。"""
import json,textwrap
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
HERE=Path(__file__).resolve().parent
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font=ImageFont.truetype(FONT,20)
small=ImageFont.truetype(FONT,17)

def screen(snapshot):
    lines=[]
    for row in snapshot['rows']:
        # 中英文混合按实际像素宽度折行。
        current=''
        for c in row:
            if font.getlength(current+c)>1210:
                lines.append(current);current=c
            else:current+=c
        lines.append(current)
    h=max(820,125+len(lines)*28)
    im=Image.new('RGB',(1280,h),(22,24,30));d=ImageDraw.Draw(im)
    d.rectangle((0,0,1280,43),fill=(45,48,58))
    for x,color in [(22,'#ff6059'),(45,'#ffbd2e'),(68,'#28ca42')]:d.ellipse((x,14,x+12,26),fill=color)
    d.text((100,6),'robocon3 · 离线模拟终端输出',font=font,fill='white')
    d.text((22,54),f"T+{snapshot['time']:.1f}s  合成RGB-D / 理想执行器 / 非现场运行",font=small,fill='#e6b866')
    y=96
    for line in lines:
        ink='#d8dce7'
        if line.startswith(('已确认：','最终汇总：')):ink='#73dfa8'
        elif line.startswith(('状态：','下一步命令：')):ink='#7dcaff'
        elif line.startswith('提示：'):ink='#e6b866'
        d.text((22,y),line,font=font,fill=ink);y+=28
    return im

def run():
    snapshots=json.loads((HERE/'snapshots.json').read_text())
    images=[screen(s) for s in snapshots]
    height=max(i.height for i in images)
    frames=[]
    for im in images:
        frame=Image.new('RGB',(1280,height),(22,24,30));frame.paste(im,(0,0));frames.append(frame)
    # 以实际阶段切换顺序播放，末帧停留更长。
    frames[0].save(HERE/'terminal_flow.gif',save_all=True,append_images=frames[1:],duration=[1300]*(len(frames)-1)+[6000],loop=0)
    images[-1].save(HERE/'terminal_final.png')
    chosen=[0]
    for kind in ['执行移动','调整车头','已到位，停止']:
        hit=next((i for i,s in enumerate(snapshots) if any(kind in r for r in s['rows'])),None)
        if hit is not None and hit not in chosen:chosen.append(hit)
    chosen=chosen[:3]+[len(snapshots)-1]
    tiles=[]
    for i in chosen:
        im=images[i];tile=Image.new('RGB',(1280,height),(22,24,30));tile.paste(im,(0,0));tiles.append(tile)
    board=Image.new('RGB',(2560,height*2),(12,14,18))
    for n,im in enumerate(tiles):board.paste(im,((n%2)*1280,(n//2)*height))
    board.save(HERE/'terminal_stages.png')
    print('Rendered',len(frames),'terminal states')
if __name__=='__main__':run()
