# 完整识别流程离线模拟

输入为按当前field_map构建的12块绿色塔基、6个塔位及红蓝顶块的射线求交RGB-D画面。真值按1至6号为：一层、两层、红顶、蓝顶、空、一层。

调用实际 `FieldWatch.update` → `inspect_frame` → 停稳检查 → `AutonomousSurvey` → `TerminalDashboard`，没有直接填充识别结论或锁定值。执行器按程序下发的移动/转向/俯仰命令理想到位，使用虚拟时间；不连接雷达、相机或机器人，不发送硬件命令。

运行：

```
PYTHONDONTWRITEBYTECODE=1 python debug_captures/full_simulation/run.py
PYTHONDONTWRITEBYTECODE=1 python debug_captures/full_simulation/render_terminal.py
```

`terminal.txt`为终端实际输出，`snapshots.json`保存状态切换，`survey.json`为自主流程保存结果。`summary.json`对照合成真值，只有六塔完成且一致才通过脚本断言。PNG/GIF是上述实际输出的渲染，非现场屏幕截图。

场景没有模拟现场纸板/柜子、深度噪声、图像曝光或定位延迟。本模拟用于检查完整流程及展示终端，不代表真实场地准确率。
