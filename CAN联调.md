# CAN 位姿联调

电脑接摄像头、小车主机读取 CAN：

```bash
python view_field.py extend --pose-source ssh
# 或打开地图后回车开相机
python map_test.py --pose-source ssh
```

默认主机 volleyball@192.168.3.194，接口 can0；SSH 在终端提示时输入密码。
可用 --ssh-host 和 --can-interface 修改。密码不存进代码。

比赛时在小车主机运行：

```bash
python view_field.py extend --pose-source local
python map_test.py --pose-source local
```

运行前在 config/field.yaml 中保持 center_to_corner: [0.0, 0.0]（兼容字段；定位点为车身右前角），确认定位坐标与场地坐标对齐后设置 aligned: true。
CAN 使用定位板 CAN1：0xAA 的两个小端 float32 是 x/y（毫米转米），0xAB 的第一个小端 float32 是旋转角（弧度）。只接收，不发送复位或控制命令。
相邻位置与角度帧间隔最多 100 ms，接收端以单调时钟匹配相机帧；超过 150 ms 无匹配则清除旧框并暂停识别。SSH 网络延迟仍会影响运动时的精度，当前协议没有板端时间戳。
SSH/本地模式下，地图点击的位置不会覆盖实时 CAN 位姿。离线手动调试使用 --pose-source static；ROS 使用 view_field.py 的 --pose-source ros。

2026-11-11 已通过 SSH 确认 can0 为 UP、1 Mbit/s，3 秒观测未收到 0xAA/0xAB；需确认定位板供电、CAN1 接线及周期发送后完成实车验证。

## 慢速移动验证（CAN 为右前角位姿，无中心偏移）

电脑连接摄像头后直接运行：

```bash
python view_field.py extend --pose-source ssh --verify-motion
```

地图入口：`python map_test.py --pose-source ssh --verify-motion`，回车打开相机。
主机本地运行将 ssh 改成 local。
验证模式明确将 CAN x/y/yaw 作为右前角在场地地图中的位姿，偏移固定为零；不会修改 aligned 配置。若 CAN 与地图方向或零点不同，验证画面会偏，不能据此确认对齐。
窗口显示实时 x/y/yaw、相对第一组有效位姿的变化以及 CAN 接收年龄。缓慢前后、左右移动，再缓慢原地旋转，观察识别框是否仍贴合固定的建塔点。按 s 保存当前位姿、画面与诊断结果，q 退出。断流超过 150 ms 清除旧框并暂停识别。
旧章节中要求测量中心偏移的说明不再适用于此处的右前角输入。

## 三种运行模式与固定计划

统一入口 `python map_test.py`，默认手动离线。窗口顶部按钮可切换：

- 手动离线：笔记本接摄像头，不连接 SSH。点击起始光心生成路线；点路线圆点表示到达，再点同一站切换观察目标。可退格修改实际 yaw，回车在此位置和角度识别。
- 笔记本 SSH：笔记本接摄像头，SSH 从主机 can0 获取右前角 x/y/yaw。
- 主机 CAN：主机接摄像头，直接读取本地 can0，用于比赛。

也可用 `--pose-source static/ssh/local` 指定初始模式。按钮切换在地图窗口进行；相机打开后按 q 返回地图再切换，旧 CAN/SSH 进程会关闭。
地图选定新起始点时生成一次路线、各站目标 yaw 和俯仰。输入实际 yaw、选择到达站点、切换模式和实时位姿变化都不改变这份计划。实际俯仰仍来自相机惯性数据，用于投影；固定目标俯仰单独显示。
打开相机后另有 route 地图显示实际光心和朝向；SSH/本地模式来自 CAN。可在 route 地图点路线站点表示到达，再点同站切换固定观察目标。地图点击不会覆盖实时 CAN 位姿。相机窗口显示固定计划及识别结果，不再混入另算的转向建议。
CAN 模式按右前角场地位姿的约定运行，不添加中心偏移；原点与轴方向仍需与场地一致。按钮不启动 CAN 接口，主机需预先启用 can0。
