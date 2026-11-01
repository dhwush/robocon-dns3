# FDCAN 通信协议文档

> 项目：R1_Locator_FDCAN2.0（排球定位器）  
> MCU：STM32G473  
> 总线规格：CAN Classic（FDCAN_FRAME_CLASSIC），标准帧（11bit ID）  

---

## 一、硬件配置

### 总线参数（FDCAN1 & FDCAN2 相同）

| 参数 | 值 | 说明 |
|---|---|---|
| 帧格式 | FDCAN_FRAME_CLASSIC | 经典 CAN 2.0 |
| 时钟源 | PCLK1 | APB1 总线时钟 |
| 时钟分频 | DIV1 | 不分频 |
| NominalPrescaler | 17 | 标称位时序预分频 |
| NominalTimeSeg1 | 6 | 相位段1 |
| NominalTimeSeg2 | 3 | 相位段2 |
| NominalSyncJumpWidth | 1 | 同步跳转宽度 |
| 波特率（估算） | ≈500 kbps | 基于 170MHz PCLK1 计算 |
| 自动重传 | 使能 | AutoRetransmission = ENABLE |
| 传输暂停 | 禁用 | TransmitPause = DISABLE |

### GPIO 引脚

| 总线 | RX 引脚 | TX 引脚 |
|---|---|---|
| FDCAN1 | PA11 | PA12 |
| FDCAN2 | PB5 | PB6 |

### 中断优先级

| 总线 | 中断线 | 优先级 |
|---|---|---|
| FDCAN1 | FDCAN1_IT0_IRQn | 0, 0 |
| FDCAN2 | FDCAN2_IT0_IRQn | 0, 0 |
| FDCAN2 | FDCAN2_IT1_IRQn | 0, 0 |

---

## 二、FDCAN1 —— 定位数据总线

FDCAN1 用于与上位机（运动控制主机）之间的定位数据交换。

### 接收过滤器（FDCAN1 RxFilter）

| 项目 | 值 |
|---|---|
| ID 类型 | 标准帧 |
| 过滤方式 | FDCAN_FILTER_DUAL（双精确 ID） |
| FIFO | RXFIFO0 |
| 允许 ID 1 | **0xAE** |
| 允许 ID 2 | **0xBE** |
| 其余帧 | 全部拒绝 |

---

### 📥 FDCAN1 接收报文

#### ID = 0xAE — 复位命令

| 字节 | 字段 | 类型 | 说明 |
|---|---|---|---|
| Byte0 | 命令字 | uint8 | 值为 `'r'`（ASCII 0x72）时触发定位复位 |
| Byte1~7 | 保留 | — | 忽略 |

**触发行为：**
- 置位 `fdcan_restart_pending`
- 在主循环 `Process_FDCAN_Events()` 中执行：
  1. 调用 `laser_start()` 向 FDCAN2 发送激光启动命令
  2. 置位 `restart_flag = 1`，下一次 `LocatorUpdate()` 时清零全部定位状态

---

#### ID = 0xBE — 激光修正命令

| 字节 | 字段 | 类型 | 说明 |
|---|---|---|---|
| Byte0 | 命令字 | uint8 | 值为 `'j'`（ASCII 0x6A）时触发激光位置修正 |
| Byte1~7 | 保留 | — | 忽略 |

**触发行为：**
- 置位 `fdcan_jiguang_pending`
- 在主循环中：
  1. 使用当前激光测距值修正 `lcResult.x` / `lcResult.y`（含偏移补偿）：
     - `x = dis1 - 5 + 23.06`（mm）
     - `y = dis2 - 5 - 21.46`（mm）
  2. 有效范围：200 ~ 2500 mm
  3. 调用 `jiguang_restart_position()` 通过 FDCAN1 广播修正后的完整位姿

---

### 📤 FDCAN1 发送报文

#### 定位数据广播（`send_lc`）

每 **20ms** 周期性发送，共 **3 个连续报文**，合计 24 字节，传输 6 个 float 值。

| 报文序号 | ID | 数据内容 |
|---|---|---|
| 报文 0 | **0xAA** | x（位置 X，mm，float32） + y（位置 Y，mm，float32） |
| 报文 1 | **0xAB** | r（偏航角，rad，float32） + vx（X 速度，mm/s，float32） |
| 报文 2 | **0xAC** | vy（Y 速度，mm/s，float32） + vr（角速度，rad/s，float32） |

每帧格式（8 字节）：

| 字节 | 字段 | 类型 | 说明 |
|---|---|---|---|
| Byte0~3 | 第1个数据 | float32 | 小端字节序（Little-Endian） |
| Byte4~7 | 第2个数据 | float32 | 小端字节序（Little-Endian） |

**报文头配置：**

| 字段 | 值 |
|---|---|
| IdType | FDCAN_STANDARD_ID |
| TxFrameType | FDCAN_DATA_FRAME |
| DataLength | FDCAN_DLC_BYTES_8 |
| BitRateSwitch | FDCAN_BRS_OFF |
| FDFormat | FDCAN_CLASSIC_CAN |

---

#### 激光修正后位姿广播（`jiguang_restart_position`）

激光修正触发后立即发送，帧格式与 `send_lc` 完全相同，但 ID 偏移不同：

| 报文序号 | ID | 数据内容 |
|---|---|---|
| 报文 0 | **0xCE** | x（修正后位置 X）+ y（修正后位置 Y） |
| 报文 1 | **0xCF** | r（偏航角）+ vx（X 速度） |
| 报文 2 | **0xD0** | vy（Y 速度）+ vr（角速度） |

---

## 三、FDCAN2 —— 激光测距总线

FDCAN2 用于与外部激光测距模块之间的通信。

### 接收过滤器（FDCAN2 RxFilter）

| 项目 | 值 |
|---|---|
| ID 类型 | 标准帧 |
| 过滤方式 | FDCAN_FILTER_MASK（ID & 掩码） |
| FIFO | RXFIFO1 |
| FilterID1（基准） | **0x50** |
| FilterID2（掩码） | **0x7FF**（精确匹配） |
| 其余帧 | 全部拒绝 |

---

### 📥 FDCAN2 接收报文

#### ID = 0x50 — 激光测距数据

| 字节 | 字段 | 类型 | 说明 |
|---|---|---|---|
| Byte0~3 | dis1_raw | uint32 | 激光1距离原始值，小端，单位 0.01mm（÷100 = mm） |
| Byte4~7 | dis2_raw | uint32 | 激光2距离原始值，小端，单位 0.01mm（÷100 = mm） |

**解析公式：**
```c
dis1 = (float)((rx_data[3]<<24)|(rx_data[2]<<16)|(rx_data[1]<<8)|(rx_data[0])) / 100.0f;  // mm
dis2 = (float)((rx_data[7]<<24)|(rx_data[6]<<16)|(rx_data[5]<<8)|(rx_data[4])) / 100.0f;  // mm
```

---

### 📤 FDCAN2 发送报文

#### ID = 0xAA — 激光启动命令（`laser_start`）

上位机发送 ID=0xAE 复位命令后，MCU 向 FDCAN2 发送此报文以唤醒激光测距模块。

| 字节 | 字段 | 类型 | 说明 |
|---|---|---|---|
| Byte0~7 | 全0 | uint8 | 内容全为 0x00，仅用于触发 |

---

## 四、错误处理机制

| 机制 | 说明 |
|---|---|
| Bus-Off 自动恢复 | 检测到 BUS_OFF 中断后，在主循环 `BSP_FDCAN_ProcessRecovery()` 中执行 DeInit → Re-Init |
| TX FIFO 满保护 | 发送前检查 `GetTxFifoFreeLevel` ≥ 3（LOCATOR_CAN_PACKET_COUNT），否则跳过并计数 |
| 错误计数统计 | `fdcan_rx0_error_count` / `fdcan_rx1_error_count` / `fdcan_tx_error_count` 供诊断 |
| 错误中断监听 | 监听 BUS_OFF / ERROR_WARNING / ERROR_PASSIVE / ARB_PROTOCOL_ERROR / DATA_PROTOCOL_ERROR |

---

## 五、报文 ID 汇总

| ID（hex） | 方向 | 总线 | 功能 |
|---|---|---|---|
| 0xAA | TX | FDCAN1 | 定位数据包0（x, y） |
| 0xAB | TX | FDCAN1 | 定位数据包1（r, vx） |
| 0xAC | TX | FDCAN1 | 定位数据包2（vy, vr） |
| 0xAE | RX | FDCAN1 | 复位命令（Byte0='r'） |
| 0xBE | RX | FDCAN1 | 激光位置修正命令（Byte0='j'） |
| 0xCE | TX | FDCAN1 | 激光修正后位姿包0（x, y） |
| 0xCF | TX | FDCAN1 | 激光修正后位姿包1（r, vx） |
| 0xD0 | TX | FDCAN1 | 激光修正后位姿包2（vy, vr） |
| 0x50 | RX | FDCAN2 | 激光测距数据（dis1, dis2） |
| 0xAA | TX | FDCAN2 | 激光模块启动命令（全0载荷） |

---

## 六、数据流示意

```
上位机
  │
  │── 0xAE (reset) ──────────────────────────────► FDCAN1 RX
  │                                                   │
  │◄── 0xAA/0xAB/0xAC (定位数据, 每20ms) ──────── FDCAN1 TX
  │
  │── 0xBE (激光修正) ─────────────────────────── FDCAN1 RX
  │◄── 0xCE/0xCF/0xD0 (修正后位姿) ───────────── FDCAN1 TX

激光模块
  │── 0x50 (测距数据) ─────────────────────────── FDCAN2 RX
  │◄── 0xAA (启动命令) ────────────────────────── FDCAN2 TX
```

