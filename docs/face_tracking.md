# Face Tracking 与 Camera 配置指南

本文档总结 Reachy Mini 项目中人脸跟踪功能的实现细节、不同 camera 模式的行为差异以及 camera tuning 工具的使用方法。

## 目录

- [Reachy vs Webcam 模式对比](#reachy-vs-webcam-模式对比)
- [图像质量差异分析](#图像质量差异分析)
- [人脸跟踪性能对比](#人脸跟踪性能对比)
- [Camera Tuning 工具使用指南](#camera-tuning-工具使用指南)
- [故障排查](#故障排查)

---

## Reachy vs Webcam 模式对比

### 模式选择

```bash
# Reachy 模式（使用机器人内置摄像头）
python -m main --cheese --camera-source reachy

# Webcam 模式（使用电脑外接/内置摄像头）
python -m main --cheese --camera-source webcam --camera-index 0
```

### 核心差异

| 特性 | Reachy 模式 | Webcam 模式 |
|------|------------|-------------|
| **硬件 ISP** | 无/弱 | 强（内置 ISP 芯片） |
| **默认格式** | MJPG (SDK 设置) | YUYV (OpenCV 默认) |
| **帧率** | 30-60 fps | 5-30 fps（取决于格式） |
| **曝光控制** | 需手动调节 | 自动优化 |
| **适用场景** | 实际控制机器人 | 本地开发测试 |
| **电机控制** | 实际控制头部/身体 | 仅模拟（无实际动作） |

### 运行时架构差异

```
Reachy 模式:
Camera → Reachy SDK → MediaManager → MJPG 解码 → 应用
                              ↓
                         电机控制（实际运动）

Webcam 模式:
Camera → OpenCV VideoCapture → YUYV/MJPG → 应用
                              ↓
                         电机控制（空操作）
```

---

## 图像质量差异分析

### 问题现象

**Reachy 模式图像偏暗、颜色暗淡**，而 Webcam 模式色彩鲜艳、清晰度高。

### 根本原因

#### 1. 硬件 ISP 差异

**Webcam（如 Logitech、笔记本内置）:**
```
光线 → 传感器 → 硬件 ISP 芯片 → USB → 电脑
                ↑
           自动曝光 (AE)
           自动白平衡 (AWB)
           自动增益 (AGC)
           降噪 (NR)
           边缘增强
           色彩增强
```

**Reachy Mini:**
```
光线 → 传感器 → 原始数据 → USB → 电脑
         ↑
    无硬件 ISP
    或 ISP 默认关闭
```

#### 2. SDK 默认设置

Reachy SDK 默认使用 **MJPG 格式**以获得高帧率（30-60fps），但这会：
- 降低单帧曝光时间（60fps 时仅 16ms 曝光）
- MJPG 压缩损失部分色彩信息

#### 3. YUYV vs MJPG 曝光时间对比

| 格式 | 分辨率 | FPS | 单帧曝光时间 | 图像质量 |
|------|--------|-----|-------------|---------|
| YUYV | 1920x1080 | 5 | ~200ms | ✅ 亮、清晰 |
| MJPG | 1920x1080 | 60 | ~16ms | ❌ 暗、压缩损失 |

### 解决方案

使用 **Camera Tuning 工具**手动调节 Reachy 的硬件参数：

```bash
python utils/camera_tuning_gui.py
```

推荐温和调整值：
```
brightness: 5-10      (提亮)
contrast: 10-15       (增强层次)
saturation: 55-60     (提升色彩)
sharpness: 3          (适度锐化)
gamma: 110            (提亮中间调)
```

---

## 人脸跟踪性能对比

### 问题现象

**Webcam 模式下人脸跟踪卡顿、延迟严重**，而 Reachy 模式流畅。

### 根本原因

#### 1. 帧率限制（YUYV 格式）

Webcam 默认使用 **YUYV 格式**，在 1080p 分辨率下：
- 带宽限制导致仅 **5fps**
- MediaPipe 需要 15-30fps 才能流畅跟踪
- 低帧率导致人脸位置变化与机械动作无法同步

#### 2. 曝光 vs 帧率权衡

```
YUYV 1080p@5fps:   每帧 200ms 曝光 → 画面亮但帧率低
MJPG 1080p@60fps:  每帧 16ms 曝光  → 画面暗但帧率高
```

**人脸跟踪需要高帧率**，因此 Reachy SDK 强制使用 MJPG。

#### 3. 实时性要求

MediaPipe Face Detection 性能需求：
- **最低**: 10 fps（严重卡顿）
- **推荐**: 30 fps（流畅）
- **最佳**: 60 fps（实时）

### 影响

| 模式 | 帧率 | 跟踪效果 | 说明 |
|------|------|---------|------|
| Webcam (YUYV) | 5 fps | ❌ 严重卡顿 | 动作延迟 200ms+ |
| Webcam (MJPG) | 30 fps | ⚠️ 可接受 | 需强制设置格式 |
| Reachy (SDK) | 30-60 fps | ✅ 流畅 | SDK 自动优化 |

### 开发建议

1. **开发阶段**: 使用 `--camera-source webcam` 快速验证逻辑
2. **调试跟踪**: 必须使用 `--camera-source reachy` 测试实际效果
3. **图像质量**: Reachy 模式配合 Camera Tuning 工具优化

---

## Camera Tuning 工具使用指南

### 工具说明

为避免 Reachy 图像质量差影响人脸识别，我们提供了两个调参工具：

| 工具 | 类型 | 用途 |
|------|------|------|
| `camera_tuning_gui.py` | GUI | 实时预览调参 |
| `camera_tuning.py` | CLI | 命令行批量操作 |

### 工具安装

```bash
# 确保有 v4l2-ctl
sudo apt install v4l-utils

# 工具已内置在项目中
chmod +x utils/camera_tuning_gui.py
chmod +x utils/camera_tuning.py
```

### GUI 工具使用

#### 启动

```bash
# 自动探测 Reachy 摄像头
python utils/camera_tuning_gui.py

# 手动指定设备
python utils/camera_tuning_gui.py --device /dev/video0
```

#### 界面说明

```
┌─────────────────┬─────────────────────┐
│   实时预览画面   │   Parameters        │
│                 │   ─────────────     │
│  (640x480)      │   brightness [██░]  │
│                 │   contrast   [█░░]  │
│                 │   saturation [██░]  │
│                 │   ...               │
│                 │                     │
│                 │   ┌────┐ ┌────┐     │
│                 │   │Save│ │Load│     │
│                 │   └────┘ └────┘     │
│                 │   ┌────┐ ┌────┐     │
│                 │   │Reset│ │Quit│    │
│                 │   └────┘ └────┘     │
└─────────────────┴─────────────────────┘
```

#### 操作步骤

1. **启动工具**
   ```bash
   python utils/camera_tuning_gui.py
   ```

2. **调整参数**
   - 拖动右侧滑块实时调整
   - 观察左侧预览画面变化
   - 绿色 = 默认值，橙色 = 已修改

3. **保存配置**
   - 点击 **Save** 按钮
   - 输入配置名（如 `indoor_bright`）
   - 配置文件保存到 `~/.config/reachy_mini/`

4. **应用配置**
   ```bash
   python -m main --cheese --camera-source reachy --camera-profile indoor_bright
   ```

### CLI 工具使用

#### 常用命令

```bash
# 查看当前参数
python utils/camera_tuning.py --list

# 查看指定设备参数
python utils/camera_tuning.py --device /dev/video0 --list

# 设置参数
python utils/camera_tuning.py --set brightness=10,contrast=15,saturation=55

# 保存当前配置
python utils/camera_tuning.py --save my_profile

# 加载配置
python utils/camera_tuning.py --load my_profile

# 重置为默认值
python utils/camera_tuning.py --reset

# 交互模式
python utils/camera_tuning.py --interactive
```

### 安全机制（重要！）

#### 自动摄像头识别

工具会**自动识别 Reachy 摄像头**，防止误操作其他设备：

```bash
# ✅ 自动探测 Reachy
python utils/camera_tuning.py --list
# 输出: 🔍 Auto-detected Reachy camera: /dev/video0

# ❌ 尝试修改笔记本摄像头（被阻止）
python utils/camera_tuning.py --device /dev/video2 --set brightness=10
# 输出: ❌ ERROR: This is NOT a Reachy camera!
#        Modification of non-Reachy cameras is FORBIDDEN.
```

#### 强制允许其他摄像头

**危险操作！**仅在明确知道后果时使用：

```bash
python utils/camera_tuning.py --device /dev/video2 --set brightness=10 --allow-any-camera
```

### 配置存储

配置文件位置：
```
~/.config/reachy_mini/
├── factory_default.json
├── indoor_bright.json
├── low_light.json
└── outdoor.json
```

配置文件格式：
```json
{
  "name": "indoor_bright",
  "device": "/dev/video0",
  "params": {
    "brightness": 10,
    "contrast": 15,
    "saturation": 55,
    ...
  }
}
```

### 推荐配置

| 场景 | brightness | contrast | saturation | sharpness |
|------|-----------|----------|------------|-----------|
| 出厂默认 | 0 | 1 | 48 | 2 |
| 室内明亮 | 5-10 | 10-15 | 55-60 | 3 |
| 低光环境 | 15-20 | 20-25 | 60-65 | 4 |
| 过度曝光 | -5 | 5 | 50 | 2 |

---

## 故障排查

### Q1: Reachy 模式图像很暗

**原因**: SDK 使用 MJPG 60fps，曝光时间短。

**解决**:
```bash
# 1. 使用 tuning 工具调整
python utils/camera_tuning_gui.py
# 2. 调高 brightness 和 gamma
# 3. 保存配置并在启动时加载
python -m main --cheese --camera-source reachy --camera-profile my_config
```

### Q2: Webcam 模式人脸跟踪卡顿

**原因**: Webcam 默认 YUYV 5fps，帧率太低。

**解决**: 这不是 bug，是硬件限制。人脸跟踪测试必须使用 Reachy 模式。

### Q3: 修改了错误摄像头的参数

**原因**: 多摄像头时指定了错误的 `--device`。

**解决**:
```bash
# 1. 重置该摄像头为默认值
python utils/camera_tuning.py --device /dev/video2 --reset --allow-any-camera

# 2. 以后使用自动探测
python utils/camera_tuning.py --list  # 自动找 Reachy
```

### Q4: Camera tuning 工具找不到 Reachy

**排查**:
```bash
# 检查摄像头是否被识别
v4l2-ctl --list-devices | grep -i reachy

# 检查设备节点
ls -la /dev/video*

# 手动指定设备
python utils/camera_tuning_gui.py --device /dev/video0
```

---

## 附录：v4l2-ctl 工具完整指南

### 安装 v4l2-ctl

```bash
# Ubuntu/Debian
sudo apt install v4l-utils

# 验证安装
v4l2-ctl --version
```

### 列举视频设备

```bash
# 列出所有视频设备
v4l2-ctl --list-devices
```

**输出示例（多摄像头场景）**:
```
Integrated Camera: Integrated C (usb-0000:00:14.0-8):
	/dev/video0
	/dev/video1

Reachy Mini Camera: Reachy Mini (usb-0000:00:14.0-3):
	/dev/video2
	/dev/video3
```

### 查看设备详细信息

```bash
# 查看基本信息
v4l2-ctl -d /dev/video0 --info

# 输出示例
Driver Info:
	Driver name      : uvcvideo
	Card type        : Reachy Mini Camera: Reachy Mini
	Bus info         : usb-0000:00:14.0-3
	Driver version   : 6.2.0
	Capabilities     : 0x84a00001
		Video Capture
		Streaming
```

### 查看支持的格式和分辨率

```bash
# 查看所有支持的格式
v4l2-ctl -d /dev/video0 --list-formats

# 输出示例
ioctl: VIDIOC_ENUM_FMT
	Type: Video Capture

	[0]: 'MJPG' (Motion-JPEG, compressed)
	[1]: 'YUYV' (YUYV 4:2:2)
```

```bash
# 查看格式支持的分辨率和帧率
v4l2-ctl -d /dev/video0 --list-formats-ext

# 输出示例
	[0]: 'MJPG' (Motion-JPEG, compressed)
		Size: Discrete 1920x1080
			Interval: Discrete 0.017s (60.000 fps)
		Size: Discrete 1280x720
			Interval: Discrete 0.033s (30.000 fps)

	[1]: 'YUYV' (YUYV 4:2:2)
		Size: Discrete 1920x1080
			Interval: Discrete 0.200s (5.000 fps)
```

### 查看当前参数

```bash
# 查看所有可调参数
v4l2-ctl -d /dev/video0 --all

# 或查看特定参数
v4l2-ctl -d /dev/video0 --get-ctrl brightness,contrast,saturation

# 输出示例
brightness: 0
contrast: 1
saturation: 48
```

### 设置参数

```bash
# 设置单个参数
v4l2-ctl -d /dev/video0 --set-ctrl brightness=10

# 同时设置多个参数
v4l2-ctl -d /dev/video0 --set-ctrl brightness=10,contrast=15,saturation=55

# 重置为默认值（手动设置）
v4l2-ctl -d /dev/video0 --set-ctrl brightness=0,contrast=1,saturation=48,sharpness=2
```

### 查看参数取值范围

```bash
# 查看参数的最小/最大值
v4l2-ctl -d /dev/video0 --info | grep -A1 "brightness\|contrast"

# 或使用 --all 查看完整信息
v4l2-ctl -d /dev/video0 --all | grep -A5 "User Controls"

# 输出示例
User Controls
                     brightness 0x00980900 (int)    : min=-64 max=64 step=1 default=0 value=0
                       contrast 0x00980901 (int)    : min=0 max=95 step=1 default=1 value=1
                     saturation 0x00980902 (int)    : min=0 max=100 step=1 default=48 value=48
```

### 实时预览调参

```bash
# 使用 ffplay 实时查看（按 q 退出）
ffplay /dev/video0

# 在另一个终端调整参数，实时观察效果
v4l2-ctl -d /dev/video0 --set-ctrl brightness=20
```

### 批量保存和恢复参数

```bash
# 保存当前所有参数到文件
v4l2-ctl -d /dev/video0 --all > camera_backup.txt

# 提取参数值并恢复（手动方式）
# 推荐直接使用 camera_tuning.py 工具
python utils/camera_tuning.py --save backup_profile
python utils/camera_tuning.py --load backup_profile
```

### 检查设备能力

```bash
# 查看设备支持的所有控制项
v4l2-ctl -d /dev/video0 --list-ctrls

# 查看扩展控制
v4l2-ctl -d /dev/video0 --list-ctrls-menus
```

### 故障排查命令

```bash
# 检查设备是否被占用
lsof /dev/video0
fuser /dev/video0

# 检查内核日志（排查驱动问题）
dmesg | grep -i video
dmesg | grep -i uvc

# 重置 USB 设备（谨慎使用）
# 先找到 bus 和 device
lsusb | grep -i reachy
# 输出: Bus 001 Device 005: ID xxxx:xxxx ...
# 然后重置
sudo usbreset 001/005
```

### 常用参数速查表

| 参数 | 命令 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| 亮度 | `brightness` | 0 | -64 ~ 64 | 整体明暗 |
| 对比度 | `contrast` | 1 | 0 ~ 95 | 明暗差异 |
| 饱和度 | `saturation` | 48 | 0 ~ 100 | 色彩浓度 |
| 色调 | `hue` | 0 | -2000 ~ 2000 | 色彩偏移 |
| 伽马 | `gamma` | 100 | 80 ~ 160 | 中间调亮度 |
| 增益 | `gain` | 32 | 0 ~ 255 | 信号增益 |
| 锐度 | `sharpness` | 2 | 0 ~ 7 | 边缘清晰度 |
| 背光补偿 | `backlight_compensation` | 2 | 0 ~ 10 | 逆光补偿 |
| 自动曝光 | `auto_exposure` | 3 | 0 ~ 3 | 3=自动模式 |
| 曝光时间 | `exposure_time_absolute` | 166 | 3 ~ 2047 | 需关闭自动曝光 |

### v4l2-ctl vs camera_tuning 工具对比

| 需求 | v4l2-ctl | camera_tuning.py | camera_tuning_gui.py |
|------|----------|------------------|---------------------|
| 快速查看参数 | ✅ | ✅ | ✅ |
| 批量设置参数 | ✅ | ✅ | ✅ |
| 实时预览效果 | ❌ | ❌ | ✅ |
| 保存/加载配置 | ❌ | ✅ | ✅ |
| 安全保护（防误操作） | ❌ | ✅ | ✅ |
| 自动探测 Reachy | ❌ | ✅ | ✅ |
| 图形化调节 | ❌ | ❌ | ✅ |

**推荐**：日常使用 `camera_tuning_gui.py`，调试时可用 `v4l2-ctl` 快速验证。

---

## 总结

| 需求 | 推荐模式 | 额外配置 |
|------|---------|---------|
| 开发调试逻辑 | Webcam | 无需调参 |
| 测试人脸跟踪 | Reachy | 必须 |
| 优化图像质量 | Reachy | Camera Tuning |
| 保存多个场景 | Reachy | 配置文件 + `--camera-profile` |

**核心原则**: 
- Webcam 仅用于快速验证逻辑，**不能**用于测试人脸跟踪效果
- Reachy 模式必须配合 Camera Tuning 才能获得最佳图像质量
- 始终使用 `--camera-profile` 保存和加载优化后的参数
