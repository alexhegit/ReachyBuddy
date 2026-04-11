# ReachyCheese 🧀🤖

**ReachyCheese** — 全离线语音交互拍照应用，专为 Reachy Mini 桌面机器人设计。

通过语音唤醒、人脸追踪、自动对齐，让你轻松拍出完美的机器人视角照片。

![Demo](./assets/ReachyMiniChat.png)

---

## ✨ 功能特点

- **🎙️ 语音唤醒**：说 "Reachy" 唤醒机器人
- **👤 人脸追踪**：自动追踪最大人脸并对齐到画面中心
- **📸 语音拍照**：说 "cheese"、"take photo" 或 "take picture" 拍照
- **⏱️ 智能倒计时**：语音提示 "One, two, three, cheese!" 后自动拍摄
- **🖼️ 实时预览**：GUI 界面显示摄像头画面、人脸框、状态信息
- **💾 自动保存**：照片保存到 `~/Pictures/ReachyMiniPhoto/`
- **🔌 全离线运行**：无需网络，保护隐私

---

## 🔄 工作流程

```
[Sleep 待机] --"Reachy"--> [Tracking 追踪] --对齐完成--> [Armed 待命] --"cheese"--> [Countdown 倒计时] --> [Capture 拍摄]
```

1. **Sleep**：待机监听唤醒词
2. **Tracking**：追踪并对齐最大人脸
3. **Armed**：人脸已对齐，等待拍照指令
4. **Countdown**：语音提示倒计时
5. **Capture**：拍照并保存

---

## 📋 系统要求

- **操作系统**：Ubuntu 22.04+ / Linux
- **硬件**：AMD Ryzen AI 或 x86_64 平台
- **机器人**：Pollen Robotics Reachy Mini（或仅摄像头模式测试）
- **摄像头**：内置 USB 摄像头或笔记本摄像头

---

## 🛠️ 安装

### 1. 系统依赖

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip ffmpeg libsndfile1 portaudio19-dev espeak
```

### 2. 创建虚拟环境

```bash
cd /path/to/ReachyBuddy
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### 3. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 4. 安装 Reachy Mini SDK（如使用真实机器人）

```bash
pip install "reachy-mini[mujoco]"
```

### 5. 下载语音模型

默认使用 Piper-TTS 的 Ryan 语音模型，已包含在 `models/` 目录：

- `models/en-us-ryan-medium.onnx` - 英文男声（推荐）
- `models/zh_CN-huayan-medium.onnx` - 中文女声

如需其他语音，从 [Piper Voices](https://huggingface.co/rhasspy/piper-voices) 下载。

---

## 🚀 使用方法

### 启动 Reachy Mini 模拟器（可选）

```bash
reachy-mini-daemon --sim
```

### 运行 ReachyCheese

#### 使用 Reachy Mini 机器人

```bash
python ReachyCheese.py --camera-source reachy
```

#### 使用本地摄像头测试

```bash
python ReachyCheese.py --camera-source webcam --camera-index 0
```

#### 指定语音模型

```bash
python ReachyCheese.py --piper-model models/zh_CN-huayan-medium.onnx
```

---

## 🎮 语音指令

| 指令 | 说明 |
|------|------|
| "Reachy" / "Ricky" | 唤醒机器人 |
| "cheese" / "cheeze" | 拍照 |
| "take photo" / "take picture" | 拍照 |
| "photo" / "picture" | 拍照 |

---

## 🖥️ GUI 界面

支持两种 GUI 后端：

1. **Dear PyGui**（默认，功能更丰富）
2. **OpenCV**（fallback，无需额外依赖）

界面元素：
- 实时摄像头预览
- 人脸检测框（绿色）
- 画面中心准星
- 当前状态显示
- 倒计时提示
- 手动控制按钮（Wake / Take Photo / Cancel / Sleep）

---

## ⚙️ 命令行参数

```
python ReachyCheese.py [OPTIONS]

Options:
  --preview-width INT       预览窗口宽度 (默认: 640)
  --preview-height INT      预览窗口高度 (默认: 480)
  --preview-fps FLOAT       预览帧率 (默认: 20.0)
  --save-dir PATH           照片保存目录 (默认: ~/Pictures/ReachyMiniPhoto)
  --wake-word TEXT          唤醒词 (默认: reachy)
  --asr-model {tiny,base,small,medium,large}  ASR模型 (默认: base)
  --vad-silence FLOAT       VAD静音阈值秒数 (默认: 0.7)
  --vad-aggressive {0,1,2,3} VAD灵敏度 (默认: 1)
  --piper-model PATH        Piper TTS模型路径
  --piper-config PATH       Piper TTS配置文件路径
  --speaker INT             说话人ID (默认: 0)
  --camera-source {reachy,webcam} 摄像头源 (默认: reachy)
  --camera-index INT        摄像头索引 (默认: 0)
  --debug                   启用调试输出
```

---

## 📁 项目结构

```
ReachyBuddy/
├── ReachyCheese.py          # 主程序
├── ReachyCheese_spec.md     # 设计规格文档
├── requirements.txt         # Python 依赖
├── models/                  # TTS 语音模型
│   ├── en-us-ryan-medium.onnx
│   └── ...
├── utils/
│   └── asr.py              # ASR 语音识别模块
├── vision/
│   └── face_tracker.py     # 人脸追踪模块
└── assets/                  # 图片资源
```

---

## 🔧 技术栈

| 组件 | 技术 |
|------|------|
| **ASR** | faster-whisper (CPU) |
| **VAD** | webrtcvad |
| **TTS** | Piper-TTS (ONNX) |
| **人脸检测** | MediaPipe Face Detection |
| **GUI** | Dear PyGui / OpenCV |
| **机器人控制** | reachy-mini SDK |

---

## 🐛 故障排除

### 摄像头无法打开

```bash
# 检查可用摄像头
ls /dev/video*

# 测试摄像头
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

### 语音无法识别

- 检查麦克风是否被占用
- 尝试调整 `--vad-silence` 参数（0.5-1.5 之间）
- 使用 `--debug` 查看详细日志

### TTS 无声音

```bash
# 检查音频输出
speaker-test -t wav

# 检查 sounddevice
python -c "import sounddevice as sd; print(sd.query_devices())"
```

---

## 📄 许可证

MIT License - 详见 [LICENSE](./LICENSE)

---

## 🙏 致谢

- [Pollen Robotics](https://www.pollen-robotics.com/) - Reachy Mini 机器人
- [Piper TTS](https://github.com/rhasspy/piper) - 离线语音合成
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - 语音识别
- [MediaPipe](https://mediapipe.dev/) - 人脸检测
