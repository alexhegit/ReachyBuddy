# ReachyBuddy 🤖

**ReachyBuddy** — A multi-mode robot application for Reachy Mini with pluggable modes:

- **🧀 Cheese Mode**: Voice photo capture with face tracking
- **🔒 Guard Mode**: Multi-modal security monitoring (placeholder)
- **💬 Chat Mode**: Voice conversation with LLM (placeholder)  
- **🤖 Agent Mode**: AI agent with tools (placeholder)

![Demo](./assets/ReachyMiniChat.png)

---

## ✨ Modes

### Cheese Mode (Implemented)

Voice-interactive photo capture with automatic face tracking and alignment.

**Workflow:**
```
[Sleep] --"Reachy"--> [Tracking] --aligned--> [Armed] --"cheese"--> [Countdown] --> [Capture]
```

**Features:**
- Voice wake-up ("Reachy")
- Automatic face tracking with head + body compensation
- Voice photo capture ("cheese", "take photo")
- Smart countdown with audio prompts
- Real-time GUI preview with face bounding box

### Guard / Chat / Agent Modes (Placeholders)

These modes are planned for future development. Running them will show a "coming soon" message.

---

## 📋 Requirements

- **OS**: Ubuntu 22.04+ / Linux
- **Hardware**: AMD Ryzen AI or x86_64 platform
- **Robot**: Pollen Robotics Reachy Mini (with built-in camera)

---

## 🛠️ Installation

### 1. System Dependencies

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip ffmpeg libsndfile1 portaudio19-dev espeak v4l-utils
```

### 2. Create Virtual Environment

```bash
cd /path/to/ReachyBuddy
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Reachy Mini SDK (for physical robot)

```bash
pip install "reachy-mini[mujoco]"
```

### 5. Voice Models

Default uses Piper-TTS Ryan voice model (included in `models/`):

- `models/en-us-ryan-medium.onnx` - English male voice (recommended)
- `models/zh_CN-huayan-medium.onnx` - Chinese female voice

Download additional voices from:
- [Piper Voices (HuggingFace)](https://huggingface.co/rhasspy/piper-voices) - Full voice collection
- [Piper Release v0.0.2](https://github.com/rhasspy/piper/releases/tag/v0.0.2) - Pre-built voice files (.onnx + .onnx.json)

---

## 🚀 Usage

### Start Reachy Mini Simulator (optional)

```bash
reachy-mini-daemon --sim
```

### Run Cheese Mode

#### With Reachy Mini Robot

Uses the robot's built-in camera and controls head/body movement:

```bash
python main.py --cheese --camera-source reachy
```

#### With Local Webcam (for testing)

Uses your computer's webcam. **Note:** Robot movement is disabled in webcam mode as there is no physical robot to control.

```bash
python main.py --cheese --camera-source webcam --camera-index 0
```

---

## 📷 Camera Source Selection

### Overview

ReachyBuddy supports two camera source modes:

| Mode | Camera Source | Robot Movement | Use Case |
|------|--------------|----------------|----------|
| `reachy` | Reachy Mini's built-in camera | ✅ Yes | Physical robot or simulator |
| `webcam` | Local USB/webcam | ❌ No | Testing without robot |

### `--camera-source reachy` (Recommended)

**Automatically uses Reachy Mini's built-in camera** through the `reachy-mini` SDK.

**Advantages:**
- ✅ Guaranteed to use the correct camera regardless of other USB devices
- ✅ Full robot control (head tracking, body rotation)
- ✅ Optimized for Reachy Mini's camera parameters

**Usage:**
```bash
# Physical robot
python main.py --cheese --camera-source reachy

# Simulator
reachy-mini-daemon --sim
python main.py --cheese --camera-source reachy
```

### `--camera-source webcam` (Testing Only)

**Uses OpenCV `VideoCapture(index)` to access local cameras.**

**Limitations:**
- ❌ No robot movement (no physical robot connected)
- ❌ Camera selection depends on system enumeration order
- ✅ Good for testing face detection and GUI without robot

**List available cameras:**
```bash
# List video devices
v4l2-ctl --list-devices

# Or check device files
ls -la /dev/video*
```

**Select specific camera:**
```bash
# Use first camera (index 0, default)
python main.py --cheese --camera-source webcam --camera-index 0

# Use second camera (index 1)
python main.py --cheese --camera-source webcam --camera-index 1

# Use third camera (index 2)
python main.py --cheese --camera-source webcam --camera-index 2
```

**Finding the right camera index:**

If you have multiple cameras and aren't sure which index corresponds to which device:

```python
# test_cameras.py
import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"Camera {i}: OK - Resolution {frame.shape[1]}x{frame.shape[0]}")
        else:
            print(f"Camera {i}: Opened but cannot read frames")
    else:
        print(f"Camera {i}: Cannot open")
    cap.release()
```

### Multiple Camera Setup Example

Scenario: Your computer has:
- Built-in laptop camera (`/dev/video0`)
- External USB webcam (`/dev/video1`)
- Reachy Mini connected (`/dev/video2`, `/dev/video3` - stereo cameras)

**To use Reachy Mini:**
```bash
# Automatically uses Reachy Mini's cameras via SDK
python main.py --cheese --camera-source reachy
```

**To use external USB webcam:**
```bash
# Try index 1 (usually the first USB camera)
python main.py --cheese --camera-source webcam --camera-index 1
```

**To verify which camera is being used:**
1. Cover the camera lens with your hand
2. Check if the video feed goes dark
3. If so, that's the camera being used

---

## 📷 Camera Tuning

Reachy Mini's camera may produce darker images compared to webcams due to hardware ISP differences. Use the built-in tuning tools to optimize image quality.

### GUI Tuning Tool (Recommended)

```bash
# Auto-detect Reachy camera and launch GUI
python utils/camera_tuning_gui.py
```

**Features:**
- Real-time preview with parameter adjustment
- Visual trackbars for all camera parameters
- Save/load configuration profiles
- Automatic Reachy camera detection
- Safety protection against modifying wrong cameras

**Workflow:**
1. Launch GUI: `python utils/camera_tuning_gui.py`
2. Adjust sliders (brightness, contrast, saturation, etc.)
3. Click **Save** button and name your profile (e.g., `indoor_bright`)
4. Use profile in your application:
   ```bash
   python main.py --cheese --camera-source reachy --camera-profile indoor_bright
   ```

### CLI Tuning Tool

```bash
# View current parameters
python utils/camera_tuning.py --list

# Set parameters
python utils/camera_tuning.py --set brightness=10,contrast=15,saturation=55

# Save profile
python utils/camera_tuning.py --save my_profile

# Load profile
python utils/camera_tuning.py --load my_profile

# Reset to defaults
python utils/camera_tuning.py --reset
```

### Recommended Settings

| Scenario | Brightness | Contrast | Saturation | Sharpness |
|----------|-----------|----------|------------|-----------|
| Factory Default | 0 | 1 | 48 | 2 |
| Indoor Bright | 5-10 | 10-15 | 55-60 | 3 |
| Low Light | 15-20 | 20-25 | 60-65 | 4 |

**Note:** Camera profiles are stored in `~/.config/reachy_mini/`

See [docs/face_tracking.md](./docs/face_tracking.md) for detailed camera documentation.

---

## 🎮 Voice Commands (Cheese Mode)

| Command | Action |
|---------|--------|
| "Reachy" / "Ricky" | Wake up the robot |
| "cheese" / "cheeze" | Take photo |
| "take photo" / "take picture" | Take photo |
| "photo" / "picture" | Take photo |

---

## 🖥️ GUI Interface

Two GUI backends supported:

1. **Dear PyGui** (default, more features)
2. **OpenCV** (fallback, no extra dependencies)

**Interface elements:**
- Real-time camera preview
- Face detection bounding box (green)
- Center crosshair
- Current state display (colored dot)
- dx/dy tracking values
- Countdown overlay
- Manual control buttons (Wake / Take Photo / Sleep)

**State Indicator Colors:**
- 🔘 Gray: Sleep mode (not tracking)
- 🔵 Cyan: Tracking mode (following face)
- 🟢 Green: Armed mode (aligned, ready for photo)
- 🔴 Red: Countdown in progress

---

## ⚙️ Command Line Arguments

```
python main.py --cheese [OPTIONS]

Global Options:
  --camera-source {reachy,webcam}  Camera source (default: reachy)
  --camera-index INT               Webcam index for webcam mode (default: 0)
  --preview-width INT              Preview width (default: 640)
  --preview-height INT             Preview height (default: 480)
  --preview-fps FLOAT              Preview FPS (default: 20.0)
  --gui-backend {auto,dpg,cv2,none}  GUI backend (default: auto)
  --debug                          Enable debug output

ASR Options:
  --asr-model {tiny,base,small,medium,large}  ASR model size (default: base)
  --vad-silence FLOAT              VAD silence threshold seconds (default: 0.7)
  --vad-aggressive {0,1,2,3}       VAD aggressiveness (default: 1)

TTS Options:
  --piper-model PATH               Piper TTS model path
  --piper-config PATH              Piper TTS config path
  --speaker INT                    Speaker ID (default: 0)

Cheese Mode Options:
  --save-dir PATH                  Photo save directory
  --wake-word TEXT                 Wake word (default: reachy)
  --timeout FLOAT                  Command timeout in seconds (default: 12)
  --camera-profile NAME            Camera profile to load (created via tuning tools)
```

---

## 📁 Project Structure

```
ReachyBuddy/
├── main.py                  # Entry point
├── core/                    # Shared framework
│   ├── base_app.py         # Base mode class
│   ├── runtime.py          # Robot/webcam runtimes
│   └── event_bus.py        # Event system
├── modes/                   # Mode implementations
│   ├── cheese/             # Photo capture mode
│   ├── guard/              # Security monitoring (placeholder)
│   ├── chat/               # Voice chat (placeholder)
│   └── agent/              # AI agent (placeholder)
├── utils/                   # Shared utilities
│   ├── asr.py              # Speech recognition
│   └── tts_engine.py       # Text-to-speech
├── vision/                  # Computer vision
│   └── face_tracker.py     # Face detection
├── docs/                    # Documentation
│   └── face_tracking.md    # Camera and face tracking guide
├── requirements/            # Per-mode dependencies
└── models/                  # Voice models
```

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|------------|
| **ASR** | faster-whisper (CPU) |
| **VAD** | webrtcvad |
| **TTS** | Piper-TTS (ONNX) |
| **Face Detection** | MediaPipe Face Detection |
| **GUI** | Dear PyGui / OpenCV |
| **Robot Control** | reachy-mini SDK |

---

## 🐛 Troubleshooting

### Camera Not Opening

```bash
# Check available cameras
ls /dev/video*

# List camera details
v4l2-ctl --list-devices

# Test specific camera
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened()); cap.release()"
```

### Reachy Mini Not Detected

```bash
# Check if robot is connected
lsusb | grep -i reachy

# Check device permissions
ls -la /dev/ttyUSB* /dev/ttyACM* 2>/dev/null

# Try with simulator
reachy-mini-daemon --sim
```

### Wrong Camera Selected

If using `--camera-source webcam` and getting the wrong camera:

1. List all cameras: `v4l2-ctl --list-devices`
2. Test each index: `python -c "import cv2; cap = cv2.VideoCapture(1); print(cap.isOpened())"`
3. Use correct index: `--camera-source webcam --camera-index 1`

### Image Too Dark (Reachy Mode)

Reachy Mini's camera may produce darker images than webcams due to hardware differences.

**Solution:**
```bash
# Use camera tuning tool to adjust parameters
python utils/camera_tuning_gui.py

# Recommended starting values:
# brightness: 5-10, contrast: 10-15, saturation: 55-60

# Save profile and use it
python main.py --cheese --camera-source reachy --camera-profile my_profile
```

See [docs/face_tracking.md](./docs/face_tracking.md) for detailed explanation.

### Speech Not Recognized

- Check if microphone is occupied by another application
- Try adjusting `--vad-silence` parameter (between 0.5-1.5)
- Use `--debug` for detailed logs

### No TTS Audio

```bash
# Check audio output
speaker-test -t wav

# Check sounddevice
python -c "import sounddevice as sd; print(sd.query_devices())"
```

---

## 📄 License

MIT License - See [LICENSE](./LICENSE)

---

## 🙏 Acknowledgements

- [Pollen Robotics](https://www.pollen-robotics.com/) - Reachy Mini robot
- [Piper TTS](https://github.com/rhasspy/piper) - Offline text-to-speech
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - Speech recognition
- [MediaPipe](https://mediapipe.dev/) - Face detection
