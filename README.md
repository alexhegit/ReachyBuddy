# ReachyBuddy 🤖

**ReachyBuddy** — A multi-mode robot application for Reachy Mini with pluggable modes:

- **🧀 Cheese Mode**: Voice-interactive photo capture with automatic face tracking and alignment
- **🔒 Guard Mode**: Multi-modal security monitoring with Ollama VLM analysis, head scanning, and voice alerts
- **💬 Chat Mode**: Voice/text conversation with Ollama LLM, Piper-TTS, and emotion-driven robot actions
- **🤖 Agent Mode**: AI agent with tool calling via Hermes API, supporting voice/text interaction and robot control

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
- Voice wake-up with fuzzy matching ("Reachy", "Ricky", "Richie", "reaching")
- Automatic face tracking with proportional head control and body-follows-head compensation
- Voice photo capture ("cheese", "take photo", "picture")
- Smart countdown with audio prompts
- Real-time OpenCV GUI preview with face bounding box, center crosshair, and state indicator
- Headless mode (`--gui-backend none`) for daemon/server deployments
- 30-second no-face timeout returns to sleep
- "sleep" / "stop" / "cancel" voice commands work in any active state

### Guard Mode (Implemented)

Multi-modal security monitoring using an Ollama vision-language model. The robot periodically analyzes the camera feed, scans its head, and speaks voice alerts when it detects people or unusual activity.

**Workflow:**
```
[Camera feed] --> [VLM analysis every N seconds] --(not OK)--> [Voice alert + screenshot]
                 |                                    |
                 +--(OK / empty)--> [Continue scanning]
```

**Features:**
- Periodic VLM analysis via Ollama `/api/chat` with image input
- Configurable analysis interval (default: 8s)
- Automatic head scanning (`--scan` / `--no-scan`)
- Voice alerts via Piper-TTS when something noteworthy is detected
- Alert screenshots saved to `~/Pictures/ReachyGuard`
- Works with physical Reachy robot or local webcam
- Bypasses `HTTP_PROXY` for local Ollama requests
- Headless mode support (`--gui-backend none`)

### Chat Mode (Implemented)

Voice conversation with Ollama LLM. The robot acts as a voice assistant, speaking replies via Piper-TTS and performing emotion-driven actions (head movements, recorded moves) based on conversation context.

**Workflow:**
```
[ASR 4s recording] --> [transcription] --> [Ollama LLM] --> [emotion analysis] --> [Piper TTS + robot actions]
```

**Features:**
- ASR input with fixed 4s recording (default) or text input (`--no-asr`)
- Voice assistant persona with enthusiastic, concise replies
- Streaming-style synchronous chat via Ollama `/api/chat`
- Conversation history (configurable size)
- Emotion analysis of LLM replies (positive / negative / question / activity / neutral)
- Emotion-driven recorded moves from Pollen emotions and dances libraries
- Gentle mode for subtler actions (`--gentle`)
- Piper-TTS fully offline speech synthesis
- Supports English and Chinese conversation (auto-detected)

### Agent Mode (Implemented)

AI agent with tool calling via Hermes API. The robot can execute tools, control its head, take photos, and perform multi-step tasks.

**Workflow:**
```
[ASR/text input] --> [Hermes API with tools] --> [tool execution] --> [TTS response]
```

**Features:**
- Hermes API integration (OpenAI-compatible, local hermes-agent)
- Tool calling support (move_head, take_photo, get_time, execute_code)
- ASR input with fixed 4s recording (default) or text input (`--no-asr`)
- TTS voice output via Piper-TTS
- Reachy Mini robot control
- Multi-step task execution with tool results
- Headless mode support (`--gui-backend none`)

---

## 📋 Requirements

- **OS**: Ubuntu 22.04+ / Linux
- **Hardware**: AMD Ryzen AI or x86_64 platform
- **Robot**: Pollen Robotics Reachy Mini (with built-in camera)
- **Ollama**: For Chat and Guard modes (local or remote)
- **Hermes Agent**: For Agent mode (local hermes-agent with API server enabled)

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

### Run Guard Mode

#### Requirements

- [Ollama](https://ollama.com/) must be running locally (or on a reachable host)
- A vision-capable model must be available, e.g. `gemma4:12b` (default) or `gemma4:e2b`

#### Pull the default model

```bash
ollama pull gemma4:12b
```

#### With Reachy Mini Robot

```bash
# Start the Reachy daemon first (if using physical robot without media streaming)
.venv/bin/python -m reachy_mini.daemon.app.main --no-media

# Then start Guard mode
python main.py --guard --camera-source reachy --debug
```

#### With Local Webcam

```bash
python main.py --guard --camera-source webcam --camera-index 0
```

#### Common Guard Options

```bash
# Use a different model or analysis interval
python main.py --guard --guard-model gemma4:e2b --guard-interval 5

# Disable head scanning
python main.py --guard --no-scan

# Use a remote Ollama instance
OLLAMA_HOST=http://192.168.1.100:11434 python main.py --guard
```

Alert screenshots are saved to `~/Pictures/ReachyGuard/` by default.

### Run Chat Mode

#### Requirements

- Ollama running locally or reachable
- A text-capable model, e.g. `qwen3.5:0.8b` (default)

```bash
ollama pull qwen3.5:0.8b
```

#### Voice chat (default, ASR enabled)

```bash
python main.py --chat --camera-source reachy
```

Just speak after the 4s recording prompt. The robot will reply with voice and emotion actions.

#### Text chat

Useful for testing without a microphone:

```bash
python main.py --chat --no-asr --gui-backend none
```

Note: `--camera-source` is not required in chat mode — the robot connects directly via the Reachy daemon.

#### Common Chat Options

```bash
# Use a different model or gentle actions
python main.py --chat --ollama-model qwen3:0.6b --gentle

# Set ASR language explicitly
python main.py --chat --asr-language zh
```

### Run Agent Mode

#### Requirements

- Hermes Agent running locally with API server enabled
- API server must be enabled in `~/.hermes/config.yaml`

#### Setup Hermes API Server

1. Enable API server in `~/.hermes/config.yaml`:
```yaml
gateway:
  platforms:
    api_server:
      enabled: true
      host: 127.0.0.1
      port: 8642
      key: alehe
```

2. Set environment variables in `~/.hermes/.env`:
```
API_SERVER_ENABLED=true
API_SERVER_KEY=alehe
```

3. Restart the gateway:
```bash
hermes gateway restart
```

4. Verify API server is running:
```bash
curl -H "Authorization: Bearer alehe" http://localhost:8642/v1/models
```

#### Voice Agent (default, ASR enabled)

```bash
python main.py --agent
```

#### Text Agent

Useful for testing without a microphone:

```bash
python main.py --agent --no-asr --gui-backend none
```

#### Common Agent Options

```bash
# Use a different Hermes URL
python main.py --agent --hermes-url http://localhost:8642

# Specify tools file (future feature)
python main.py --agent --tools config/tools.yaml
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

### Global Options

```
python main.py --cheese| --guard [OPTIONS]

Global Options:
  --camera-source {reachy,webcam}    Camera source (default: reachy)
  --camera-index INT                 Webcam index for webcam mode (default: 0)
  --preview-width INT                Preview width (default: 640)
  --preview-height INT               Preview height (default: 480)
  --preview-fps FLOAT                Preview FPS (default: 20.0)
  --gui-backend {auto,cv2,none}      GUI backend (default: auto)
  --debug                            Enable debug output
```

### ASR Options

```
  --asr-model {tiny,base,small,medium,large}  ASR model size (default: base)
  --vad-silence FLOAT                         VAD silence threshold seconds (default: 0.7)
  --vad-aggressive {0,1,2,3}                  VAD aggressiveness (default: 1)
```

### TTS Options

```
  --piper-model PATH    Piper TTS model path
  --piper-config PATH   Piper TTS config path
  --speaker INT         Speaker ID (default: 0)
```

### Chat Mode Options

```
  --ollama-url URL        Ollama API URL (default: http://localhost:11434)
  --ollama-model MODEL    Ollama model name (default: qwen3.5:0.8b)
  --history-size N        Conversation history size (default: 5)
  --no-asr                Disable ASR; use text input
  --asr-language LANG     ASR language (default: auto)
  --gentle                Enable gentle emotion actions
```

### Guard Mode Options

```
  --guard-model MODEL      Ollama VLM model (default: gemma4:12b)
  --guard-interval SECONDS Seconds between analyses (default: 8.0)
  --scan / --no-scan       Enable/disable head scanning (default: on)
  --scan-range RADIANS     Head scan range (default: 0.6)
```

### Agent Mode Options

```
  --hermes-url URL        Hermes API URL (default: http://localhost:8642)
  --hermes-model MODEL    Hermes model name (default: hermes3-llama-3.1-8b)
  --tools PATH            Tools configuration file (future feature)
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
│   ├── guard/              # Security monitoring mode
│   ├── chat/               # Voice/text chat with LLM + emotion actions
│   └── agent/              # AI agent with Hermes API + tool calling
├── utils/                   # Shared utilities
│   ├── asr.py              # Speech recognition
│   ├── tts_engine.py       # Text-to-speech
│   ├── camera_tuning.py    # V4L2 camera tuning CLI
│   └── camera_tuning_gui.py# V4L2 camera tuning GUI
├── vision/                  # Computer vision
│   └── face_tracker.py     # Face detection
├── docs/                    # Documentation
│   ├── face_tracking.md    # Camera and face tracking guide
│   └── TO-DO.md            # Future enhancements
├── requirements/            # Per-mode dependencies
└── models/                  # Voice models + vision models
```

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|------------|
| **ASR** | faster-whisper (CPU) |
| **VAD** | webrtcvad |
| **TTS** | Piper-TTS (ONNX) |
| **Face Detection** | MediaPipe Face Detection |
| **GUI** | OpenCV (`cv2`) / headless (`none`) |
| **Robot Control** | reachy-mini SDK |
| **Guard VLM** | Ollama (gemma4:12b / gemma4:e2b) |
| **Chat LLM** | Ollama (qwen3.5:0.8b / qwen3:0.6b) |
| **Agent LLM** | Hermes API (OpenAI-compatible, local hermes-agent) |

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

### Ollama Not Responding (Guard Mode)

If Guard mode starts but the model never loads:

```bash
# Verify Ollama is running
ollama ps
curl http://localhost:11434/api/tags

# Check that the model exists
ollama list

# Pull the default model if missing
ollama pull gemma4:12b
```

If you have `HTTP_PROXY` / `HTTPS_PROXY` environment variables set, the app will bypass them for local Ollama requests. If you still see 502 errors, set:

```bash
export NO_PROXY=localhost,127.0.0.1
```

### No Guard Alerts

1. Check the terminal for `🧠 Analysis:` output. If none appears, the analysis thread may not have started — run with `--debug`.
2. The VLM may return "OK" for normal scenes. Place a person or object in front of the camera and wait one analysis interval.
3. Verify screenshots directory exists: `ls ~/Pictures/ReachyGuard`

### Hermes API Not Responding (Agent Mode)

If Agent mode starts but cannot reach Hermes API:

```bash
# Check if Hermes gateway is running
ps aux | grep hermes

# Check if API server is listening on port 8642
ss -tlnp | grep 8642

# Test API server health
curl http://localhost:8642/health

# Test with authentication
curl -H "Authorization: Bearer alehe" http://localhost:8642/v1/models
```

If the API server is not running:

```bash
# Check if API server is enabled in config
grep -A 10 "api_server" ~/.hermes/config.yaml

# Restart the gateway
hermes gateway restart
```

If you have `HTTP_PROXY` / `HTTPS_PROXY` environment variables set, the app will bypass them for local Hermes requests.

---

## 📄 License

MIT License - See [LICENSE](./LICENSE)

---

## 🙏 Acknowledgements

- [Pollen Robotics](https://www.pollen-robotics.com/) - Reachy Mini robot
- [Piper TTS](https://github.com/rhasspy/piper) - Offline text-to-speech
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - Speech recognition
- [MediaPipe](https://mediapipe.dev/) - Face detection
