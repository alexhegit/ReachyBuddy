# ReachyCheese 🧀🤖

**ReachyCheese** — A fully offline voice-interactive photo app for the Reachy Mini desktop robot.

Capture perfect robot-perspective photos with voice wake-up, face tracking, and automatic alignment.

![Demo](./assets/ReachyMiniChat.png)

---

## ✨ Features

- **🎙️ Voice Wake-up**: Say "Reachy" to wake the robot
- **👤 Face Tracking**: Automatically track the largest face and align to center
- **📸 Voice Capture**: Say "cheese", "take photo", or "take picture" to capture
- **⏱️ Smart Countdown**: Audio prompts "One, two, three, cheese!" before capture
- **🖼️ Real-time Preview**: GUI showing camera feed, face bounding box, status info
- **💾 Auto Save**: Photos saved to `~/Pictures/ReachyMiniPhoto/`
- **🔌 Fully Offline**: No network required, privacy protected

---

## 🔄 Workflow

```
[Sleep] --"Reachy"--> [Tracking] --aligned--> [Armed] --"cheese"--> [Countdown] --> [Capture]
```

1. **Sleep**: Standby, listening for wake word
2. **Tracking**: Track and align the largest face
3. **Armed**: Face aligned, waiting for capture command
4. **Countdown**: Audio countdown prompt
5. **Capture**: Take photo and save

---

## 📋 Requirements

- **OS**: Ubuntu 22.04+ / Linux
- **Hardware**: AMD Ryzen AI or x86_64 platform
- **Robot**: Pollen Robotics Reachy Mini (or webcam-only mode for testing)
- **Camera**: USB webcam or built-in laptop camera

---

## 🛠️ Installation

### 1. System Dependencies

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip ffmpeg libsndfile1 portaudio19-dev espeak
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

Download additional voices from [Piper Voices](https://huggingface.co/rhasspy/piper-voices).

---

## 🚀 Usage

### Start Reachy Mini Simulator (optional)

```bash
reachy-mini-daemon --sim
```

### Run ReachyCheese

#### With Reachy Mini Robot

```bash
python ReachyCheese.py --camera-source reachy
```

#### With Local Webcam (for testing)

```bash
python ReachyCheese.py --camera-source webcam --camera-index 0
```

#### Specify Voice Model

```bash
python ReachyCheese.py --piper-model models/zh_CN-huayan-medium.onnx
```

---

## 🎮 Voice Commands

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

Interface elements:
- Real-time camera preview
- Face detection bounding box (green)
- Center crosshair
- Current state display
- Countdown overlay
- Manual control buttons (Wake / Take Photo / Cancel / Sleep)

---

## ⚙️ Command Line Arguments

```
python ReachyCheese.py [OPTIONS]

Options:
  --preview-width INT       Preview window width (default: 640)
  --preview-height INT      Preview window height (default: 480)
  --preview-fps FLOAT       Preview frame rate (default: 20.0)
  --save-dir PATH           Photo save directory (default: ~/Pictures/ReachyMiniPhoto)
  --wake-word TEXT          Wake word (default: reachy)
  --asr-model {tiny,base,small,medium,large}  ASR model (default: base)
  --vad-silence FLOAT       VAD silence threshold in seconds (default: 0.7)
  --vad-aggressive {0,1,2,3} VAD aggressiveness (default: 1)
  --piper-model PATH        Piper TTS model path
  --piper-config PATH       Piper TTS config path
  --speaker INT             Speaker ID (default: 0)
  --camera-source {reachy,webcam} Camera source (default: reachy)
  --camera-index INT        Camera index (default: 0)
  --debug                   Enable debug output
```

---

## 📁 Project Structure

```
ReachyBuddy/
├── ReachyCheese.py          # Main application
├── ReachyCheese_spec.md     # Design specification
├── requirements.txt         # Python dependencies
├── models/                  # TTS voice models
│   ├── en-us-ryan-medium.onnx
│   └── ...
├── utils/
│   └── asr.py              # ASR speech recognition module
├── vision/
│   └── face_tracker.py     # Face tracking module
└── assets/                  # Image assets
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

# Test camera
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

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
