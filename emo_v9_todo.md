# Reachy Mini Chat v9 - Development Todo List

> Current version: emo_v9.py (v9 branch)  
> Status: Development paused; features below are pending implementation

---

## 🎯 High Priority

### 1. Multilingual TTS Auto-switching ⭐
- **Description**: Automatically switch between Chinese/English Piper models based on ASR-detected language
- **Current state**: Requires manual `--piper-model` switching
- **Plan**:
  - Use faster-whisper language detection
  - Configure Chinese-English model path mappings
  - Automatically select the corresponding TTS model

### 2. Recording Volume Visualization ✅ Completed
- **Description**: Display a real-time ASCII volume bar during recording
- **Purpose**: Help users confirm the microphone is working and debug VAD issues
- **Implementation**:
  - Added `_calculate_rms()` to compute audio RMS values
  - Added `_draw_volume_bar()` to render ASCII bars `[████████░░░░] -45dB`
  - Supports both VAD and fixed-duration recording modes
  - Uses `\r` carriage return for single-line live updates

---

## 🔧 Medium Priority

### 3. Voice Barge-in
- **Description**: Allow users to interrupt the robot by speaking while TTS is playing
- **Technical notes**:
  - Need to monitor the microphone while audio is playing
  - Stop TTS immediately upon detecting human voice and switch to ASR mode

### 4. Configuration Persistence
- **Description**: Save common settings to `config.json`
- **Includes**:
  - Default model selections (LLM / ASR / TTS)
  - VAD parameters (silence / aggressiveness)
  - History toggle
  - Robot connection settings
- **Extension**: Persist conversation history to a file so context survives restarts

---

## 💡 Low Priority / Optimizations

### 5. TTS Speed Adjustment
- **CLI**: `--tts-speed` parameter (0.5–2.0)
- **Implementation**: Piper supports speed control parameters

### 6. Logging System
- **Description**: Replace `print` statements with Python `logging`
- **CLI**: `--log-level` (DEBUG / INFO / WARNING / ERROR)
- **Benefits**: Better debugging support and log file recording

### 7. Further VAD Optimization
- Evaluate Silero VAD as an alternative to webrtcvad
- Support streaming ASR (recognize while recording to reduce latency)

### 8. Interactive Configuration Wizard
- On first run, guide the user through:
  - Detecting available Piper models
  - Testing microphone and speaker
  - Selecting the default language

---

## 🐛 Known Issues (Pending Fixes)

| Issue | Status | Temporary Workaround |
|-------|--------|----------------------|
| VAD may cut off speech | ✅ Mitigated | Use `--vad-silence 1.5 --vad-aggressive 1` or `--no-vad` |
| Occasional Ctrl+C stutter | ✅ Fixed | Wrap blocking calls with `asyncio.to_thread()` |

---

## 👁️ Vision Capability Extension (New)

Architecture decision: **Plugin Mode (Option A)** - Incremental extension based on v9
- Keep `emo_v9.py` stable, create `emo_v9_vision.py` + `vision_controller.py`
- Vision is optional via `--vision` flag
- Event-driven: visual detection triggers existing v9 features

### Phase 1: Basic Vision (1-2 weeks)

| Feature | Description | Tech Stack | Integration Point | Status |
|---------|-------------|------------|-------------------|--------|
| **Face Tracking** ⭐ P0 | Robot head follows user's face in real-time | MediaPipe Face Detection + `look_at_image()` | `animation_thread()` - check face pos every 1.5s | 🚧 **In Progress** |
| **Motion Wake-up** | Auto-wake when person enters frame; sleep when idle | Frame differencing + face detection | Idle loop state machine | ⏳ Pending |

#### Face Tracking Implementation
```
vision/
├── __init__.py           # Module exports
├── face_tracker.py       # FaceTracker class with EMA smoothing
└── controller.py         # VisionController with event callbacks

emo_v9_vision.py          # Entry point extending ChatAppWithPiper
```

**Usage:**
```bash
python emo_v9_vision.py --vision --chat    # Enable face tracking
python emo_v9_vision.py --no-vision        # Pure v9 mode
```

**Features:**
- Real-time face detection via MediaPipe (CPU-friendly)
- EMA smoothing to prevent head jitter
- Periodic `look_at_image()` calls during speech (every 1.5s)
- Auto person enter/leave detection with callbacks
- Configurable FPS and detection timeout

### Phase 2: Interaction Enhancement (2-3 weeks)

| Feature | Description | Tech Stack | Integration Point |
|---------|-------------|------------|-------------------|
| **Gesture Barge-in** ⭐ P0 | Wave hand to interrupt TTS; gestures for pause/stop | MediaPipe Hands | Parallel gesture monitor during `speak_with_interrupt()` |
| **Emotion Awareness** | Detect user facial emotion to adjust robot response | DeepFace/FER | Inject into LLM system prompt via `analyze_user_emotion()` |

### Phase 3: Cognitive Capabilities (3-4 weeks)

| Feature | Description | Tech Stack | Integration Point |
|---------|-------------|------------|-------------------|
| **Visual QA** | "What do you see?" → Describe environment | LLaVA (local) or GPT-4V (API) | Prepend visual context to LLM prompt |
| **Point & Ask** | User points at object, robot identifies it | MediaPipe Hands (index finger) + YOLOv8 | Hand landmark → `look_at_image()` → object crop → detection |
| **Face Memory** | Recognize returning users by name | face_recognition library | Greeting logic in `start_chat_async()` |

### Implementation Structure

```
vision/
├── __init__.py
├── controller.py         # VisionController main class
├── face_tracker.py       # Face detection + tracking
├── gesture_recognizer.py # Hand gesture recognition
├── emotion_analyzer.py   # Facial expression analysis
├── object_detector.py    # YOLO object detection (optional)
└── visual_qa.py          # VLM visual question answering (optional)

emo_v9_vision.py          # v9 + vision integration entry point
```

### Priority Matrix

| Feature | Value | Difficulty | Priority | Est. Effort |
|---------|-------|------------|----------|-------------|
| Face Tracking | ⭐⭐⭐⭐⭐ | Low | P0 | 2 days |
| Gesture Barge-in | ⭐⭐⭐⭐⭐ | Medium | P0 | 3 days |
| Motion Wake-up | ⭐⭐⭐⭐ | Low | P1 | 1 day |
| Emotion Awareness | ⭐⭐⭐⭐ | Medium | P1 | 3 days |
| Visual QA | ⭐⭐⭐⭐⭐ | High | P2 | 5 days |
| Point & Ask | ⭐⭐⭐ | High | P2 | 4 days |
| Face Memory | ⭐⭐⭐ | Low | P2 | 2 days |

---

## 📋 Feature Checklist (Already Implemented)

- [x] Piper TTS offline speech synthesis
- [x] faster-whisper ASR (with model selection)
- [x] VAD dynamic recording (`--vad-silence`, `--vad-aggressive`, `--no-vad`)
- [x] Conversation history management (`--history-size`, `--no-history`, `clear` command)
- [x] Performance timing statistics (`--debug` mode)
- [x] Robot emotion control integration
- [x] Asynchronous architecture with concurrency support

---

*Last updated: 2026-04-05*
