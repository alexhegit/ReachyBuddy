# ReachyBuddy — Agent Guide

> This file is intended for AI coding agents. Expect the reader to know nothing about the project.

---

## Project Overview

**ReachyBuddy** is a multi-mode robot application for the **Pollen Robotics Reachy Mini** robot. It provides pluggable operating modes with voice interaction, computer vision, and robot control.

| Mode | Status | Description |
|------|--------|-------------|
| **Cheese** | ✅ Implemented | Voice-interactive photo capture with automatic face tracking and alignment |
| **Guard** | ✅ Implemented | Multi-modal security monitoring with Ollama VLM analysis |
| **Chat** | ✅ Implemented | Voice assistant chat with Ollama LLM and emotion actions |
| **Agent** | ✅ Implemented | AI agent with Hermes API tool calling and robot control |

The core workflow for Cheese mode:
```
[Sleep] --"Reachy"--> [Tracking] --aligned--> [Armed] --"cheese"--> [Countdown] --> [Capture]
```

**License:** Apache 2.0 (see `LICENSE`)

---

## Technology Stack

| Component | Technology | Version/Notes |
|-----------|------------|---------------|
| **Language** | Python 3.12+ | |
| **ASR** | faster-whisper (CPU) | Models: tiny, base, small, medium, large |
| **VAD** | webrtcvad-wheels | Configurable aggressiveness (0–3) |
| **TTS** | Piper-TTS (ONNX Runtime) | Offline, multi-speaker support |
| **Face Detection** | MediaPipe Face Detection | CPU-based, short-range model |
| **GUI** | OpenCV (cv2) only | `--gui-backend` accepts `cv2` or `none`; DearPyGui listed in deps but never imported |
| **Robot Control** | reachy-mini SDK | Optional (falls back to webcam) |
| **Camera Control** | v4l2-ctl (v4l-utils) | Hardware parameter tuning |
| **Guard VLM** | Ollama | gemma4:12b / gemma4:e2b |
| **Chat LLM** | Ollama | qwen3.5:0.8b / qwen3:0.6b |
| **Agent LLM** | Hermes API | OpenAI-compatible, local hermes-agent |

---

## Project Structure

```
ReachyBuddy/
├── main.py                  # CLI entry point, argument parsing, config builder
├── core/                    # Shared framework
│   ├── base_app.py         # BaseModeApp abstract class, ModeConfig dataclass
│   ├── runtime.py          # RobotRuntime, ReachyRuntime, WebcamRuntime, camera profile loading
│   └── event_bus.py        # Simple pub/sub event system (global singleton)
├── modes/                   # Mode implementations
│   ├── __init__.py         # Mode registry with auto-registration
│   ├── cheese/             # ✅ Fully implemented
│   │   ├── app.py          # CheeseModeApp, FaceAligner, VoiceIO, RCState enum
│   │   ├── config.py       # CheeseConfig dataclass
│   │   └── gui.py          # CheeseGUI (cv2/none backends)
│   ├── guard/              # ✅ Fully implemented
│   │   ├── app.py          # GuardModeApp, background VLM analysis, head scanning
│   │   ├── config.py       # GuardConfig dataclass
│   │   └── gui.py          # GuardGUI (cv2/none backends)
│   ├── chat/               # ✅ Fully implemented
│   │   ├── app.py          # ChatModeApp, blocking chat loop, Ollama integration
│   │   ├── config.py       # ChatConfig dataclass
│   │   ├── gui.py          # ChatGUI (cv2/none backends)
│   │   └── emotion_controller.py # EmotionController, EmotionAnalyzer, SpeakingActor
│   └── agent/              # ✅ Fully implemented
│       └── app.py          # AgentModeApp, Hermes API integration, tool calling
├── utils/                   # Shared utilities
│   ├── asr.py              # FasterWhisperASREngine with VAD recording
│   ├── tts_engine.py       # PiperTTSEngine wrapper
│   ├── camera_tuning.py    # CLI camera tuning (v4l2-ctl wrapper)
│   └── camera_tuning_gui.py # OpenCV GUI for real-time camera tuning
├── vision/                  # Computer vision
│   └── face_tracker.py     # MediaPipe FaceTracker with EMA smoothing
├── docs/
│   ├── face_tracking.md    # Detailed Chinese documentation on camera/tracking
│   └── TO-DO.md            # Future enhancements
├── requirements/            # Per-mode dependency files
│   ├── base.txt            # Core dependencies
│   ├── cheese.txt          # Inherits base.txt
│   ├── guard.txt           # Inherits base.txt + future transformers/torch
│   ├── chat.txt            # Inherits base.txt + aiohttp
│   └── agent.txt           # Inherits base.txt + aiohttp
├── models/                  # Voice model directory
│   └── MODEL_CARD          # Kathleen voice model metadata
└── assets/
    └── ReachyMiniChat.png  # Demo screenshot
```

---

## Build and Run Commands

**No formal build system** — pure Python project with pip-based installation.

### System Dependencies

```bash
sudo apt install -y python3 python3-venv python3-pip ffmpeg libsndfile1 portaudio19-dev espeak v4l-utils
```

### Python Dependencies

```bash
# Install default mode (cheese)
pip install -r requirements.txt   # Currently points to requirements/cheese.txt

# Or install per-mode:
pip install -r requirements/cheese.txt
pip install -r requirements/chat.txt
pip install -r requirements/agent.txt
pip install -r requirements/guard.txt
```

### Robot SDK (Optional)

```bash
pip install "reachy-mini[mujoco]"
```

### Running the Application

```bash
# Cheese mode with robot
python main.py --cheese --camera-source reachy

# Cheese mode with webcam (testing)
python main.py --cheese --camera-source webcam --camera-index 0

# Disable head/body tracking (useful for testing)
python main.py --cheese --camera-source webcam --no-track

# Connect to remote Reachy daemon
python main.py --cheese --camera-source reachy --reachy-host 192.168.1.100 --reachy-port 8000

# Camera tuning
python utils/camera_tuning_gui.py
```

**Startup scripts** (convenience): `start-cheese-reachy.sh`, `start-cheese.sh`, `start-daemon.sh`, `start-daemon-real.sh`
**Connection diagnostic**: `scripts/reachy_connect_check.py`

---

## Architecture

```
main.py (entry point)
    │
    ├── argparse → build_config() → mode + config
    │
    ├── modes/__init__.py (MODE_REGISTRY)
    │       ├── cheese/  → CheeseModeApp
    │       ├── guard/   → GuardModeApp
    │       ├── chat/    → ChatModeApp
    │       └── agent/   → AgentModeApp
    │
    └── ModeClass(config).run()
            │
            ├── core/base_app.py: BaseModeApp (abstract base)
            │       ├── check_requirements()
            │       ├── setup() → initialize runtime, voice, GUI
            │       └── run() → frame loop
            │
            ├── core/runtime.py: RobotRuntime abstraction
            │       ├── ReachyRuntime (physical robot)
            │       └── WebcamRuntime (local webcam, no-op movement)
            │
            ├── core/event_bus.py: EventBus (minimal, future use)
            │
            ├── utils/asr.py: FasterWhisperASREngine
            │       ├── transcribe_from_mic() (fixed duration)
            │       └── transcribe_from_mic_vad() (VAD-based)
            │
            ├── utils/tts_engine.py: PiperTTSEngine
            │       ├── speak_with_emotion()
            │       └── speak_with_interrupt()
            │
            ├── vision/face_tracker.py: FaceTracker (MediaPipe)
            │       └── detect() → bbox, get_face_center() → smoothed coords
            │
            └── modes/cheese/gui.py: CheeseGUI
                    ├── OpenCV backend (default)
                    └── none backend (headless)
```

**Key Design Patterns:**
- **Abstract Base Class**: `BaseModeApp` defines the lifecycle (`setup()` → `run_frame()` → `cleanup()`)
- **Factory Pattern**: `create_runtime()` returns `ReachyRuntime` or `WebcamRuntime`
- **Registry Pattern**: `modes/__init__.py` auto-registers all mode classes
- **Strategy Pattern**: GUI backends (`cv2`, `none`) and face selection strategies (`largest`, `center`, `leftmost`)

---

## Code Style Guidelines

- **Type hints**: Used throughout (`from __future__ import annotations`, `Optional`, `Tuple`, etc.)
- **Dataclasses**: Configuration objects (`ModeConfig`, `CheeseConfig`, `CameraProfile`)
- **Docstrings**: Google-style docstrings with Args/Returns
- **Error handling**: Graceful degradation with try/except; debug mode for stack traces
- **Threading**: Background threads for ASR listener and TTS speech queue
- **Cleanup**: Comprehensive cleanup in `_cleanup_base()` / `_cleanup_without_exit()` including `atexit` registration and `os._exit(0)` to prevent segfaults on exit

---

## Testing

**No test files found** — no `tests/`, `pytest.ini`, `tox.ini`, or CI/CD configurations.

**No CI/CD** — no GitHub Actions, pre-commit hooks, or automated checks.

**No linting/formatting configs** — no `.flake8`, `.black`, `.pre-commit-config.yaml`, `pyproject.toml`, `setup.py`, `setup.cfg`, or `Makefile`.

---

## Security Considerations

1. **Camera Tuning Safety**: Camera tuning tools **auto-detect Reachy cameras** and reject modifications to non-Reachy devices unless `--allow-any-camera` is used.
2. **Reachy Runtime Fallback**: Reachy runtime falls back to webcam if SDK initialization fails.
3. **Model Files**: `.gitignore` excludes `*.onnx` and `*.onnx.json`, but existing files were committed before the rule and remain tracked.

---

## Special Configuration & Setup Requirements

### Camera Source Selection

- `reachy`: Uses Reachy Mini SDK, enables robot movement
- `webcam`: Uses OpenCV VideoCapture, **movement is no-op**

### Camera Tuning (Critical for Reachy)

- Reachy's camera lacks hardware ISP → images are darker than webcams
- Use `utils/camera_tuning_gui.py` for real-time adjustment
- Profiles saved to `~/.config/reachy_mini/`
- Load with `--camera-profile <name>`

### Voice Models

- Default: `models/en-us-ryan-medium.onnx` (already tracked in repo despite `.gitignore`)
- Download alternatives from [Piper Voices](https://huggingface.co/rhasspy/piper-voices)

### GUI Backends

- `--gui-backend cv2` → OpenCV (default)
- `--gui-backend none` → headless mode

---

## Dependencies

**Base requirements (`requirements/base.txt`):**
```
numpy>=1.24.0
opencv-python>=4.8.0
soundfile>=0.12.1
sounddevice>=0.4.8
faster-whisper>=1.2.1
webrtcvad-wheels>=2.0.14
piper-tts>=1.4.0
mediapipe>=0.10.0
requests>=2.31.0
```

**Mode-specific additions:**
- **chat**: `aiohttp>=3.9.0`
- **agent**: `aiohttp>=3.9.0`
- **guard**: (future: `transformers`, `torch`, `accelerate`, `pillow`)

**System dependencies:** `ffmpeg`, `libsndfile1`, `portaudio19-dev`, `espeak`, `v4l-utils`

**Optional:** `reachy-mini[mujoco]` (for physical robot control)

---

## Notes for Agents

- **3 of 4 modes are placeholders** — only Cheese mode is fully functional.
- **No `pyproject.toml`, `setup.py`, or `setup.cfg`** — not installable as a package.
- **Virtual environment committed** — `.venv/` is present in the repo (should ideally be in `.gitignore` but the existing `.gitignore` already excludes `.venv/` — the committed `.venv/` may be an oversight).
- **Chinese documentation** (`docs/face_tracking.md`) suggests the developer may be Chinese-speaking; code comments and docstrings are in English.
- When adding new modes, follow the pattern in `modes/cheese/` and register them in `modes/__init__.py`.
- **Cheese mode fuzzy voice matching**: Wake phrases include "reachy", "ricky", "richie", "reaching"; capture phrases include "cheese", "take photo", etc. See `modes/cheese/app.py:742-756`.
- **Placeholder modes (guard/chat/agent) each have CLI args** defined in `main.py:194-238` even though the modes themselves are stubs.
- **FaceAligner control parameters** live in `CheeseConfig` (`modes/cheese/config.py:25-53`), not hardcoded. Tune `lock_hold_x`, `release_x`, `ema_alpha`, `body_step`, etc. without touching code.
- **Photo saving is async**: `_capture_photo` submits to a `ThreadPoolExecutor` so disk I/O doesn't block the frame loop.
- **Voice sentinel uses `None`**: `VoiceIO.close()` sends `None` (not `""`) to stop the speech thread — `""` was a bug (falsy, skipped by `if not text`).
