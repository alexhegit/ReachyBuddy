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

## 📋 Feature Checklist (Already Implemented)

- [x] Piper TTS offline speech synthesis
- [x] faster-whisper ASR (with model selection)
- [x] VAD dynamic recording (`--vad-silence`, `--vad-aggressive`, `--no-vad`)
- [x] Conversation history management (`--history-size`, `--no-history`, `clear` command)
- [x] Performance timing statistics (`--debug` mode)
- [x] Robot emotion control integration
- [x] Asynchronous architecture with concurrency support

---

*Last updated: 2026-03-31*
