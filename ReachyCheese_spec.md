# ReachyCheese Design Specification

## 1. Project Goals

- A fully offline voice-interactive photo app for Reachy Mini.
- Designed to evolve into an independently publishable application.

## 2. Core User Flow

1. Standby and listen for wake word `"Reachy"`.
2. Upon wake-up, enter face tracking: align the **largest face** to center.
3. Display real-time GUI preview (face bounding box, center crosshair, status).
4. Wait for capture phrase (`"cheese"`, `"take photo"`, `"take picture"`).
5. Voice prompt and countdown:
   - `"Look at me. Hold still... Ready? One, two, three, cheese!"`
6. Capture photo and save to `~/Pictures/ReachyMiniPhoto/`, then return to standby.

## 3. State Machine

- `Sleep`: Standby, listening for wake word only
- `Tracking`: Continuous face tracking and alignment
- `Armed`: Face aligned, waiting for capture command
- `Countdown`: Voice prompt + countdown
- `Capture`: Photo capture and save
- `SaveAndConfirm`: Announce result, return to `Sleep`

## 4. Offline Tech Stack

- **ASR**: `faster-whisper` + VAD (command recognition)
- **TTS**: `Piper` (offline neural TTS)
- **Face Detection**: MediaPipe Face Detection
- **Wake Word**: Heuristic matching on ASR results

## 5. GUI Design

- **Primary**: Dear PyGui; fallback to OpenCV window GUI (with mouse button interaction)
- Real-time preview downsampled to `640x480` for smooth performance
- Overlays:
  - Largest face detection box
  - Center crosshair
  - Current state (Sleep/Tracking/Armed/Countdown/Capture)
  - Countdown timer
- Mouse interaction (manual capture, cancel countdown, retake, etc.)

## 6. Face Tracking Strategy (Head + Body)

**"Head-first, body compensation"** dual-loop control:

1. Detect face and select largest target each frame.
2. Calculate center offset `dx/dy` (target center vs. image center).
3. Apply EMA smoothing and dead-zone thresholding to reduce jitter.
4. **Inner head loop**: High-frequency small-step head yaw/pitch adjustment.
5. **Outer body loop**: When head approaches limit or large offset persists, low-frequency small-step body_yaw compensation.
6. Only enter countdown when "stable alignment" satisfies N consecutive frames.

## 7. Alignment and Capture Criteria

- **Alignment success conditions**:
  - `|dx|`, `|dy|` consistently below threshold
  - Face bounding box area reaches minimum threshold (reasonable distance)
  - Continuous stable frames meet threshold
- During countdown, maintain low-frequency fine-tuning to prevent drift.
- Grab current frame at the `"cheese"` moment for saving.

## 8. Photo Storage Specification

- Save directory: `~/Pictures/ReachyMiniPhoto/`
- File naming: `IMG_YYYYMMDD_HHMMSS.jpg`
- Preview at 640x480, save original frame resolution if available.

## 9. Error Handling

- No face / low confidence: Prompt user to look at camera, continue tracking.
- Face lost or large deviation during countdown: Cancel countdown, return to `Tracking/Armed`.
- Save failure: Explicit error message, no silent failure.

## 10. Architecture (Independent App)

Prefer lightweight component-based architecture over extending `emo_v*` monolithic structure.

Recommended module boundaries:
- `state_machine`
- `vision_tracker`
- `voice_io` (wake/asr/tts)
- `camera_preview_gui`
- `photo_storage`
- `reachy_controller`

Complete MVP main flow first, then iterate.
