"""Cheese mode application - voice photo capture."""

from __future__ import annotations

import difflib
import os
import queue
import re
import threading
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import cv2
import numpy as np

from core.base_app import BaseModeApp
from core.runtime import create_runtime
from utils.asr import FasterWhisperASREngine
from utils.tts_engine import PiperTTSEngine

from .config import CheeseConfig


class RCState(str, Enum):
    """Cheese mode states."""
    SLEEP = "sleep"
    TRACKING = "tracking"
    ARMED = "armed"
    COUNTDOWN = "countdown"


class FaceAligner:
    """Face tracking and alignment controller."""
    
    def __init__(self, config: CheeseConfig, debug: bool = False):
        from vision.face_tracker import FaceTracker
        
        self._debug = debug
        self._tracker = FaceTracker(
            smooth_factor=config.smooth_factor,
            multi_face_strategy="largest",
            min_detection_confidence=config.min_detection_confidence,
        )
        self._deadzone_x = config.deadzone_x
        self._deadzone_y = config.deadzone_y
        self._stable_needed = config.stable_needed
        
        # Internal state
        self._stable_frames = 0
        self._ema_dx = 0.0
        self._ema_dy = 0.0
        self._alpha = 0.30
        self._last_track_at = 0.0
        self._locked = False
        self._body_yaw = 0.0
        self._last_cmd_center: Optional[Tuple[int, int]] = None
    
    def reset(self) -> None:
        """Reset aligner state."""
        self._stable_frames = 0
        self._ema_dx = 0.0
        self._ema_dy = 0.0
        self._locked = False
        self._body_yaw = 0.0
        self._last_cmd_center = None
    
    def update(self, runtime, frame, soft: bool = False) -> dict:
        """Update face tracking."""
        bbox = self._tracker.detect(frame)
        
        # Debug: log first detection
        if bbox is not None and self._debug and not hasattr(self, '_first_detect_logged'):
            print(f"   👤 Face first detected: bbox={bbox}")
            self._first_detect_logged = True
        
        if bbox is None:
            self._stable_frames = 0
            self._locked = False
            return {
                "has_face": False,
                "aligned": False,
                "bbox": None,
                "center": None,
                "dx": 0.0,
                "dy": 0.0,
                "stable_frames": 0,
            }
        
        x, y, w, h = bbox
        frame_h, frame_w = frame.shape[:2]
        cx, cy = x + (w // 2), y + (h // 2)
        dx = float(cx - frame_w // 2)
        dy = float(cy - frame_h // 2)
        
        # EMA smoothing
        self._ema_dx = self._alpha * dx + (1 - self._alpha) * self._ema_dx
        self._ema_dy = self._alpha * dy + (1 - self._alpha) * self._ema_dy
        
        # Check alignment
        aligned_now = (abs(self._ema_dx) <= self._deadzone_x and 
                       abs(self._ema_dy) <= self._deadzone_y)
        self._stable_frames = self._stable_frames + 1 if aligned_now else 0
        aligned = self._stable_frames >= self._stable_needed
        
        if aligned:
            self._locked = True
        
        # Control robot (simplified)
        now = time.time()
        if now - self._last_track_at >= (0.28 if soft else 0.16):
            self._last_track_at = now
            if runtime and (abs(self._ema_dx) > 45 or abs(self._ema_dy) > 35):
                try:
                    target_x = frame_w // 2 + int(np.clip(self._ema_dx * 0.55, -95, 95))
                    target_y = frame_h // 2 + int(np.clip(self._ema_dy * 0.50, -70, 70))
                    should_send = True
                    if self._last_cmd_center:
                        dcmd_x = abs(target_x - self._last_cmd_center[0])
                        dcmd_y = abs(target_y - self._last_cmd_center[1])
                        if dcmd_x < 24 and dcmd_y < 24:
                            should_send = False
                    if should_send:
                        runtime.look_at_image(target_x, target_y, 
                                              duration=0.34 if soft else 0.24)
                        self._last_cmd_center = (target_x, target_y)
                except Exception:
                    pass
        
        return {
            "has_face": True,
            "aligned": aligned,
            "bbox": (x, y, w, h),
            "center": (cx, cy),
            "dx": self._ema_dx,
            "dy": self._ema_dy,
            "stable_frames": self._stable_frames,
        }


class VoiceIO:
    """Voice input/output handler."""
    
    def __init__(self, config: CheeseConfig):
        self.cfg = config
        self._speak_queue = queue.Queue()
        self._speak_running = True
        self._speak_thread = threading.Thread(target=self._speak_loop, daemon=True)
        self._tts = None
        
        if PiperTTSEngine is not None:
            self._tts = PiperTTSEngine(
                model_path=config.piper_model,
                config_path=config.piper_config,
                speaker_id=config.speaker_id,
                debug=config.debug,
            )
        
        self._asr = FasterWhisperASREngine(
            model_name=config.asr_model, 
            device="cpu"
        )
        self._speak_thread.start()
    
    def close(self) -> None:
        """Cleanup voice resources."""
        self._speak_running = False
        while not self._speak_queue.empty():
            try:
                self._speak_queue.get_nowait()
            except queue.Empty:
                break
        self._speak_queue.put("")
        self._speak_thread.join(timeout=2.0)
        
        if self._tts:
            try:
                self._tts.close()
            except Exception:
                pass
        if self._asr:
            try:
                self._asr.close()
            except Exception:
                pass
        
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
    
    def _speak_loop(self) -> None:
        """Background speech thread."""
        while self._speak_running:
            text = self._speak_queue.get()
            if not text:
                continue
            try:
                if self._tts and getattr(self._tts, "voice", None):
                    self._tts.speak_with_emotion(text, "neutral")
                else:
                    print(f"🔊 {text}")
            except Exception as exc:
                print(f"⚠️ TTS error: {exc}")
    
    def speak(self, text: str) -> None:
        """Queue speech."""
        if text.strip():
            self._speak_queue.put(text.strip())
    
    def listen(self) -> str:
        """Listen for voice input."""
        try:
            text = self._asr.transcribe_from_mic_vad(
                max_duration=4.5,
                silence_threshold=self.cfg.vad_silence,
                aggressiveness=self.cfg.vad_aggressive,
                trailing_buffer_ms=400,
                show_volume=False,
            )
            return (text or "").strip().lower()
        except Exception:
            return ""


class CheeseModeApp(BaseModeApp):
    """Cheese photo mode application."""
    
    def __init__(self, config: CheeseConfig):
        super().__init__(config)
        self.cfg = config
        self.state = RCState.SLEEP
        self.aligner: Optional[FaceAligner] = None
        self.voice: Optional[VoiceIO] = None
        
        # State machine
        self._armed_since = 0.0
        self._last_frame: Optional[np.ndarray] = None
        self._last_saved_path = ""
        
        # Countdown
        self._countdown_started_at = 0.0
        self._countdown_index = 0
        self._countdown_lines = [
            (0.0, "Look at me. Hold still... Ready?"),
            (0.8, "One"),
            (1.6, "Two"),
            (2.4, "Three"),
            (3.2, "Cheese"),
        ]
        
        # Voice listener
        self._asr_queue: "queue.Queue[str]" = queue.Queue()
        self._listener_running = False
        self._listener_thread: Optional[threading.Thread] = None
    
    def get_mode_name(self) -> str:
        return "cheese"
    
    def get_requirements(self) -> list:
        return [
            "numpy",
            "opencv-python",
            "sounddevice",
            "soundfile",
            "faster-whisper",
            "webrtcvad-wheels",
            "piper-tts",
            "mediapipe",
        ]
    
    def setup(self) -> None:
        """Initialize cheese mode components."""
        from .gui import CheeseGUI
        
        # Create save directory
        self.cfg.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.aligner = FaceAligner(self.cfg, debug=self.cfg.debug)
        self.voice = VoiceIO(self.cfg)
        self.gui = CheeseGUI(self.cfg)
        
        # Create runtime
        self.runtime = create_runtime(self.cfg.camera_source, self.cfg.camera_index)
        try:
            self.runtime.__enter__()
        except Exception as exc:
            if self.cfg.camera_source == "reachy":
                print(f"⚠️ Reachy init failed: {exc}")
                print("↪ Falling back to webcam")
                self.cfg.camera_source = "webcam"
                self.runtime = create_runtime("webcam", self.cfg.camera_index)
                self.runtime.__enter__()
            else:
                raise
        
        # Setup robot
        self.runtime.set_automatic_body_yaw(False)
        self.runtime.reset_head(duration=0.5)
        
        # Start voice listener
        self._start_listener()
    
    def _start_listener(self) -> None:
        """Start background ASR listener."""
        self._listener_running = True
        
        def loop():
            while self._listener_running:
                try:
                    heard = self.voice.listen()
                    if heard:
                        self._asr_queue.put(heard)
                        if self.cfg.debug:
                            print(f"🎤 Heard: {heard}")
                except Exception as exc:
                    if self.cfg.debug:
                        print(f"⚠️ ASR error: {exc}")
                    time.sleep(0.2)
        
        self._listener_thread = threading.Thread(target=loop, daemon=True)
        self._listener_thread.start()
    
    def cleanup(self) -> None:
        """Cleanup cheese mode resources."""
        self._listener_running = False
        if self._listener_thread:
            self._listener_thread.join(timeout=1.5)
        
        if self.runtime:
            self.runtime.__exit__(None, None, None)
            self.runtime = None
    
    def run_frame(self, frame: np.ndarray) -> bool:
        """Process one frame."""
        self._last_frame = frame.copy()
        
        # Process voice commands
        self._process_voice_commands()
        
        # Process GUI events
        if not self._process_gui_events():
            return False
        
        # State machine
        status = None
        
        if self.state == RCState.SLEEP:
            # Just wait for wake word (processed in _process_voice_commands)
            pass
        
        elif self.state == RCState.TRACKING:
            status = self.aligner.update(self.runtime, frame, soft=False)
            if status["aligned"]:
                self.state = RCState.ARMED
                self._armed_since = time.time()
                self.voice.speak("Look at me. Hold still.")
        
        elif self.state == RCState.ARMED:
            status = self.aligner.update(self.runtime, frame, soft=True)
            if not status["has_face"]:
                self.state = RCState.TRACKING
            elif time.time() - self._armed_since > self.cfg.command_timeout_s:
                self.voice.speak("Timeout. Back to sleep.")
                self._enter_sleep()
        
        elif self.state == RCState.COUNTDOWN:
            status = self.aligner.update(self.runtime, frame, soft=True)
            if not self._update_countdown(status):
                return True  # Continue
        
        # Draw GUI
        if self.gui:
            self.gui.draw(frame, self.state, status, self._last_saved_path)
        
        return True
    
    def _process_voice_commands(self) -> None:
        """Process queued voice commands."""
        while not self._asr_queue.empty():
            try:
                text = self._asr_queue.get_nowait().lower()
            except queue.Empty:
                break
            
            if self.cfg.debug:
                print(f"📝 Processing: {text}")
            
            # Wake word detection
            if self.state == RCState.SLEEP:
                if self._is_wake_phrase(text):
                    self.voice.speak("Hi. I am awake.")
                    self.state = RCState.TRACKING
                    self.aligner.reset()
                    continue
            
            # Capture command detection
            if self.state == RCState.ARMED:
                if self._is_capture_phrase(text):
                    self._start_countdown()
                    continue
    
    def _process_gui_events(self) -> bool:
        """Process GUI events. Returns False to quit."""
        if not self.gui:
            return True
        
        events = self.gui.get_events()
        for event in events:
            if event == "quit":
                return False
            elif event == "manual_wake":
                self.voice.speak("Manual wake.")
                self.state = RCState.TRACKING
                self.aligner.reset()
            elif event == "manual_capture":
                if self.state in (RCState.TRACKING, RCState.ARMED):
                    self._start_countdown()
            elif event == "manual_sleep":
                self.voice.speak("Going to sleep.")
                self._enter_sleep()
        
        return True
    
    def _enter_sleep(self) -> None:
        """Enter sleep state."""
        self.state = RCState.SLEEP
        if self.aligner:
            self.aligner.reset()
    
    def _start_countdown(self) -> None:
        """Start photo countdown."""
        self.state = RCState.COUNTDOWN
        self._countdown_started_at = time.time()
        self._countdown_index = 0
    
    def _update_countdown(self, status: dict) -> bool:
        """Update countdown state. Returns True if countdown continues."""
        elapsed = time.time() - self._countdown_started_at
        
        # Check if face is still visible
        if not status or not status.get("has_face"):
            self.voice.speak("Please look at me. Countdown cancelled.")
            self.state = RCState.TRACKING
            return False
        
        # Speak countdown lines
        while self._countdown_index < len(self._countdown_lines):
            t_mark, line = self._countdown_lines[self._countdown_index]
            if elapsed < t_mark:
                break
            self.voice.speak(line)
            self._countdown_index += 1
        
        # Capture at end
        if elapsed >= 3.4:
            self._capture_photo()
            return False
        
        return True
    
    def _capture_photo(self) -> None:
        """Capture and save photo."""
        if self._last_frame is None:
            self.voice.speak("Sorry, cannot get camera frame.")
            self.state = RCState.TRACKING
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.cfg.save_dir / f"IMG_{timestamp}.jpg"
        
        try:
            cv2.imwrite(str(out_path), self._last_frame)
            self._last_saved_path = str(out_path)
            print(f"📸 Saved: {out_path}")
            self.voice.speak("Photo saved.")
        except Exception as e:
            print(f"❌ Failed to save: {e}")
            self.voice.speak("Failed to save photo.")
        
        self._enter_sleep()
    
    @staticmethod
    def _is_wake_phrase(text: str) -> bool:
        """Check if text contains wake phrase."""
        normalized = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        aliases = {"reachy", "ricky", "richie", "reaching"}
        return any(alias in normalized for alias in aliases)
    
    @staticmethod
    def _is_capture_phrase(text: str) -> bool:
        """Check if text contains capture phrase."""
        normalized = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        phrases = {"cheese", "take photo", "take picture", "photo", "picture"}
        return any(p in normalized for p in phrases)
