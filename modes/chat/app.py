"""Chat mode - voice/text conversation with Ollama LLM and emotion actions."""

from __future__ import annotations

import base64
import queue
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import requests

from core.base_app import BaseModeApp
from core.runtime import create_runtime, find_reachy_camera

from .config import ChatConfig
from .emotion_controller import ChatEmotionController
from .gui import ChatGUI


class ChatModeApp(BaseModeApp):
    """Chat mode: ASR/text input -> Ollama LLM -> TTS + robot emotion actions."""

    def __init__(self, config: ChatConfig):
        super().__init__(config)
        self.cfg: ChatConfig = config

        # Conversation state
        self._history: List[dict] = []
        self._message_queue: "queue.Queue[str]" = queue.Queue()
        self._status = "Initializing..."
        self._subtitle = ""
        self._last_response = ""

        # Threads
        self._asr_thread: Optional[threading.Thread] = None
        self._thinking_thread: Optional[threading.Thread] = None
        self._speak_thread: Optional[threading.Thread] = None

        # Runtime tracking
        self._robot_runtime = None
        self._emotion_controller: Optional[ChatEmotionController] = None

    # ── BaseModeApp interface ──────────────────────────────────────

    def get_mode_name(self) -> str:
        return "chat"

    def get_requirements(self) -> list:
        return ["numpy", "opencv-python", "requests", "sounddevice", "soundfile"]

    def setup(self) -> None:
        print("💬 Initializing Chat mode...")

        # Camera/runtime
        device = find_reachy_camera()
        self.runtime = create_runtime(
            camera_source=self.cfg.camera_source,
            camera_index=self.cfg.camera_index,
            reachy_device_path=device,
            reachy_host=self.cfg.reachy_host,
            reachy_port=self.cfg.reachy_port,
        )

        self._reachy_connected = False
        self._camera_runtime = None

        if self.cfg.camera_source == "reachy":
            try:
                self.runtime.__enter__()
                self._reachy_connected = True
                self._robot_runtime = self.runtime
                print("✅ Reachy daemon connected (robot control available)")
            except Exception as e:
                print(f"⚠️ Reachy init failed: {e}")
                print("   ↪ Falling back to webcam-only mode")
                self.runtime = create_runtime("webcam", self.cfg.camera_index, reachy_device_path=device)
                self.runtime.__enter__()

            # If Reachy is connected but media is unavailable, open camera directly
            if self._reachy_connected:
                try:
                    frame = self.runtime.get_frame()
                except Exception:
                    frame = None
                if frame is None:
                    cam_dev = find_reachy_camera()
                    if cam_dev:
                        print(f"ℹ️ Reachy media unavailable; using camera device {cam_dev} for frames")
                        camera_rt = create_runtime("webcam", self.cfg.camera_index, reachy_device_path=cam_dev)
                        camera_rt.__enter__()
                        self._camera_runtime = camera_rt
                        self.runtime = camera_rt
                        print("✅ Camera runtime initialized for frames; robot control kept")
                    else:
                        print("⚠️ No camera device found; frames may be unavailable")
        else:
            try:
                self.runtime.__enter__()
            except Exception as e:
                print(f"❌ Webcam init failed: {e}")
                raise

        # TTS
        from utils.tts_engine import PiperTTSEngine
        self.voice = PiperTTSEngine(
            model_path=self.cfg.piper_model,
            config_path=self.cfg.piper_config,
            speaker_id=self.cfg.speaker_id,
        )
        self.voice.speak_with_emotion("Chat mode activated.")

        # Emotion controller (needs robot object)
        robot_obj = getattr(self._robot_runtime, "_reachy", None) if self._robot_runtime else None
        if robot_obj:
            self._emotion_controller = ChatEmotionController(
                robot_obj,
                gentle_mode=self.cfg.gentle_mode,
                debug=self.cfg.debug,
            )
            print("🎭 Emotion controller ready")
        else:
            print("ℹ️ No robot object available; emotion actions disabled")

        # ASR
        if self.cfg.use_asr:
            self._start_asr()
        else:
            self._start_text_input()

        # GUI
        self.gui = ChatGUI(self.cfg)

        print(f"🤖 Chat running — Ollama: {self.cfg.ollama_url} model: {self.cfg.ollama_model}")
        self._verify_ollama()

    def run_frame(self, frame: np.ndarray) -> bool:
        # Process incoming user messages
        user_text = ""
        try:
            while True:
                user_text = self._message_queue.get_nowait()
        except queue.Empty:
            pass

        if user_text:
            self._subtitle = f"You: {user_text[:50]}"
            self._status = "Thinking..."
            self._respond(user_text)

        # Render GUI
        if self.gui and self.gui.available:
            self.gui.draw(frame, self._status, self._subtitle)

        return True

    def cleanup(self) -> None:
        self._stop_thinking()
        if self._speak_thread and self._speak_thread.is_alive():
            self._speak_thread.join(timeout=2.0)
        if self._asr_thread and self._asr_thread.is_alive():
            self._asr_thread.join(timeout=2.0)
        if self._camera_runtime:
            self._camera_runtime.__exit__(None, None, None)
        if self.runtime:
            self.runtime.__exit__(None, None, None)
        if self.gui:
            self.gui.close()
        if self.voice:
            self.voice.close()

    # ── Internal helpers ───────────────────────────────────────────

    def _verify_ollama(self):
        proxies = {"http": None, "https": None}
        try:
            resp = requests.get(f"{self.cfg.ollama_url}/api/tags", timeout=10, proxies=proxies)
            resp.raise_for_status()
            models = {m.get("name") for m in resp.json().get("models", [])}
            if self.cfg.ollama_model not in models:
                print(f"   ⚠️ Model '{self.cfg.ollama_model}' not found. Run: ollama pull {self.cfg.ollama_model}")
            else:
                print(f"   ✅ Ollama model '{self.cfg.ollama_model}' is available")
        except Exception as e:
            print(f"   ⚠️ Cannot reach Ollama at {self.cfg.ollama_url}: {e}")

    def _start_asr(self):
        """Start background ASR listener."""
        try:
            from utils.asr import FasterWhisperASREngine
            self._asr_engine = FasterWhisperASREngine(
                model_name=self.cfg.asr_model,
                device=self.cfg.asr_device,
            )
        except Exception as e:
            print(f"❌ ASR init failed: {e}")
            print("   ↪ Falling back to text input mode")
            self._start_text_input()
            return

        self._asr_thread = threading.Thread(target=self._asr_loop, daemon=True)
        self._asr_thread.start()
        self._status = "🎙️ Listening..."
        print("🎙️ ASR listener started")

    def _asr_loop(self):
        """Continuously listen and queue transcriptions."""
        while getattr(self, '_running', True):
            try:
                text = self._asr_engine.transcribe_from_mic_vad(
                    max_duration=10.0,
                    silence_threshold=1.2,
                    language=self.cfg.asr_language,
                )
                if text and text.strip():
                    print(f"   🎤 ASR: {text.strip()}")
                    self._message_queue.put(text.strip())
            except Exception as e:
                if self.cfg.debug:
                    print(f"   ⚠️ ASR error: {e}")
                time.sleep(0.5)

    def _start_text_input(self):
        """Start background stdin reader for text mode."""
        self._text_thread = threading.Thread(target=self._text_loop, daemon=True)
        self._text_thread.start()
        self._status = "💬 Text input mode"
        print("💬 Text input mode: type your message and press Enter")

    def _text_loop(self):
        """Read lines from stdin and queue them."""
        import sys
        while getattr(self, '_running', True):
            try:
                line = sys.stdin.readline()
                if not line:
                    time.sleep(0.5)
                    continue
                line = line.strip()
                if line:
                    self._message_queue.put(line)
            except Exception as e:
                if self.cfg.debug:
                    print(f"   ⚠️ Text input error: {e}")
                time.sleep(0.5)

    def _respond(self, user_text: str):
        """Generate response, update history, speak with actions."""
        self._history.append({"role": "user", "content": user_text})
        self._trim_history()

        self._start_thinking()
        response = self._call_ollama(user_text)
        self._stop_thinking()

        if not response:
            self._status = "⚠️ No response from Ollama"
            return

        self._last_response = response
        self._history.append({"role": "assistant", "content": response})
        self._subtitle = f"Robot: {response[:50]}"
        self._status = "Speaking..."

        # Run TTS + actions in background so GUI stays responsive
        self._speak_thread = threading.Thread(
            target=self._speak_response,
            args=(response,),
            daemon=True,
        )
        self._speak_thread.start()

    def _call_ollama(self, user_text: str) -> Optional[str]:
        """Send conversation to Ollama and return response text."""
        proxies = {"http": None, "https": None}
        messages = [{"role": "system", "content": self.cfg.system_prompt}]
        messages.extend(self._history)

        try:
            resp = requests.post(
                f"{self.cfg.ollama_url}/api/chat",
                json={
                    "model": self.cfg.ollama_model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": self.cfg.temperature,
                        "num_predict": self.cfg.max_tokens,
                    },
                },
                timeout=120,
                proxies=proxies,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "").strip()
            thinking = data.get("message", {}).get("thinking", "").strip()
            return content or thinking or None
        except Exception as e:
            print(f"   ⚠️ Ollama error: {e}")
            return None

    def _speak_response(self, text: str):
        """Analyze emotion and speak with robot actions."""
        emotion, intensity, level = "neutral", "medium", 0.5
        if self._emotion_controller:
            emotion, intensity, level = self._emotion_controller.analyze(text)
            # Immediate reaction move before speaking
            self._emotion_controller.react(text, emotion, intensity)

        def speak_fn(t):
            self.voice.speak_with_emotion(t)

        if self._emotion_controller:
            self._emotion_controller.speak_with_actions(speak_fn, text, emotion, intensity)
        else:
            speak_fn(text)

        self._status = "🎙️ Listening..." if self.cfg.use_asr else "💬 Text input mode"

    def _trim_history(self):
        """Keep last N exchanges."""
        max_len = self.cfg.max_history * 2
        if len(self._history) > max_len:
            self._history = self._history[-max_len:]

    def _start_thinking(self):
        """Start background thinking animation on robot head."""
        if not self._robot_runtime:
            return
        self._stop_thinking()
        self._thinking_event = threading.Event()
        self._thinking_thread = threading.Thread(
            target=self._thinking_loop,
            args=(self._thinking_event,),
            daemon=True,
        )
        self._thinking_thread.start()

    def _stop_thinking(self):
        if getattr(self, '_thinking_event', None):
            self._thinking_event.set()
        if self._thinking_thread and self._thinking_thread.is_alive():
            self._thinking_thread.join(timeout=1.0)
        self._thinking_thread = None

    def _thinking_loop(self, stop_event: threading.Event):
        """Gentle head roll while waiting for LLM response."""
        import math
        start = time.time()
        while not stop_event.is_set() and time.time() - start < self.cfg.thinking_duration:
            angle = math.sin((time.time() - start) * 3) * 0.1
            try:
                self._robot_runtime.move_head(pan=angle, duration=0.3)
            except Exception:
                pass
            time.sleep(0.3)
        try:
            self._robot_runtime.move_head(pan=0.0, duration=0.5)
        except Exception:
            pass
