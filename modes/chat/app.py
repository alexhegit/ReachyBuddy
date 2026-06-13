"""Chat mode - voice/text conversation with Ollama LLM and emotion actions.

Rewritten to emulate emo_v7.py's blocking loop pattern instead of the frame-loop
architecture, since chat doesn't need continuous camera frames.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Optional

import requests

from core.base_app import BaseModeApp

from .config import ChatConfig
from .gui import ChatGUI


class ChatModeApp:
    """Chat mode: ASR/text -> Ollama LLM -> Piper TTS + emotion actions.

    Uses emo_v7.py's simple blocking loop pattern: record -> transcribe -> LLM -> speak.
    """

    def __init__(self, config: ChatConfig):
        self.cfg: ChatConfig = config
        self._running = False
        self._history: list[dict] = []
        self.voice = None
        self.asr_engine = None
        self._reachy = None
        self._emotion_controller = None
        self.gui = None

    # ── Public interface ──────────────────────────────────────────

    def get_mode_name(self) -> str:
        return "chat"

    def run(self) -> None:
        """Main entry point."""
        print("🚀 Starting CHAT mode...")
        self._running = True
        self._setup()
        self._chat_loop()
        self._cleanup()

    def stop(self) -> None:
        self._running = False

    # ── Setup ─────────────────────────────────────────────────────

    def _setup(self) -> None:
        print("💬 Initializing Chat mode...")

        # 1. Connect to Reachy Mini (same as emo_v7.py)
        self._reachy = self._connect_reachy()
        if self._reachy is None:
            print("⚠️ Reachy Mini not available; running in TTS-only mode")

        # 2. TTS
        if self._reachy:
            from utils.tts_engine import PiperTTSEngine
            self.voice = PiperTTSEngine(
                model_path=self.cfg.piper_model,
                config_path=self.cfg.piper_config,
                speaker_id=self.cfg.speaker_id,
            )
            print("🎙️ Piper TTS initialized")
        else:
            self.voice = None

        # 3. Emotion controller (needs ReachyMini object)
        if self._reachy:
            from .emotion_controller import ChatEmotionController
            self._emotion_controller = ChatEmotionController(
                self._reachy,
                gentle_mode=self.cfg.gentle_mode,
                debug=self.cfg.debug,
            )
            print("🎭 Emotion controller ready")

        # 4. ASR
        if self.cfg.use_asr:
            self._init_asr()

        # 5. GUI (camera preview, optional)
        try:
            if self._reachy:
                import cv2
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        self._camera = cv2.VideoCapture(0)
                    else:
                        self._camera = None
                else:
                    self._camera = None
            else:
                self._camera = None
        except Exception:
            self._camera = None

        if self.cfg.gui_backend != "none":
            self.gui = ChatGUI(self.cfg)
        else:
            self.gui = None

        # 6. Verify Ollama
        self._verify_ollama()

        print(f"🤖 Chat running — Ollama: {self.cfg.ollama_url} model: {self.cfg.ollama_model}")

    def _connect_reachy(self):
        """Connect to Reachy Mini (same approach as emo_v7.py)."""
        try:
            from reachy_mini import ReachyMini
            reachy = ReachyMini(media_backend="no_media")
            print("✅ Connected to Reachy Mini")
            self._reachy = reachy
            return reachy
        except Exception as e:
            print(f"⚠️ Reachy Mini connection failed: {e}")
            return None

    def _init_asr(self):
        """Initialize ASR engine (same model as emo_v7.py)."""
        try:
            from utils.asr import FasterWhisperASREngine
        except Exception:
            print("❌ faster-whisper not installed")
            print("   Install: pip install faster-whisper")
            self.cfg.use_asr = False
            return

        try:
            self.asr_engine = FasterWhisperASREngine(
                model_name=self.cfg.asr_model,
                device=self.cfg.asr_device,
            )
            print(f"🎤 ASR initialized (model: {self.cfg.asr_model})")
        except Exception as e:
            print(f"❌ ASR init failed: {e}")
            self.cfg.use_asr = False

    def _verify_ollama(self):
        """Quick check Ollama is reachable."""
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
            print(f"   ⚠️ Cannot reach Ollama: {e}")
            print("      If you have HTTP_PROXY set, try: export NO_PROXY=localhost,127.0.0.1")

    # ── Main chat loop ────────────────────────────────────────────

    def _chat_loop(self):
        """Blocking chat loop (same pattern as emo_v7.py)."""
        if self.cfg.use_asr:
            self._asr_chat_loop()
        else:
            self._text_chat_loop()

    def _asr_chat_loop(self):
        """ASR mode: record 4s -> transcribe -> LLM -> TTS (same as emo_v7.py)."""
        print("\n🎤 ASR mode: speak, wait 4s for recording. Press Ctrl-C to stop.")
        while self._running:
            try:
                print("\n⏺️ Recording (4s)...")
                transcription = self.asr_engine.transcribe_from_mic(4.0)
                if not transcription:
                    print("⚠️ No speech detected, try again")
                    continue

                print(f"📝 You: {transcription}")
                self._respond(transcription)

            except KeyboardInterrupt:
                print("\n👋 Exiting ASR chat")
                self._running = False
                break
            except Exception as e:
                print(f"⚠️ ASR error: {e}")
                time.sleep(1.0)

    def _text_chat_loop(self):
        """Text mode: type input (same as emo_v7.py)."""
        print("\n💬 Type your message (type 'quit' to exit)")
        while self._running:
            try:
                user_input = input("\n🧑 You: ").strip()
                if user_input.lower() in ('quit', 'exit', 'q'):
                    break
                if not user_input:
                    continue
                self._respond(user_input)
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\n👋 Interrupted")
                break
            except Exception as e:
                print(f"⚠️ Error: {e}")

    def _respond(self, user_text: str):
        """Get LLM response and speak with emotion actions."""
        print("\n🤖 Reachy: ", end="", flush=True)

        response = self._call_ollama(user_text)
        if not response:
            return

        # Emotion analysis
        emotion, intensity, level = "neutral", "medium", 0.5
        if self._emotion_controller:
            emotion, intensity, level = self._emotion_controller.analyze(response)
            if self.cfg.debug:
                print(f"\n🎭 Emotion: {emotion}, Intensity: {intensity}, Level: {level:.2f}")

        # Speak with emotion + actions
        if self._emotion_controller:
            self._emotion_controller.speak_with_actions(
                lambda t: self.voice.speak_with_emotion(t),
                response, emotion, intensity,
            )
        elif self.voice:
            self.voice.speak_with_emotion(response)

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Streaming response from Ollama (same as emo_v7.py)."""
        proxies = {"http": None, "https": None}

        # Build messages with history
        messages = [
            {"role": "system", "content": self.cfg.system_prompt},
        ]
        messages.extend(self._history)
        messages.append({"role": "user", "content": prompt})

        try:
            resp = requests.post(
                f"{self.cfg.ollama_url}/api/chat",
                json={
                    "model": self.cfg.ollama_model,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": self.cfg.temperature,
                        "num_predict": self.cfg.max_tokens,
                    },
                },
                stream=True,
                timeout=120,
                proxies=proxies,
            )

            if not resp.ok:
                print(f"\n⚠️ Ollama HTTP {resp.status_code}: {resp.text[:200]}")
                return None

            full_response = ""
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line.decode("utf-8"))
                    if chunk.get("error"):
                        print(f"\n⚠️ Ollama error: {chunk['error']}")
                        return None
                    content = chunk.get("message", {}).get("content") or ""
                    thinking = chunk.get("message", {}).get("thinking") or ""
                    if not content and thinking:
                        content = thinking
                    if content:
                        print(content, end="", flush=True)
                        full_response += content
                except Exception:
                    continue

            print()
            if not full_response:
                print("⚠️ Empty response from Ollama")
                return None

            # Update history
            self._history.append({"role": "user", "content": prompt})
            self._history.append({"role": "assistant", "content": full_response})
            if len(self._history) > self.cfg.max_history * 2:
                self._history = self._history[-(self.cfg.max_history * 2):]

            return full_response

        except requests.exceptions.ConnectionError:
            print(f"\n⚠️ Cannot reach Ollama at {self.cfg.ollama_url}")
            return None
        except Exception as e:
            print(f"\n⚠️ Ollama error: {e}")
            return None

    # ── Cleanup ───────────────────────────────────────────────────

    def _cleanup(self):
        print("\n🧹 Cleaning up...")
        if self.gui:
            try:
                self.gui.close()
            except Exception:
                pass
        if self._reachy:
            try:
                self._reachy.goto_sleep()
            except Exception:
                pass
        if self.asr_engine:
            try:
                self.asr_engine.close()
            except Exception:
                pass
        if self.voice:
            try:
                self.voice.close()
            except Exception:
                pass
        if self._camera:
            try:
                self._camera.release()
            except Exception:
                pass
        print("👋 Goodbye!")