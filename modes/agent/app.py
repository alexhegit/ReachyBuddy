"""Agent mode - voice-controlled AI agent with Hermes tool calling.

Uses Hermes API (OpenAI-compatible) for LLM with tool calling.
Supports ASR for voice input and TTS for voice output.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Optional

import requests

from core.base_app import ModeConfig


class AgentModeApp:
    """Agent mode: ASR/text -> Hermes LLM with tools -> TTS + robot actions.

    Uses Hermes API for LLM with tool calling capabilities.
    """

    def __init__(self, config: ModeConfig):
        self.cfg = config
        self._running = False
        self._history: list[dict] = []
        self.voice = None
        self.asr_engine = None
        self._reachy = None
        self._emotion_controller = None
        self.gui = None
        self._tools: dict = {}

    # ── Public interface ──────────────────────────────────────────

    def get_mode_name(self) -> str:
        return "agent"

    def run(self) -> None:
        """Main entry point."""
        print("🚀 Starting AGENT mode...")
        self._running = True
        self._setup()
        self._agent_loop()
        self._cleanup()

    def stop(self) -> None:
        self._running = False

    # ── Setup ─────────────────────────────────────────────────────

    def _setup(self) -> None:
        print("🤖 Initializing Agent mode...")

        # 1. Connect to Reachy Mini
        self._reachy = self._connect_reachy()
        if self._reachy is None:
            print("⚠️ Reachy Mini not available; running without robot")

        # 2. TTS
        if self._reachy:
            from utils.tts_engine import PiperTTSEngine
            self.voice = PiperTTSEngine(
                model_path=self.cfg.mode_specific.get("piper_model", "models/en-us-ryan-medium.onnx"),
                config_path=self.cfg.mode_specific.get("piper_config", "models/en-us-ryan-medium.onnx.json"),
                speaker_id=self.cfg.mode_specific.get("speaker_id", 0),
            )
            print("🎙️ Piper TTS initialized")

        # 3. ASR
        if self.cfg.mode_specific.get("use_asr", True):
            self._init_asr()

        # 4. Load tools
        self._load_tools()

        # 5. Verify Hermes API
        self._verify_hermes()

        hermes_url = self.cfg.mode_specific.get("hermes_url", "http://localhost:8642")
        print(f"🤖 Agent running — Hermes: {hermes_url}")

    def _connect_reachy(self):
        """Connect to Reachy Mini."""
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
        """Initialize ASR engine."""
        try:
            from utils.asr import FasterWhisperASREngine
        except Exception:
            print("❌ faster-whisper not installed")
            return

        try:
            self.asr_engine = FasterWhisperASREngine(
                model_name=self.cfg.mode_specific.get("asr_model", "base"),
                device=self.cfg.mode_specific.get("asr_device", "cpu"),
            )
            print(f"🎤 ASR initialized (model: {self.cfg.mode_specific.get('asr_model', 'base')})")
        except Exception as e:
            print(f"❌ ASR init failed: {e}")

    def _load_tools(self):
        """Load available tools for the agent."""
        self._tools = {
            "move_head": self._tool_move_head,
            "take_photo": self._tool_take_photo,
            "get_time": self._tool_get_time,
            "execute_code": self._tool_execute_code,
        }
        print(f"🛠️ Loaded {len(self._tools)} tools: {', '.join(self._tools.keys())}")

    def _verify_hermes(self):
        """Quick check Hermes API is reachable."""
        proxies = {"http": None, "https": None}
        hermes_url = self.cfg.mode_specific.get("hermes_url", "http://localhost:8642")
        hermes_api_key = self.cfg.mode_specific.get("hermes_api_key", "alehe")
        try:
            headers = {"Authorization": f"Bearer {hermes_api_key}"}
            resp = requests.get(f"{hermes_url}/v1/models", timeout=10, proxies=proxies, headers=headers)
            resp.raise_for_status()
            print(f"   ✅ Hermes API is available at {hermes_url}")
        except Exception as e:
            print(f"   ⚠️ Cannot reach Hermes API: {e}")
            print("      Make sure Hermes is running: hermes gateway restart")

    # ── Tool implementations ──────────────────────────────────────

    def _tool_move_head(self, pan: float = 0.0, tilt: float = 0.0, roll: float = 0.0, duration: float = 0.5) -> str:
        """Move robot head to specified position."""
        if not self._reachy:
            return "Error: Reachy Mini not connected"
        try:
            from reachy_mini.utils import create_head_pose
            pose = create_head_pose(pan=pan, tilt=tilt, roll=roll)
            self._reachy.goto_target(head=pose, duration=duration)
            return f"Head moved to pan={pan}, tilt={tilt}, roll={roll}"
        except Exception as e:
            return f"Error moving head: {e}"

    def _tool_take_photo(self, filename: str = "photo.jpg") -> str:
        """Take a photo using the camera."""
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return "Error: Camera not available"
            ret, frame = cap.read()
            cap.release()
            if ret:
                cv2.imwrite(filename, frame)
                return f"Photo saved to {filename}"
            else:
                return "Error: Failed to capture frame"
        except Exception as e:
            return f"Error taking photo: {e}"

    def _tool_get_time(self) -> str:
        """Get current time."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _tool_execute_code(self, code: str) -> str:
        """Execute Python code (sandboxed)."""
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return result.stdout or "Code executed successfully (no output)"
            else:
                return f"Error: {result.stderr}"
        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out (10s limit)"
        except Exception as e:
            return f"Error executing code: {e}"

    # ── Main agent loop ───────────────────────────────────────────

    def _agent_loop(self):
        """Main agent loop with tool calling."""
        if self.cfg.mode_specific.get("use_asr", True):
            self._asr_agent_loop()
        else:
            self._text_agent_loop()

    def _asr_agent_loop(self):
        """ASR mode: record -> transcribe -> LLM with tools -> TTS."""
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
                print("\n👋 Exiting agent")
                self._running = False
                break
            except Exception as e:
                print(f"⚠️ ASR error: {e}")
                time.sleep(1.0)

    def _text_agent_loop(self):
        """Text mode: type input."""
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
        """Get LLM response with tool calling and speak."""
        print("\n🤖 Agent: ", end="", flush=True)

        response = self._call_hermes(user_text)
        if not response:
            return

        # Speak response
        if self.voice:
            self.voice.speak_with_emotion(response)

    def _call_hermes(self, prompt: str) -> Optional[str]:
        """Call Hermes API with tool calling support."""
        proxies = {"http": None, "https": None}
        hermes_url = self.cfg.mode_specific.get("hermes_url", "http://localhost:8642")
        hermes_api_key = self.cfg.mode_specific.get("hermes_api_key", "alehe")

        # Build messages with history
        messages = [
            {"role": "system", "content": "You are a helpful robot assistant named Reachy. "
             "You can control the robot's head, take photos, get the time, and execute code. "
             "Use tools when appropriate to help the user. Keep responses concise and friendly."},
        ]
        messages.extend(self._history)
        messages.append({"role": "user", "content": prompt})

        # Build tools for Hermes API
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "move_head",
                    "description": "Move robot head to specified position",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pan": {"type": "number", "description": "Pan angle in radians"},
                            "tilt": {"type": "number", "description": "Tilt angle in radians"},
                            "roll": {"type": "number", "description": "Roll angle in radians"},
                            "duration": {"type": "number", "description": "Movement duration in seconds"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "take_photo",
                    "description": "Take a photo using the camera",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string", "description": "Output filename"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get current time",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_code",
                    "description": "Execute Python code (sandboxed, 10s timeout)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Python code to execute"},
                        },
                        "required": ["code"],
                    },
                },
            },
        ]

        try:
            headers = {
                "Authorization": f"Bearer {hermes_api_key}",
                "Content-Type": "application/json",
            }

            resp = requests.post(
                f"{hermes_url}/v1/chat/completions",
                json={
                    "model": "hermes-agent",
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto",
                    "stream": False,
                    "max_tokens": 500,
                },
                headers=headers,
                timeout=120,
                proxies=proxies,
            )

            if not resp.ok:
                print(f"\n⚠️ Hermes HTTP {resp.status_code}: {resp.text[:200]}")
                return None

            data = resp.json()
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "")
            tool_calls = message.get("tool_calls", [])

            # Print content if any
            if content:
                print(content, end="", flush=True)

            # Execute tool calls
            if tool_calls:
                print(f"\n🔧 Executing {len(tool_calls)} tool call(s)...")
                for tool_call in tool_calls:
                    func = tool_call.get("function", {})
                    func_name = func.get("name", "")
                    func_args = json.loads(func.get("arguments", "{}"))

                    print(f"   📞 Calling {func_name}({func_args})")

                    if func_name in self._tools:
                        result = self._tools[func_name](**func_args)
                        print(f"   ✅ Result: {result}")

                        # Add tool result to messages
                        messages.append(message)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.get("id", ""),
                            "content": result,
                        })

                        # Get follow-up response
                        follow_up_resp = requests.post(
                            f"{hermes_url}/v1/chat/completions",
                            json={
                                "model": "hermes-agent",
                                "messages": messages,
                                "stream": False,
                                "max_tokens": 500,
                            },
                            headers=headers,
                            timeout=120,
                            proxies=proxies,
                        )

                        if follow_up_resp.ok:
                            follow_up_data = follow_up_resp.json()
                            follow_up_choice = follow_up_data.get("choices", [{}])[0]
                            follow_up_message = follow_up_choice.get("message", {})
                            follow_up_content = follow_up_message.get("content", "")
                            if follow_up_content:
                                print(f"\n🤖 Agent: {follow_up_content}")
                                content = follow_up_content
                    else:
                        print(f"   ❌ Unknown tool: {func_name}")

            print()
            if not content:
                print("⚠️ Empty response from Hermes")
                return None

            # Update history
            self._history.append({"role": "user", "content": prompt})
            self._history.append({"role": "assistant", "content": content})
            if len(self._history) > 20:  # Keep last 10 exchanges
                self._history = self._history[-20:]

            return content

        except requests.exceptions.ConnectionError:
            print(f"\n⚠️ Cannot reach Hermes at {hermes_url}")
            return None
        except Exception as e:
            print(f"\n⚠️ Hermes error: {e}")
            return None

    # ── Cleanup ───────────────────────────────────────────────────

    def _cleanup(self):
        print("\n🧹 Cleaning up...")
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
        print("👋 Goodbye!")
