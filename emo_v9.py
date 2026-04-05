#!/usr/bin/env python3
"""emo_v9.py - Reachy Mini Chat v9 with Vision and Piper-TTS

Features:
- Face tracking: Robot head follows user's face during conversation
- Security monitor: Motion detection and person presence logging
- Piper-TTS: Offline text-to-speech with lip sync
- ASR: Voice input via faster-whisper
- Emotion animation: Expressive robot movements
- Conversation history: Context-aware dialogue

Usage:
  # Pure chat mode (no vision)
  python emo_v9.py --piper-model models/en_US-lessac-high.onnx --asr
  
  # Face tracking + voice chat
  python emo_v9.py --vision face --asr --piper-model models/en_US-lessac-high.onnx
  
  # Security monitoring mode (no chat, head patrol)
  python emo_v9.py --vision monitor
  
  # Text chat with face tracking
  python emo_v9.py --vision face --chat
"""

import os
import sys
import time
import json
import wave
import tempfile
import asyncio
import argparse
import threading
import subprocess
import numpy as np
import select  # For checking keyboard/EOF input
import soundfile as sf
import sounddevice as sd
import aiohttp
from typing import Optional, Tuple, Dict, List
from collections import deque

# Import from existing modules
# We need to ensure we can import from current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from emo_v6 import EmotionControllerV6, LipSyncControllerV5
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from reachy_mini.motion.recorded_move import RecordedMoves

# Optional faster-whisper ASR engine
try:
    from utils.asr import FasterWhisperASREngine
except Exception:
    try:
        from .utils.asr import FasterWhisperASREngine
    except Exception:
        FasterWhisperASREngine = None

# Optional vision module
try:
    from vision import VisionController, VisionConfig, MonitorTracker, FaceTracker
    VISION_AVAILABLE = True
except ImportError as e:
    VISION_AVAILABLE = False


class PiperTTSEngine:
    """Piper-TTS engine wrapper for offline speech synthesis."""

    def __init__(self, model_path: str, config_path: str = None, speaker_id: int = 0, debug: bool = False):
        self.debug = debug
        self.model_path = model_path
        self.config_path = config_path
        self.speaker_id = speaker_id
        self.voice = None

        try:
            from piper import PiperVoice, PiperConfig
            # Import SynthesisConfig if available, else use default dict
            try:
                from piper import SynthesisConfig
                self.SynthesisConfig = SynthesisConfig
            except ImportError:
                self.SynthesisConfig = None

            import onnxruntime
            self.PiperVoice = PiperVoice
            self.PiperConfig = PiperConfig
            self.onnxruntime = onnxruntime
        except ImportError:
            print("❌ piper-tts not installed. Install with: pip install piper-tts")
            return

        if not os.path.exists(model_path):
            print(f"❌ Piper model not found at: {model_path}")

            # Try to find any onnx model in models/ or current directory
            print("🔍 Searching for available models...")
            found_models = []
            for search_dir in ['.', 'models']:
                if os.path.exists(search_dir):
                    for f in os.listdir(search_dir):
                        if f.endswith('.onnx'):
                            found_models.append(os.path.join(search_dir, f))

            if found_models:
                print(f"💡 Found available models:")
                for m in found_models:
                    print(f"   --piper-model {m}")
                print(f"\nExample: python emo_v8.py --piper-model {found_models[0]}")
            else:
                print("⚠️ No .onnx models found. Please download one from https://github.com/rhasspy/piper/releases/tag/v0.0.2")

            self.voice = None
            return

        try:
            # If config path not provided, assume .json with same name as .onnx
            if not config_path:
                potential_config = model_path + ".json"
                if os.path.exists(potential_config):
                    self.config_path = potential_config

            print(f"🎙️ Loading Piper model: {model_path}")

            # Manually load config to fix legacy phoneme_type issue
            with open(self.config_path or (model_path + ".json"), 'r', encoding='utf-8') as f:
                config_dict = json.load(f)

            # FIX: Replace legacy "PhonemeType.ESPEAK" string with "espeak"
            if config_dict.get('phoneme_type') == 'PhonemeType.ESPEAK':
                print("🔧 Fixing legacy phoneme_type in config...")
                config_dict['phoneme_type'] = 'espeak'

            # Create config object
            config = self.PiperConfig.from_dict(config_dict)

            # Create ONNX session
            session = self.onnxruntime.InferenceSession(
                str(model_path),
                sess_options=self.onnxruntime.SessionOptions(),
                providers=["CPUExecutionProvider"]
            )

            # Initialize voice manually
            self.voice = self.PiperVoice(session=session, config=config)

            print(f"✅ Piper TTS initialized")

        except Exception as e:
            print(f"❌ Failed to load Piper model: {e}")
            self.voice = None

    def speak_with_emotion(self, text: str, emotion: str = 'neutral'):
        """Speak text using Piper (blocking)."""
        if not text.strip():
            return

        if not self.voice:
            print(f"⚠️ Piper voice not loaded. Skipping speech: '{text[:20]}...'")
            return

        try:
            # Create a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name

            # Synthesize to file
            with wave.open(tmp_path, "wb") as wav_file:
                # Use synthesize_wav which handles wave header automatically
                syn_config = None
                if self.SynthesisConfig and self.speaker_id is not None:
                    syn_config = self.SynthesisConfig(speaker_id=self.speaker_id)

                self.voice.synthesize_wav(text, wav_file, syn_config=syn_config)

            # Read and play
            data, sr = sf.read(tmp_path, dtype='float32')
            if data.size > 0:
                sd.play(data, samplerate=sr)
                sd.wait()

            # Cleanup
            try:
                os.remove(tmp_path)
            except:
                pass

        except Exception as e:
            print(f"⚠️ Piper TTS error: {e}")

    async def speak_with_emotion_async(self, text: str, emotion: str = 'neutral'):
        """Async version of speak_with_emotion (runs in thread)."""
        # Piper synthesis is CPU bound, so run in a separate thread
        await asyncio.to_thread(self.speak_with_emotion, text, emotion)

    def speak_with_interrupt(self, text: str, emotion: str = 'neutral',
                             stop_event: threading.Event = None) -> bool:
        """
        Speak text with support for interrupt via stop_event.

        Args:
            text: Text to speak
            emotion: Emotion for TTS
            stop_event: Threading Event to signal interruption

        Returns:
            True if completed without interrupt
            False if interrupted
        """
        print(f"🔊 Speaking: '{text[:50]}...'")

        if not text.strip():
            return True

        if not self.voice:
            print(f"❌ Voice not loaded!")
            return True

        tmp_path = None
        interrupted = False
        try:
            # Create a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name

            # Synthesize to file
            with wave.open(tmp_path, "wb") as wav_file:
                syn_config = None
                if self.SynthesisConfig and self.speaker_id is not None:
                    syn_config = self.SynthesisConfig(speaker_id=self.speaker_id)
                self.voice.synthesize_wav(text, wav_file, syn_config=syn_config)

            # Read audio data
            data, sr = sf.read(tmp_path, dtype='float32')

            if data.size == 0:
                return True

            # Play audio with interrupt support
            sd.play(data, samplerate=sr)
            print(f"   💡 Press Ctrl+D to interrupt")

            # Wait for playback or interrupt
            while True:
                # Check if interrupted
                if stop_event and stop_event.is_set():
                    sd.stop()
                    print("\n⏹️  Interrupted")
                    interrupted = True
                    break

                # Check if playback finished
                stream = sd.get_stream()
                if stream is None or not stream.active:
                    break

                time.sleep(0.05)

            return not interrupted

        except Exception as e:
            print(f"❌ TTS error: {e}")
            return True
        finally:
            # Cleanup
            try:
                sd.stop()
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except:
                pass

    async def speak_with_interrupt_async(self, text: str, emotion: str = 'neutral') -> bool:
        """Async version of speak_with_interrupt."""
        return await asyncio.to_thread(self.speak_with_interrupt, text, emotion)


class ConversationHistory:
    """Manages conversation history for context-aware responses."""

    def __init__(self, max_rounds: int = 5):
        """
        Initialize conversation history.

        Args:
            max_rounds: Maximum number of conversation rounds to keep (default: 5)
        """
        self.max_rounds = max_rounds
        self.history: deque = deque(maxlen=max_rounds * 2)  # Each round has user + assistant
        self.enabled = True

    def add_user_message(self, message: str):
        """Add a user message to history."""
        if self.enabled and message.strip():
            self.history.append({"role": "user", "content": message.strip()})

    def add_assistant_message(self, message: str):
        """Add an assistant message to history."""
        if self.enabled and message.strip():
            self.history.append({"role": "assistant", "content": message.strip()})

    def get_messages(self, include_system: bool = True) -> List[Dict[str, str]]:
        """
        Get all messages formatted for Ollama API.

        Args:
            include_system: Whether to include system prompt

        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        messages = []

        if include_system:
            messages.append({
                "role": "system",
                "content": "You are a cute desktop robot assistant. Respond with enthusiasm and warmth. Remember the user's name and preferences from the conversation."
            })

        messages.extend(list(self.history))
        return messages

    def clear(self):
        """Clear all conversation history."""
        self.history.clear()
        print("🗑️  Conversation history cleared")

    def get_summary(self) -> str:
        """Get a summary of current history."""
        rounds = len(self.history) // 2
        return f"History: {rounds} rounds (max {self.max_rounds})"


class EmotionControllerV71(EmotionControllerV6):
    """Emotion controller using Piper-TTS instead of Edge-TTS."""

    def __init__(self, reachy: ReachyMini, piper_model: str, piper_config: str = None,
                 speaker_id: int = 0, debug: bool = False, gentle_mode: bool = False):
        # Step 1 Fix: Skip parent __init__ to avoid creating EdgeTTSEngine
        # Instead, directly initialize only what we need
        self.reachy = reachy
        self.debug = debug
        self.gentle_mode = gentle_mode
        self.is_speaking_action = False

        # Use Piper TTS directly (no Edge-TTS)
        self.tts_engine = PiperTTSEngine(piper_model, piper_config, speaker_id, debug)
        self.lip_sync = LipSyncControllerV5(reachy, debug=self.debug)

        # Load both libraries for richer motions
        self.emotions_lib = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
        self.dances_lib = RecordedMoves("pollen-robotics/reachy-mini-dances-library")

        self._categorize_recorded_moves()

        self.simple_actions = {
            'nod': self._simple_nod,
            'shake': self._simple_shake,
            'look_curious': self._simple_look_curious,
            'look_sad': self._simple_look_sad,
            'excited_wiggle': self._simple_excited_wiggle,
            'thoughtful_tilt': self._simple_thoughtful_tilt,
        }

    def _get_all_gentle_moves(self):
        """Collect all gentle moves from every category."""
        gentle_names = ['calming1', 'serenity1', 'thoughtful1', 'thoughtful2',
                        'attentive1', 'attentive2']
        all_gentle = []
        for category_moves in self.emotion_to_moves.values():
            for lib, name in category_moves:
                if name in gentle_names:
                    all_gentle.append((lib, name))
        return all_gentle

    def _choose_animation_for_emotion(self, emotion: str, intensity: str,
                                       avoid_move=None, used_moves=None):
        """Choose animation move based on emotion and intensity.

        Args:
            avoid_move: Optional last move to avoid repeating immediately.
            used_moves: Optional set of moves already used this round.

        Returns:
            (move, anim_intensity, speed) tuple
        """
        import random

        # Map emotion to category
        category_map = {
            'positive': 'positive',
            'negative': 'negative',
            'question': 'question',
            'activity': 'activity',
            'neutral': 'neutral',
            'happy': 'positive',
            'sad': 'negative',
            'angry': 'negative',
            'excited': 'activity',
            'curious': 'question',
        }
        category = category_map.get(emotion, 'neutral')

        # Build pool: 60% weight to primary category, 40% to all others combined
        primary = list(self.emotion_to_moves.get(category, []))
        if not primary:
            primary = list(self.emotion_to_moves.get('neutral', []))

        all_others = []
        for cat, moves in self.emotion_to_moves.items():
            if cat != category:
                all_others.extend(moves)

        # Create weighted pool (3x primary vs 2x others ≈ 60/40)
        pool = primary * 3 + all_others * 2

        # Filter gentle moves if in gentle mode
        if self.gentle_mode:
            gentle_names = ['calming1', 'serenity1', 'thoughtful1', 'thoughtful2',
                            'attentive1', 'attentive2']
            gentle_pool = [m for m in pool if m[1] in gentle_names]
            if gentle_pool:
                pool = gentle_pool
            else:
                all_gentle = self._get_all_gentle_moves()
                if all_gentle:
                    pool = all_gentle * 3
                else:
                    return None, intensity, 1.0

        if not pool:
            return None, intensity, 1.0

        # Prefer unused moves; reset tracking when pool is exhausted
        if used_moves:
            fresh = [m for m in pool if m not in used_moves]
            if fresh:
                pool = fresh
            else:
                used_moves.clear()

        # Avoid immediate repetition if possible
        if avoid_move and avoid_move in pool and len(set(pool)) > 1:
            pool = [m for m in pool if m != avoid_move]

        if intensity == 'high' and len(pool) > 1:
            move = pool[-1]
        elif intensity == 'low' and len(pool) > 1:
            move = pool[0]
        else:
            move = random.choice(pool)

        # Calculate speed
        speed_map = {'high': 1.2, 'medium': 1.0, 'low': 0.8}
        if self.gentle_mode:
            speed_map = {'high': 1.0, 'medium': 0.8, 'low': 0.6}
        speed = speed_map.get(intensity, 1.0)

        return move, intensity, speed

    def _execute_random_combined_action(self, emotion: str):
        """Execute a random combined simple action for richer variety."""
        import random
        category_map = {
            'positive': 'positive', 'negative': 'negative',
            'question': 'question', 'activity': 'activity',
            'neutral': 'neutral', 'happy': 'positive',
            'sad': 'negative', 'angry': 'negative',
            'excited': 'activity', 'curious': 'question',
        }
        cat = category_map.get(emotion, 'neutral')

        sequences = {
            'positive': [
                self._combined_nod_blink,
                self._combined_shake_blink_yaw,
                self._combined_wiggle_blink,
                self._combined_happy_tilt_blink_yaw,
                self._combined_excited_sequence,
            ],
            'negative': [
                self._combined_sad_blink,
                self._combined_thoughtful_blink_yaw,
                self._combined_slow_sequence,
                self._combined_negative_gesture,
            ],
            'question': [
                self._combined_curious_blink,
                self._combined_thoughtful_blink_yaw,
                self._combined_question_sequence,
                self._combined_nod_blink,
            ],
            'activity': [
                self._combined_wiggle_blink,
                self._combined_shake_blink_yaw,
                self._combined_activity_sequence,
                self._combined_happy_tilt_blink_yaw,
            ],
            'neutral': [
                self._combined_nod_blink,
                self._combined_thoughtful_blink_yaw,
                self._combined_neutral_sequence,
                self._combined_curious_blink,
            ],
        }
        actions = sequences.get(cat, [self._combined_nod_blink])
        random.choice(actions)()

    def _play_recorded_move(self, move, duration: float = 2.0):
        """Execute a recorded move."""
        if isinstance(move, tuple) and len(move) == 2:
            lib_tag, move_name = move
            if self.debug:
                print(f"🎬 Playing {lib_tag}/{move_name}")
            if lib_tag == 'emotions':
                mv = self.emotions_lib.get(move_name)
            else:
                mv = self.dances_lib.get(move_name)
        else:
            move_name = move
            if self.debug:
                print(f"🎬 Playing {move_name}")
            # Try dances first, then emotions
            try:
                mv = self.dances_lib.get(move_name)
            except Exception:
                mv = self.emotions_lib.get(move_name)

        self.reachy.play_move(mv, initial_goto_duration=duration)

    def speak_with_interrupt(self, text: str, emotion: str = 'neutral',
                             intensity: str = 'medium', level: float = 0.5,
                             stop_event: threading.Event = None) -> bool:
        """
        Speak with expression and support interrupt.
        TTS and animation run in parallel for natural interaction.

        Args:
            stop_event: Threading Event to signal interruption

        Returns:
            True if completed without interrupt
            False if interrupted
        """
        print(f"🎙️ Speaking: '{text[:50]}...'")

        # Use shorter move durations for more dynamic action (like emo_v6)
        duration_map = {'high': 0.8, 'medium': 1.0, 'low': 1.2}
        if self.gentle_mode:
            duration_map = {'high': 1.0, 'medium': 1.3, 'low': 1.5}
        base_move_duration = duration_map.get(intensity, 1.0)

        # Run TTS and animation in parallel
        import threading

        tts_done = threading.Event()

        def animation_thread():
            """Run animation in separate thread."""
            try:
                emotion_level = 0.5 if emotion == 'neutral' else 0.8

                if self.gentle_mode:
                    print(f"   😌 Gentle mode: subtle lip sync and gentle moves")
                else:
                    print(f"   🎵 Starting lip sync and continuous animations")

                # Always start lip sync so the robot keeps moving during the whole speech
                self.lip_sync.start_lip_sync(text, emotion_level)

                # Continuously play moves while TTS is running (like emo_v6/v8)
                last_move = None
                used_moves = set()
                while not tts_done.is_set():
                    import random
                    # Randomly choose action type for richer variety:
                    # 0-49% recorded move, 50-74% combined action, 75-99% body yaw
                    roll = random.randint(0, 99)

                    if roll < 50:
                        move, _, speed = self._choose_animation_for_emotion(
                            emotion, intensity, avoid_move=last_move, used_moves=used_moves
                        )
                        if move:
                            last_move = move
                            used_moves.add(move)
                            move_duration = base_move_duration / speed
                            if not self.gentle_mode:
                                print(f"   🎬 Animating: {move} (duration: {move_duration:.1f}s, speed: {speed:.1f}x)")
                            else:
                                print(f"   🎬 Gentle move: {move} (duration: {move_duration:.1f}s, speed: {speed:.1f}x)")
                            self._play_recorded_move(move, move_duration)
                        else:
                            print("   🎬 Simple action fallback")
                            self._simple_nod_once()
                            time.sleep(0.8)
                    elif roll < 75:
                        if not self.gentle_mode:
                            print("   🎭 Combined action")
                            self._execute_random_combined_action(emotion)
                        else:
                            print("   🎭 Gentle combined action")
                            self._simple_thoughtful_tilt_once()
                            time.sleep(0.8)
                    else:
                        # Body yaw rotation with slight head tilt for more natural movement
                        print("   🔄 Body turn")
                        try:
                            angle = random.choice([-0.5, -0.25, 0.25, 0.5])
                            head_tilt = random.choice([
                                create_head_pose(),
                                create_head_pose(roll=10, degrees=True),
                                create_head_pose(roll=-10, degrees=True),
                            ])
                            self.reachy.goto_target(head=head_tilt, body_yaw=angle, duration=0.4)
                            time.sleep(0.45)
                            self.reachy.goto_target(head=create_head_pose(), body_yaw=0.0, duration=0.4)
                            time.sleep(0.45)
                        except Exception:
                            pass

                    # Small pause between moves
                    if not tts_done.is_set():
                        time.sleep(0.3)

                self.lip_sync.stop_lip_sync()
                print(f"   ✅ Animations completed")
            except Exception as e:
                print(f"⚠️ Animation error: {e}")
                import traceback
                traceback.print_exc()
                self.lip_sync.stop_lip_sync()

        # Start animation in background thread
        anim_thread = threading.Thread(target=animation_thread, daemon=True)
        anim_thread.start()

        # Run TTS (blocking with interrupt support)
        speak_result = self.tts_engine.speak_with_interrupt(
            text, emotion=emotion, stop_event=stop_event
        )

        tts_done.set()

        # Wait for animation to complete (with timeout)
        anim_thread.join(timeout=20.0)

        # Reset body yaw to center so the robot faces forward again
        try:
            self.reachy.goto_target(body_yaw=0.0, duration=0.5)
        except Exception:
            pass

        return speak_result


class ChatAppWithPiper:
    def __init__(self,
                 model: str = "qwen3:0.6b",
                 ollama_url: str = "http://localhost:11434",
                 piper_model: str = "models/en-us-ryan-medium.onnx",
                 piper_config: str = None,
                 speaker_id: int = 0,
                 debug: bool = False,
                 use_asr: bool = False,
                 gentle: bool = False,
                 history_size: int = 5,
                 enable_history: bool = True,
                 asr_model: str = "small",
                 vad_silence: float = 0.8,
                 vad_aggressive: int = 1,
                 use_vad: bool = True):
        self.model = model
        self.ollama_url = ollama_url
        self.debug = debug
        self.use_asr = use_asr
        self.gentle = gentle
        self.piper_model = piper_model
        self.piper_config = piper_config
        self.speaker_id = speaker_id
        self.asr_model = asr_model  # Step 4 B: ASR model selection
        self.vad_silence = vad_silence  # VAD silence threshold (default 0.8s, increase if cutting off)
        self.vad_aggressive = vad_aggressive  # VAD aggressiveness 0-3 (1=gentle, 3=strict)
        self.use_vad = use_vad  # Whether to use VAD or fixed-duration recording

        self.controller: Optional[EmotionControllerV71] = None
        self.asr_engine = None

        # Step 2: Conversation history
        self.history = ConversationHistory(max_rounds=history_size)
        self.history.enabled = enable_history

        # Interrupt handling
        self._stop_speaking_event = threading.Event()

    async def check_ollama_model(self, session: aiohttp.ClientSession) -> bool:
        """Check if the requested model is available in Ollama."""
        try:
            print(f"🔍 Checking Ollama model '{self.model}'...")
            async with session.get(f"{self.ollama_url}/api/tags", timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [m['name'] for m in data.get('models', [])]
                    # Check for exact match or match without tag (e.g. 'qwen2.5:0.5b' vs 'qwen2.5:0.5b-instruct')
                    # Ollama models usually have tags.
                    if self.model in models:
                        print(f"✅ Model '{self.model}' found.")
                        return True
                    # Check if 'latest' tag is implied
                    if f"{self.model}:latest" in models:
                        print(f"✅ Model '{self.model}:latest' found.")
                        return True

                    print(f"⚠️ Model '{self.model}' not found in Ollama list.")
                    print(f"   Available models: {', '.join(models)}")
                    print("   Attempting to use it anyway (Ollama might pull it or error)...")
                    return False
        except Exception as e:
            print(f"⚠️ Could not check available models: {e}")
        return True  # Assume it might work

    async def _get_ollama_response_async(self, prompt: str, session: aiohttp.ClientSession) -> Optional[str]:
        """Get response from Ollama (streaming) using /api/chat with history."""
        try:
            if self.debug:
                print(f"\nDEBUG: Sending request to {self.ollama_url}/api/chat")
                print(f"DEBUG: Model: {self.model}")
                if self.history.enabled:
                    print(f"DEBUG: {self.history.get_summary()}")

            # Increase timeout significantly as loading a model can take time
            timeout_seconds = 300

            # Build messages with history
            if self.history.enabled:
                # Get existing history (includes system prompt)
                messages = self.history.get_messages(include_system=True)
                # Add current user message
                messages.append({"role": "user", "content": prompt})
            else:
                # Original behavior without history
                messages = [
                    {"role": "system", "content": "You are a cute desktop robot assistant. Respond with enthusiasm and warmth."},
                    {"role": "user", "content": prompt}
                ]

            async with session.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                    # Some thinking-capable models can emit only `message.thinking`.
                    # Ask for direct answer text in `message.content`.
                    "think": False,
                    "options": {"temperature": 0.8, "num_predict": 200}
                },
                timeout=aiohttp.ClientTimeout(total=timeout_seconds)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"\n⚠️ Ollama error ({response.status}): {error_text}")
                    return None

                if self.debug:
                    print(f"DEBUG: Response received (Status {response.status}). Streaming content...")

                full_response = ""
                thinking_response = ""
                chunk_count = 0

                async for line in response.content:
                    if line:
                        try:
                            decoded = line.decode('utf-8')
                            chunk = json.loads(decoded)
                            chunk_count += 1

                            if self.debug and chunk_count <= 3:
                                print(f"DEBUG Chunk {chunk_count}: {decoded.strip()}")

                            content = ""
                            # Handle /api/chat response format
                            if 'message' in chunk and 'content' in chunk['message']:
                                content = chunk['message']['content']
                                thinking_response += chunk['message'].get('thinking', '')
                            # Fallback for /api/generate format (just in case)
                            elif 'response' in chunk:
                                content = chunk['response']

                            if content:
                                print(content, end="", flush=True)
                                full_response += content

                            if chunk.get('done'):
                                if self.debug:
                                    print(f"\nDEBUG: Generation complete. Total stats: {chunk.get('total_duration', 0)/1e9:.2f}s")

                        except Exception as e:
                            if self.debug:
                                print(f"\nDEBUG: JSON parse error: {e}")
                            continue

                if not full_response and thinking_response:
                    # Fallback for servers/models that still stream into `thinking`.
                    print(thinking_response, end="", flush=True)
                    full_response = thinking_response
                    if self.debug:
                        print("\nDEBUG: Used thinking stream as fallback response.")

                print()
                if not full_response and self.debug:
                    print("DEBUG: Warning - Empty response received from Ollama")

                return full_response

        except asyncio.TimeoutError:
            print(f"\n⚠️ Ollama request timed out after {timeout_seconds}s")
            return None
        except Exception as e:
            print(f"\n⚠️ Ollama async error: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return None

    async def _show_thinking_animation(self, reachy: ReachyMini, duration: float = 5.0):
        """Show robot 'thinking' animation."""
        import math
        start_time = time.time()
        while time.time() - start_time < duration:
            angle = math.sin((time.time() - start_time) * 3) * 0.1
            pose = create_head_pose(roll=angle)
            reachy.goto_target(head=pose, duration=0.3)
            await asyncio.sleep(0.1)

            if hasattr(reachy, 'l_antenna') and hasattr(reachy, 'r_antenna'):
                reachy.l_antenna.goto_position(angle * 0.5, duration=0.2)
                reachy.r_antenna.goto_position(-angle * 0.5, duration=0.2)

            await asyncio.sleep(0.2)

        reachy.goto_target(head=create_head_pose(), duration=0.5)

    def _speak_and_animate(self, response: str, emotion: str, intensity: str,
                           emotion_level: float, stop_event: threading.Event = None) -> bool:
        """
        Helper to run speech and animation with interrupt support.
        """
        if not self.controller:
            print("❌ No controller available!")
            return True

        try:
            return self.controller.speak_with_interrupt(
                response, emotion, intensity, emotion_level, stop_event
            )
        except Exception as e:
            print(f"❌ Speech error: {e}")
            return True

    async def start_chat_async(self):
        print("="*60)
        print("🤖 Reachy Mini Chat v9 with Piper-TTS")
        print("="*60)
        print(f"Ollama Model: {self.model}")
        print(f"Ollama URL: {self.ollama_url}")
        print(f"Piper Model: {self.piper_model}")
        # Step 2: Show history status
        if self.history.enabled:
            print(f"💬 Conversation history: {self.history.max_rounds} rounds (type 'clear' to reset)")
        else:
            print("💬 Conversation history: disabled")
        print("💡 Need more voices? Download .onnx models from:")
        print("   https://github.com/rhasspy/piper/releases/tag/v0.0.2")

        try:
            with ReachyMini(media_backend="no_media") as reachy:
                print("✅ Connected to Reachy Mini")

                # Disable automatic body yaw so recorded moves can control body rotation
                reachy.set_automatic_body_yaw(False)

                # Initialize controller with Piper
                self.controller = EmotionControllerV71(
                    reachy,
                    self.piper_model,
                    self.piper_config,
                    self.speaker_id,
                    self.debug,
                    gentle_mode=self.gentle
                )

                reachy.goto_target(head=create_head_pose(), duration=1.0)
                await asyncio.sleep(1.0)

                if self.use_asr:
                    if FasterWhisperASREngine is None:
                        print("❌ ASR requested but FasterWhisperASREngine not available.")
                        return

                    print(f"Initializing ASR engine ({self.asr_model}, VAD: {self.vad_silence}s silence)... (this may take a few seconds)")
                    try:
                        # Run ASR initialization in thread to not block event loop
                        self.asr_engine = await asyncio.to_thread(
                            FasterWhisperASREngine,
                            model_name=self.asr_model,
                            device='cpu'
                        )
                    except Exception as e:
                        print(f"❌ Failed to initialize ASR engine: {e}")
                        return

                    print("\n🎤 VAD ASR + Async mode: press Ctrl-C to stop")

                    async with aiohttp.ClientSession() as session:
                        # Check model once
                        await self.check_ollama_model(session)

                        while True:
                            try:
                                print("\n🎙️ Speak now... (Ctrl+C to exit)")

                                # Step 4 A: Timing - ASR
                                asr_start = time.time()

                                if self.use_vad:
                                    # VAD-based recording - stops on silence
                                    transcription = await asyncio.to_thread(
                                        self.asr_engine.transcribe_from_mic_vad,
                                        max_duration=4.0,
                                        silence_threshold=self.vad_silence,
                                        aggressiveness=self.vad_aggressive,
                                        trailing_buffer_ms=300,
                                        show_volume=True
                                    )
                                else:
                                    # Fixed-duration recording - always records 4s
                                    transcription = await asyncio.to_thread(
                                        self.asr_engine.transcribe_from_mic,
                                        duration=4.0,
                                        show_volume=True
                                    )

                                asr_time = time.time() - asr_start

                                if not transcription:
                                    print("⚠️ No speech detected, try again")
                                    continue

                                # Step 2: Add to history
                                self.history.add_user_message(transcription)

                                print(f"📝 You: {transcription}")
                                if self.history.enabled:
                                    print(f"  {self.history.get_summary()}")
                                print("\n🤖 Reachy Mini: ", end="", flush=True)

                                # Step 4 A: Timing - LLM
                                llm_start = time.time()

                                thinking_task = asyncio.create_task(self._show_thinking_animation(reachy, 10.0))
                                llm_task = asyncio.create_task(self._get_ollama_response_async(transcription, session))

                                response = await llm_task

                                llm_time = time.time() - llm_start

                                thinking_task.cancel()
                                try:
                                    await thinking_task
                                except asyncio.CancelledError:
                                    pass

                                if response and self.controller:
                                    # Step 4 A: Timing - TTS/Animation
                                    tts_start = time.time()

                                    emotion, intensity, emotion_level = self.controller.analyze_emotion(response)

                                    # Reset interrupt event
                                    self._stop_speaking_event.clear()

                                    # Run TTS in thread with interrupt support
                                    speech_task = asyncio.create_task(asyncio.to_thread(
                                        self._speak_and_animate,
                                        response, emotion, intensity, emotion_level,
                                        self._stop_speaking_event
                                    ))

                                    # Wait for TTS to complete or Ctrl+D to interrupt
                                    try:
                                        while not speech_task.done():
                                            # Check for Ctrl+D (EOF)
                                            if select.select([sys.stdin], [], [], 0)[0]:
                                                try:
                                                    char = sys.stdin.read(1)
                                                    if char == '':  # EOF = Ctrl+D
                                                        print("\n⏹️  Interrupting...")
                                                        self._stop_speaking_event.set()
                                                        break
                                                except:
                                                    pass
                                            await asyncio.sleep(0.05)

                                        speech_completed = await speech_task
                                    except asyncio.CancelledError:
                                        speech_completed = False

                                    tts_time = time.time() - tts_start
                                    total_time = asr_time + llm_time + tts_time

                                    # Step 2: Add to history
                                    self.history.add_assistant_message(response)

                                    if not speech_completed:
                                        print("🎤 Ready for your next question...")

                                    # Step 4 A: Display timing
                                    if self.debug:
                                        status = "completed" if speech_completed else "interrupted"
                                        print(f"\n  ⏱️  [Timing] ASR: {asr_time:.2f}s, LLM: {llm_time:.2f}s, TTS: {tts_time:.2f}s ({status}), Total: {total_time:.2f}s")

                            except KeyboardInterrupt:
                                print("\n\n👋 Goodbye!")
                                return
                            except Exception as e:
                                print(f"⚠️ Error: {e}")
                                await asyncio.sleep(1.0)

                else:
                    print("\n💬 Start chatting (type 'quit' or Ctrl+C to exit)")
                    async with aiohttp.ClientSession() as session:
                        # Check model once
                        await self.check_ollama_model(session)

                        while True:
                            try:
                                user_input = input("\n🧑 You: ").strip()
                                if user_input.lower() in ['quit', 'exit', 'q']:
                                    break
                                if user_input.lower() == 'clear':
                                    self.history.clear()
                                    continue
                                if not user_input:
                                    continue

                                # Step 2: Add to history
                                self.history.add_user_message(user_input)

                                print("\n🤖 Reachy Mini: ", end="", flush=True)

                                # Step 4 A: Timing - LLM
                                llm_start = time.time()

                                thinking_task = asyncio.create_task(self._show_thinking_animation(reachy, 10.0))
                                llm_task = asyncio.create_task(self._get_ollama_response_async(user_input, session))

                                response = await llm_task

                                llm_time = time.time() - llm_start

                                thinking_task.cancel()
                                try:
                                    await thinking_task
                                except asyncio.CancelledError:
                                    pass

                                if response and self.controller:
                                    # Step 4 A: Timing - TTS/Animation
                                    tts_start = time.time()

                                    emotion, intensity, emotion_level = self.controller.analyze_emotion(response)

                                    # Reset interrupt event
                                    self._stop_speaking_event.clear()

                                    # Run TTS in thread with interrupt support
                                    speech_task = asyncio.create_task(asyncio.to_thread(
                                        self._speak_and_animate,
                                        response, emotion, intensity, emotion_level,
                                        self._stop_speaking_event
                                    ))

                                    # Wait for TTS to complete or Ctrl+D to interrupt
                                    try:
                                        while not speech_task.done():
                                            # Check for Ctrl+D (EOF)
                                            if select.select([sys.stdin], [], [], 0)[0]:
                                                try:
                                                    char = sys.stdin.read(1)
                                                    if char == '':  # EOF = Ctrl+D
                                                        print("\n⏹️  Interrupting...")
                                                        self._stop_speaking_event.set()
                                                        break
                                                except:
                                                    pass
                                            await asyncio.sleep(0.05)

                                        speech_completed = await speech_task
                                    except asyncio.CancelledError:
                                        speech_completed = False

                                    tts_time = time.time() - tts_start
                                    total_time = llm_time + tts_time

                                    # Step 2: Add to history
                                    self.history.add_assistant_message(response)

                                    if not speech_completed:
                                        print("🎤 Ready for your next question...")

                                    # Step 4 A: Display timing
                                    if self.debug:
                                        status = "completed" if speech_completed else "interrupted"
                                        print(f"\n  ⏱️  [Timing] LLM: {llm_time:.2f}s, TTS: {tts_time:.2f}s ({status}), Total: {total_time:.2f}s")

                            except KeyboardInterrupt:
                                print("\n\n👋 Goodbye!")
                                return
                            except Exception as e:
                                print(f"\n⚠️ Error: {e}")

        except Exception as e:
            print(f"\n❌ Cannot connect to Reachy Mini: {e}")
            self._tts_only_mode()

    def start_chat(self):
        asyncio.run(self.start_chat_async())

    def _tts_only_mode(self):
        print("\n📻 Running in TTS-only mode (no robot)")
        print("💡 Need more voices? Download .onnx models from:")
        print("   https://github.com/rhasspy/piper/releases/tag/v0.0.2")
        tts = PiperTTSEngine(self.piper_model, self.piper_config, self.speaker_id, self.debug)
        tts.speak_with_emotion("Hello! Piper TTS is working.", "neutral")


class ChatAppWithVision(ChatAppWithPiper):
    """Chat application with optional vision capabilities.
    
    Extends ChatAppWithPiper to add face tracking and visual interaction.
    Vision features are optional and controlled via --vision flag.
    
    Args:
        vision_enabled: Master switch for vision features
        vision_fps: Target camera processing FPS
        vision_auto_wake: Wake robot when person detected
        *args, **kwargs: Passed to ChatAppWithPiper
    """
    
    def __init__(
        self,
        vision_enabled: bool = False,
        vision_mode: Optional[str] = None,
        vision_fps: float = 15.0,
        vision_auto_wake: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.vision_enabled = vision_enabled and VISION_AVAILABLE
        self.vision_mode = vision_mode
        self.enable_face = vision_mode == 'face' if vision_mode else False
        
        self.vision: Optional[VisionController] = None
        self.monitor_tracker: Optional[MonitorTracker] = None
        
        self._vision_config = VisionConfig(
            enabled=self.vision_enabled and vision_mode == 'face',
            face_tracking=True,
            target_fps=vision_fps,
            auto_wake=vision_auto_wake,
            track_while_speaking=True
        ) if VISION_AVAILABLE else None
        
        # State for vision integration
        self._person_present = False
        self._face_tracking_active = False
        self._is_speaking = False
        self._monitor_active = False
        self._patrol_active = False
        self._patrol_paused = False
    
    async def start_chat_async(self):
        """Start chat with optional vision capabilities."""
        print("=" * 60)
        print("🤖 Reachy Mini Chat v9 with Vision")
        print("=" * 60)
        
        if self.vision_enabled:
            if self.vision_mode == 'monitor':
                print(f"👁️  Vision mode: SECURITY MONITOR 🔒")
                print(f"   - Motion detection: Yes")
                print(f"   - Person presence: Yes")
                print(f"   - Event logging: Yes")
            else:
                print(f"👁️  Vision features: ENABLED")
                print(f"   - Face tracking: Yes")
                print(f"   - Target FPS: {self._vision_config.target_fps}")
                print(f"   - Auto wake: {self._vision_config.auto_wake}")
        else:
            print("👁️  Vision features: DISABLED")
            if not VISION_AVAILABLE:
                print("   (MediaPipe not installed)")
        
        print(f"🎙️  Piper Model: {self.piper_model}")
        print(f"💬 History: {self.history.max_rounds} rounds")
        print("-" * 60)
        
        try:
            media_backend = "default" if self.vision_enabled else "no_media"
            
            if self.vision_mode == 'monitor':
                self._vision_config = VisionConfig(enabled=True, face_tracking=True)
            
            with ReachyMini(media_backend=media_backend) as reachy:
                print("✅ Connected to Reachy Mini")
                
                reachy.set_automatic_body_yaw(False)
                
                if self.vision_enabled:
                    if self.vision_mode == 'monitor':
                        self._start_security_monitoring(reachy)
                        
                        print("\n   🔒 Monitor mode active. Press Ctrl+C to stop.\n")
                        try:
                            while self._monitor_active:
                                await asyncio.sleep(1.0)
                        except KeyboardInterrupt:
                            print("\n   🛑 Monitor stopped by user")
                        return
                    else:
                        self.vision = VisionController(reachy, self._vision_config)
                        self._setup_vision_callbacks()
                        self.vision.start()
                        self._start_idle_face_tracking(reachy)
                
                self.controller = EmotionControllerV71(
                    reachy,
                    self.piper_model,
                    self.piper_config,
                    self.speaker_id,
                    self.debug,
                    gentle_mode=self.gentle
                )
                
                reachy.goto_target(head=create_head_pose(), duration=1.0)
                await asyncio.sleep(1.0)
                
                if self.use_asr:
                    await self._chat_with_asr_vision(reachy)
                else:
                    await self._chat_text_vision(reachy)
                    
        except Exception as e:
            print(f"❌ Error: {e}")
            raise
        finally:
            if self.vision:
                self.vision.stop()
            if self.monitor_tracker:
                self._monitor_active = False
                self._patrol_active = False
                self.monitor_tracker.stop()
    
    def _setup_vision_callbacks(self):
        """Setup vision event callbacks."""
        if not self.vision:
            return
        
        def on_person_enter():
            if self._vision_config.auto_wake and not self._person_present:
                print("   👋 Person detected!")
                self._person_present = True
        
        def on_person_leave():
            self._person_present = False
            print("   😴 No person detected")
        
        self.vision.on_person_enter = on_person_enter
        self.vision.on_person_leave = on_person_leave
    
    def _start_idle_face_tracking(self, reachy):
        """Start background thread for ultra-smooth face tracking."""
        def idle_tracker():
            print("   👁️  Idle face tracking started (ultra-smooth)")
            
            ema1_x: Optional[float] = None
            ema1_y: Optional[float] = None
            alpha1 = 0.3
            
            ema2_x: Optional[float] = None
            ema2_y: Optional[float] = None
            alpha2 = 0.2
            
            ema3_x: Optional[float] = None
            ema3_y: Optional[float] = None
            alpha3 = 0.15
            
            last_sent_pos: Optional[Tuple[int, int]] = None
            min_update_interval = 0.08
            position_threshold = 40
            
            last_update_time = 0.0
            
            while self.vision and self.vision._running:
                current_time = time.time()
                
                if current_time - last_update_time < min_update_interval:
                    time.sleep(0.01)
                    continue
                
                if not self._is_speaking:
                    if self.vision.is_person_present():
                        if pos := self.vision.get_face_position():
                            raw_x, raw_y = pos
                            
                            if ema1_x is None:
                                ema1_x, ema1_y = float(raw_x), float(raw_y)
                                ema2_x, ema2_y = ema1_x, ema1_y
                                ema3_x, ema3_y = ema2_x, ema2_y
                            else:
                                ema1_x = alpha1 * raw_x + (1 - alpha1) * ema1_x
                                ema1_y = alpha1 * raw_y + (1 - alpha1) * ema1_y
                                ema2_x = alpha2 * ema1_x + (1 - alpha2) * ema2_x
                                ema2_y = alpha2 * ema1_y + (1 - alpha2) * ema2_y
                                ema3_x = alpha3 * ema2_x + (1 - alpha3) * ema3_x
                                ema3_y = alpha3 * ema2_y + (1 - alpha3) * ema3_y
                            
                            final_pos = (int(ema3_x), int(ema3_y))
                            
                            should_update = True
                            if last_sent_pos:
                                dx = abs(final_pos[0] - last_sent_pos[0])
                                dy = abs(final_pos[1] - last_sent_pos[1])
                                if dx < position_threshold and dy < position_threshold:
                                    should_update = False
                            
                            if should_update:
                                try:
                                    reachy.look_at_image(
                                        final_pos[0], final_pos[1], 
                                        duration=0.6
                                    )
                                    last_sent_pos = final_pos
                                    last_update_time = current_time
                                    if self.debug:
                                        print(f"   👁️  Track ({raw_x},{raw_y})→({final_pos[0]},{final_pos[1]})")
                                except Exception:
                                    pass
                    else:
                        if last_sent_pos is not None:
                            try:
                                reachy.goto_target(head=create_head_pose(), duration=0.8)
                                last_sent_pos = None
                                ema1_x = ema1_y = None
                                ema2_x = ema2_y = None
                                ema3_x = ema3_y = None
                                last_update_time = current_time
                                if self.debug:
                                    print("   👁️  No face - center")
                            except Exception:
                                pass
                
                time.sleep(0.01)
        
        tracker_thread = threading.Thread(target=idle_tracker, daemon=True)
        tracker_thread.start()
        print("   ✅ Ultra-smooth face tracking started")
    
    def _start_security_monitoring(self, reachy):
        """Start security monitoring mode."""
        self.monitor_tracker = MonitorTracker(
            motion_threshold=25,
            min_motion_area=500,
            cooldown_seconds=5.0,
            buffer_seconds=3.0
        )
        self.monitor_tracker.start()
        self._monitor_active = True
        
        def on_motion_event(event):
            print(f"   🚨 MOTION DETECTED at {event.timestamp.strftime('%H:%M:%S')}")
            print(f"      {event.description}")
            try:
                reachy.goto_target(head=create_head_pose(), duration=0.5)
            except Exception:
                pass
        
        def on_person_enter_event(event):
            print(f"   👤 PERSON ENTERED at {event.timestamp.strftime('%H:%M:%S')}")
            print(f"      {event.description}")
            try:
                reachy.goto_target(head=create_head_pose(yaw=15, degrees=True), duration=0.3)
                time.sleep(0.3)
                reachy.goto_target(head=create_head_pose(yaw=-15, degrees=True), duration=0.3)
                time.sleep(0.3)
                reachy.goto_target(head=create_head_pose(), duration=0.3)
            except Exception:
                pass
        
        def on_person_leave_event(event):
            print(f"   🚪 PERSON LEFT at {event.timestamp.strftime('%H:%M:%S')}")
            print(f"      {event.description}")
        
        def on_anomaly_event(event):
            print(f"   ⚠️  ANOMALY at {event.timestamp.strftime('%H:%M:%S')}")
            print(f"      {event.description}")
        
        self.monitor_tracker.on_motion = on_motion_event
        self.monitor_tracker.on_person_enter = on_person_enter_event
        self.monitor_tracker.on_person_leave = on_person_leave_event
        self.monitor_tracker.on_anomaly = on_anomaly_event
        
        face_tracker = FaceTracker(smooth_factor=0.3)
        
        self._patrol_active = True
        self._patrol_paused = False
        
        def patrol_loop():
            """Head patrol loop: specific waypoints with 1s hold at each position."""
            print("   🔄 Head patrol started: Center ↔ Left50° ↔ Right50°")
            
            waypoints = [
                (25, 1.0), (50, 1.0), (25, 1.0), (0, 1.0),
                (-25, 1.0), (-50, 1.0), (-25, 1.0), (0, 1.0),
            ]
            
            try:
                reachy.goto_target(head=create_head_pose(yaw=0, degrees=True), duration=0.5)
                time.sleep(0.5)
            except Exception:
                pass
            
            waypoint_index = 0
            
            while self._patrol_active:
                if self._patrol_paused:
                    time.sleep(0.1)
                    continue
                
                try:
                    yaw_angle, hold_time = waypoints[waypoint_index]
                    reachy.goto_target(
                        head=create_head_pose(yaw=yaw_angle, degrees=True),
                        duration=0.4
                    )
                    time.sleep(0.4 + hold_time)
                    waypoint_index = (waypoint_index + 1) % len(waypoints)
                except Exception as e:
                    if self.debug:
                        print(f"      ⚠️ Patrol error: {e}")
                    time.sleep(0.5)
            
            print("   🛑 Head patrol stopped")
        
        def monitor_loop():
            """Continuous monitoring loop."""
            print("   🔒 Monitor loop started")
            frame_count = 0
            
            while self._monitor_active and self.monitor_tracker:
                try:
                    frame = None
                    if hasattr(reachy, 'media') and reachy.media:
                        frame = reachy.media.get_frame()
                    
                    if frame is not None:
                        frame_count += 1
                        
                        person_detected = False
                        face_pos = face_tracker.get_face_center(frame)
                        person_detected = face_pos is not None
                        
                        event = self.monitor_tracker.process_frame(frame, person_detected)
                        
                        if event and event.event_type in ('motion', 'person_enter'):
                            self._patrol_paused = True
                            threading.Timer(5.0, lambda: setattr(self, '_patrol_paused', False)).start()
                        
                        if frame_count % 300 == 0:
                            stats = self.monitor_tracker.get_event_stats()
                            print(f"\n   📊 Monitor Stats (last 30s):")
                            print(f"      Total events: {stats['total_events']}")
                            print(f"      Motion: {stats['motion_count']}")
                            print(f"      Person enter: {stats['person_enter_count']}")
                except Exception as e:
                    if self.debug:
                        print(f"      ⚠️ Monitor error: {e}")
                
                time.sleep(0.1)
        
        patrol_thread = threading.Thread(target=patrol_loop, daemon=True)
        patrol_thread.start()
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        
        print("   ✅ Security monitoring + patrol active")
    
    def _speak_with_face_tracking(
        self, text: str, emotion: str, intensity: str,
        emotion_level: float, stop_event: threading.Event
    ) -> bool:
        """Speak with face tracking enabled."""
        self._is_speaking = True
        print(f"🎙️ Speaking: '{text[:50]}...'")
        
        duration_map = {'high': 0.8, 'medium': 1.0, 'low': 1.2}
        if self.controller.gentle_mode:
            duration_map = {'high': 1.0, 'medium': 1.3, 'low': 1.5}
        base_move_duration = duration_map.get(intensity, 1.0)
        
        tts_done = threading.Event()
        
        def animation_thread():
            try:
                emotion_level = 0.5 if emotion == 'neutral' else 0.8
                
                if self.controller.gentle_mode:
                    print("   😌 Gentle mode with face tracking")
                else:
                    print("   🎵 Animation + face tracking")
                
                self.controller.lip_sync.start_lip_sync(text, emotion_level)
                
                last_move = None
                used_moves = set()
                last_face_look = 0.0
                
                while not tts_done.is_set():
                    import random
                    
                    if self.enable_face:
                        current_time = time.time()
                        if current_time - last_face_look > 3.0:
                            if self.vision and self.vision.is_person_present():
                                if pos := self.vision.get_face_position():
                                    try:
                                        self.controller.reachy.look_at_image(
                                            pos[0], pos[1], duration=0.2
                                        )
                                        print(f"   👁️  Glance at face ({pos[0]}, {pos[1]})", flush=True)
                                    except Exception:
                                        pass
                            last_face_look = current_time
                    
                    roll = random.randint(0, 99)
                    
                    if roll < 50:
                        move, _, speed = self.controller._choose_animation_for_emotion(
                            emotion, intensity, avoid_move=last_move, used_moves=used_moves
                        )
                        if move:
                            last_move = move
                            used_moves.add(move)
                            move_duration = base_move_duration / speed
                            if not self.controller.gentle_mode:
                                print(f"   🎬 {move} ({move_duration:.1f}s)")
                            else:
                                print(f"   🎬 Gentle: {move}")
                            self.controller._play_recorded_move(move, move_duration)
                        else:
                            self.controller._simple_nod_once()
                            time.sleep(0.8)
                    elif roll < 75:
                        if not self.controller.gentle_mode:
                            print("   🎭 Combined action")
                            self.controller._execute_random_combined_action(emotion)
                        else:
                            self.controller._simple_thoughtful_tilt_once()
                            time.sleep(0.8)
                    else:
                        print("   🔄 Body turn")
                        try:
                            angle = random.choice([-0.5, -0.25, 0.25, 0.5])
                            head_tilt = random.choice([
                                create_head_pose(),
                                create_head_pose(roll=10, degrees=True),
                                create_head_pose(roll=-10, degrees=True),
                            ])
                            self.controller.reachy.goto_target(
                                head=head_tilt, body_yaw=angle, duration=0.4
                            )
                            time.sleep(0.45)
                            self.controller.reachy.goto_target(
                                head=create_head_pose(), body_yaw=0.0, duration=0.4
                            )
                            time.sleep(0.45)
                        except Exception:
                            pass
                    
                    if not tts_done.is_set():
                        time.sleep(0.3)
                
                self.controller.lip_sync.stop_lip_sync()
                print("   ✅ Animation completed")
                
            except Exception as e:
                print(f"⚠️ Animation error: {e}")
                import traceback
                traceback.print_exc()
                self.controller.lip_sync.stop_lip_sync()
                self._is_speaking = False
        
        anim_thread = threading.Thread(target=animation_thread, daemon=True)
        anim_thread.start()
        
        speak_result = self.controller.tts_engine.speak_with_interrupt(
            text, emotion=emotion, stop_event=stop_event
        )
        
        tts_done.set()
        anim_thread.join(timeout=20.0)
        
        self._is_speaking = False
        
        try:
            self.controller.reachy.goto_target(body_yaw=0.0, duration=0.5)
        except Exception:
            pass
        
        return speak_result
    
    async def _chat_with_asr_vision(self, reachy):
        """ASR chat mode with vision."""
        import select
        
        print("\n🎤 VAD ASR + Vision mode: press Ctrl-C to stop")
        
        if FasterWhisperASREngine is None:
            print("❌ ASR not available")
            return
        
        print(f"Initializing ASR ({self.asr_model}, VAD: {self.vad_silence}s silence)...")
        try:
            self.asr_engine = await asyncio.to_thread(
                FasterWhisperASREngine,
                model_name=self.asr_model,
                device='cpu'
            )
        except Exception as e:
            print(f"❌ Failed to initialize ASR: {e}")
            return
        
        async with aiohttp.ClientSession() as session:
            await self.check_ollama_model(session)
            
            while True:
                try:
                    print("\n🎙️ Speak now... (Ctrl+C to exit)")
                    
                    asr_start = time.time()
                    
                    if self.use_vad:
                        transcription = await asyncio.to_thread(
                            self.asr_engine.transcribe_from_mic_vad,
                            max_duration=4.0,
                            silence_threshold=self.vad_silence,
                            aggressiveness=self.vad_aggressive,
                            trailing_buffer_ms=300,
                            show_volume=True
                        )
                    else:
                        transcription = await asyncio.to_thread(
                            self.asr_engine.transcribe_from_mic,
                            duration=4.0,
                            show_volume=True
                        )
                    
                    asr_time = time.time() - asr_start
                    
                    if not transcription:
                        print("⚠️ No speech detected, try again")
                        continue
                    
                    self.history.add_user_message(transcription)
                    print(f"📝 You: {transcription}")
                    
                    print("\n🤖 Reachy Mini: ", end="", flush=True)
                    llm_start = time.time()
                    
                    response = await self._get_ollama_response_async(transcription, session)
                    llm_time = time.time() - llm_start
                    
                    if response and self.controller:
                        emotion, intensity, emotion_level = self.controller.analyze_emotion(response)
                        self._stop_speaking_event.clear()
                        
                        tts_start = time.time()
                        speech_task = asyncio.create_task(asyncio.to_thread(
                            self._speak_with_face_tracking,
                            response, emotion, intensity, emotion_level,
                            self._stop_speaking_event
                        ))
                        
                        try:
                            while not speech_task.done():
                                if select.select([sys.stdin], [], [], 0)[0]:
                                    try:
                                        char = sys.stdin.read(1)
                                        if char == '':
                                            print("\n⏹️ Interrupting...")
                                            self._stop_speaking_event.set()
                                            break
                                    except:
                                        pass
                                await asyncio.sleep(0.05)
                            
                            await speech_task
                        except asyncio.CancelledError:
                            pass
                        
                        tts_time = time.time() - tts_start
                        self.history.add_assistant_message(response)
                        
                        if self.debug:
                            print(f"\n  ⏱️ ASR: {asr_time:.2f}s, LLM: {llm_time:.2f}s, TTS: {tts_time:.2f}s")
                    
                except KeyboardInterrupt:
                    print("\n\n👋 Goodbye!")
                    return
                except Exception as e:
                    print(f"\n⚠️ Error: {e}")
                    await asyncio.sleep(1.0)
    
    async def _chat_text_vision(self, reachy):
        """Text chat mode with vision."""
        import select
        
        print("\n💬 Start chatting (type 'quit' or Ctrl+C to exit)")
        
        async with aiohttp.ClientSession() as session:
            await self.check_ollama_model(session)
            
            while True:
                try:
                    user_input = input("\n🧑 You: ").strip()
                    
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        break
                    if user_input.lower() == 'clear':
                        self.history.clear()
                        continue
                    if not user_input:
                        continue
                    
                    self.history.add_user_message(user_input)
                    print("\n🤖 Reachy Mini: ", end="", flush=True)
                    
                    llm_start = time.time()
                    response = await self._get_ollama_response_async(user_input, session)
                    llm_time = time.time() - llm_start
                    
                    if response and self.controller:
                        emotion, intensity, emotion_level = self.controller.analyze_emotion(response)
                        self._stop_speaking_event.clear()
                        
                        tts_start = time.time()
                        speech_task = asyncio.create_task(asyncio.to_thread(
                            self._speak_with_face_tracking,
                            response, emotion, intensity, emotion_level,
                            self._stop_speaking_event
                        ))
                        
                        try:
                            while not speech_task.done():
                                if select.select([sys.stdin], [], [], 0)[0]:
                                    try:
                                        char = sys.stdin.read(1)
                                        if char == '':
                                            print("\n⏹️ Interrupting...")
                                            self._stop_speaking_event.set()
                                            break
                                    except:
                                        pass
                                await asyncio.sleep(0.05)
                            
                            await speech_task
                        except asyncio.CancelledError:
                            pass
                        
                        tts_time = time.time() - tts_start
                        self.history.add_assistant_message(response)
                        
                        if self.debug:
                            print(f"\n  ⏱️ LLM: {llm_time:.2f}s, TTS: {tts_time:.2f}s")
                    
                except KeyboardInterrupt:
                    print("\n\n👋 Goodbye!")
                    return
                except Exception as e:
                    print(f"\n⚠️ Error: {e}")
                    await asyncio.sleep(1.0)


def main():
    parser = argparse.ArgumentParser(description="Reachy Mini Chat v9 with Vision and Piper-TTS")
    
    # Vision arguments
    parser.add_argument(
        '--vision', 
        nargs='?',
        const='face',
        default=None,
        choices=['face', 'monitor'],
        help='Enable vision features: face (face tracking), monitor (security monitoring). Default: disabled'
    )
    parser.add_argument('--vision-fps', type=float, default=15.0, help='Vision processing FPS (default: 15)')
    parser.add_argument('--no-auto-wake', action='store_true', help='Disable auto-wake on person detection')
    
    # Chat arguments
    parser.add_argument('--chat', action='store_true', help='Start interactive chat')
    parser.add_argument('--asr', action='store_true', help='Use microphone ASR input')
    parser.add_argument('--model', default='qwen3:0.6b', help='Ollama model name (e.g., qwen2.5:0.5b)')
    parser.add_argument('--url', default='http://localhost:11434', help='Ollama URL')
    parser.add_argument('--piper-model', default='models/en-us-ryan-medium.onnx', help='Path to Piper .onnx model')
    parser.add_argument('--piper-config', default=None, help='Path to Piper .json config')
    parser.add_argument('--speaker', type=int, default=0, help='Speaker ID for multi-speaker models')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--gentle', action='store_true', help='Enable gentle_mode for subtle emotions')
    # History options
    parser.add_argument('--history-size', type=int, default=5, help='Conversation history size (default: 5)')
    parser.add_argument('--no-history', action='store_true', help='Disable conversation history')
    # ASR model selection
    parser.add_argument('--asr-model', default='small', choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='ASR model size: tiny=fastest, base=balanced, small=default, medium/large=slow but accurate')
    # VAD optimization
    parser.add_argument('--vad-silence', type=float, default=0.8,
                        help='VAD silence threshold in seconds (default: 0.8). Increase if speech is cut off')
    parser.add_argument('--vad-aggressive', type=int, default=1, choices=[0, 1, 2, 3],
                        help='VAD aggressiveness: 0=least aggressive, 1=gentle, 2=strict, 3=most aggressive')
    parser.add_argument('--no-vad', action='store_true', help='Disable VAD - use fixed 4s recording instead')

    args = parser.parse_args()
    
    # Determine vision mode
    vision_enabled = args.vision is not None
    vision_mode = args.vision
    
    if vision_enabled:
        if vision_mode == 'monitor':
            print(f"🔒 Security monitor mode: ENABLED")
        else:
            print(f"👁️  Vision features: ENABLED (face tracking)")

    # Needs aiohttp
    try:
        import aiohttp
    except ImportError:
        print("❌ aiohttp not found. Please install: pip install aiohttp")
        return

    if vision_enabled:
        app = ChatAppWithVision(
            vision_enabled=True,
            vision_mode=vision_mode,
            vision_fps=args.vision_fps,
            vision_auto_wake=not args.no_auto_wake,
            model=args.model,
            ollama_url=args.url,
            piper_model=args.piper_model,
            piper_config=args.piper_config,
            speaker_id=args.speaker,
            debug=args.debug,
            use_asr=args.asr,
            gentle=args.gentle,
            history_size=args.history_size,
            enable_history=not args.no_history,
            asr_model=args.asr_model,
            vad_silence=args.vad_silence,
            vad_aggressive=args.vad_aggressive,
            use_vad=not args.no_vad
        )
    else:
        app = ChatAppWithPiper(
            model=args.model,
            ollama_url=args.url,
            piper_model=args.piper_model,
            piper_config=args.piper_config,
            speaker_id=args.speaker,
            debug=args.debug,
            use_asr=args.asr,
            gentle=args.gentle,
            history_size=args.history_size,
            enable_history=not args.no_history,
            asr_model=args.asr_model,
            vad_silence=args.vad_silence,
            vad_aggressive=args.vad_aggressive,
            use_vad=not args.no_vad
        )

    app.start_chat()

if __name__ == '__main__':
    main()
