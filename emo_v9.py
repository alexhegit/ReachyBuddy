#!/usr/bin/env python3
"""emo_v9.py - Reachy Mini Chat v9 (Development)

Incremental improvements over v8:
- Step 1: Fix EmotionControllerV71 inheritance (avoid EdgeTTSEngine creation)
- Step 2: (Optional) Add conversation history/context
- Step 3: (Optional) Add performance timing statistics

Usage:
  python emo_v9.py --piper-model models/en_US-lessac-high.onnx --asr
  python emo_v9.py --debug  # Show detailed logs

Development workflow:
  1. Make small change
  2. Test thoroughly  
  3. Commit
  4. Next feature
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

        # Get available moves for this category
        available = list(self.emotion_to_moves.get(category, []))
        if not available:
            available = list(self.emotion_to_moves.get('neutral', []))

        # Filter gentle moves if in gentle mode
        if self.gentle_mode:
            gentle_names = ['calming1', 'serenity1', 'thoughtful1', 'thoughtful2',
                            'attentive1', 'attentive2']
            gentle_moves = [(lib, name) for lib, name in available if name in gentle_names]
            if gentle_moves:
                available = gentle_moves
            else:
                all_gentle = self._get_all_gentle_moves()
                if all_gentle:
                    available = all_gentle
                else:
                    available = []
        else:
            # Non-gentle: boost variety by also pulling from other categories
            # if the primary category is running low on fresh moves
            other_moves = []
            if used_moves:
                unused_in_primary = [m for m in available if m not in used_moves]
                if len(unused_in_primary) < 3:
                    for cat, moves in self.emotion_to_moves.items():
                        if cat != category:
                            other_moves.extend(moves)
            if other_moves:
                available = available + other_moves

        if not available:
            return None, intensity, 1.0

        # Prefer moves not yet used this round
        pool = available
        if used_moves:
            fresh = [m for m in available if m not in used_moves]
            if fresh:
                pool = fresh

        # Also avoid the immediate last move if possible
        if avoid_move and avoid_move in pool and len(pool) > 1:
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
                move_counter = 0
                while not tts_done.is_set():
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
                        # No move available, fallback to simple actions (gentle if needed)
                        if self.gentle_mode:
                            print("   🎬 Gentle simple action")
                            self._simple_thoughtful_tilt_once()
                        else:
                            print("   🎬 Simple action fallback")
                            self._simple_nod_once()
                        time.sleep(0.8)

                    move_counter += 1

                    # Add explicit body yaw rotation between moves so the robot turns
                    # (most recorded moves from dances_lib have body_yaw=0)
                    if not tts_done.is_set() and move_counter % 2 == 1:
                        try:
                            import random
                            angle = random.choice([-0.4, -0.2, 0.2, 0.4])
                            self.reachy.goto_target(body_yaw=angle, duration=0.4)
                            time.sleep(0.45)
                            self.reachy.goto_target(body_yaw=0.0, duration=0.4)
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


def main():
    parser = argparse.ArgumentParser(description="Reachy Mini Chat v9 with Piper-TTS and History")
    parser.add_argument('--chat', action='store_true', help='Start interactive chat')
    parser.add_argument('--asr', action='store_true', help='Use microphone ASR input')
    parser.add_argument('--model', default='qwen3:0.6b', help='Ollama model name (e.g., qwen2.5:0.5b)')
    parser.add_argument('--url', default='http://localhost:11434', help='Ollama URL')
    parser.add_argument('--piper-model', default='models/en-us-ryan-medium.onnx', help='Path to Piper .onnx model')
    parser.add_argument('--piper-config', default=None, help='Path to Piper .json config')
    parser.add_argument('--speaker', type=int, default=0, help='Speaker ID for multi-speaker models')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--gentle', action='store_true', help='Enable gentle_mode for subtle emotions')
    # Step 2: History options
    parser.add_argument('--history-size', type=int, default=5, help='Conversation history size (default: 5)')
    parser.add_argument('--no-history', action='store_true', help='Disable conversation history')
    # Step 4 B: ASR model selection
    parser.add_argument('--asr-model', default='small', choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='ASR model size: tiny=fastest, base=balanced, small=default, medium/large=slow but accurate')
    # VAD optimization: dynamic silence detection
    parser.add_argument('--vad-silence', type=float, default=0.8,
                        help='VAD silence threshold in seconds (default: 0.8). Increase if speech is cut off')
    parser.add_argument('--vad-aggressive', type=int, default=1, choices=[0, 1, 2, 3],
                        help='VAD aggressiveness: 0=least aggressive (more false positives), 1=gentle(recommended), 2=strict, 3=most aggressive (may cut speech)')
    parser.add_argument('--no-vad', action='store_true',
                        help='Disable VAD - use fixed 4s recording instead')

    args = parser.parse_args()
    
    # Needs aiohttp
    try:
        import aiohttp
    except ImportError:
        print("❌ aiohttp not found. Please install: pip install aiohttp")
        return

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
