import time
"""Piper-TTS engine wrapper for offline speech synthesis."""

import os
import json
import wave
import tempfile
import threading
import asyncio
from typing import Optional, Tuple

import soundfile as sf
import sounddevice as sd
import numpy as np


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
                print(f"\nExample: python main.py --cheese --piper-model {found_models[0]}")
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

    def close(self):
        """Cleanup TTS resources."""
        try:
            sd.stop()
        except Exception:
            pass
        self.voice = None

    async def speak_with_emotion_async(self, text: str, emotion: str = 'neutral'):
        """Async version of speak_with_emotion (runs in thread)."""
        # Piper synthesis is CPU bound, so run in a separate thread
        await asyncio.to_thread(self.speak_with_emotion, text, emotion)

    def synthesize_to_buffer(self, text: str) -> Optional[Tuple[np.ndarray, int]]:
        """Synthesize text to audio buffer without playing.

        Returns:
            Tuple of (audio_data, sample_rate) or None if failed
        """
        if not text.strip() or not self.voice:
            return None

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name

            with wave.open(tmp_path, "wb") as wav_file:
                syn_config = None
                if self.SynthesisConfig and self.speaker_id is not None:
                    syn_config = self.SynthesisConfig(speaker_id=self.speaker_id)
                self.voice.synthesize_wav(text, wav_file, syn_config=syn_config)

            data, sr = sf.read(tmp_path, dtype='float32')
            return (data, sr) if data.size > 0 else None

        except Exception as e:
            if self.debug:
                print(f"⚠️ Synthesis error: {e}")
            return None
        finally:
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except:
                pass

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
