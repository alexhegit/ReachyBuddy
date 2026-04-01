#!/usr/bin/env python3
"""ASR engine using faster-whisper (CPU) with a simple mic recorder.

This module provides `FasterWhisperASREngine` which loads a faster-whisper
`WhisperModel` on CPU and exposes `transcribe_file` and
`transcribe_from_mic` helper methods.

Notes:
- Install dependencies: `pip install faster-whisper sounddevice soundfile`
- Use `model_name='small'` or smaller for reasonable CPU latency.
"""
from __future__ import annotations

import tempfile
import os
import time
from typing import Optional

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - runtime optional
    WhisperModel = None


class FasterWhisperASREngine:
    """ASR engine built on faster-whisper (CPU) with VAD support.

    Example:
        engine = FasterWhisperASREngine(model_name='small')
        # Fixed recording
        text = engine.transcribe_from_mic(4.0)
        # VAD-based recording (stops when speech ends)
        text = engine.transcribe_from_mic_vad(max_duration=4.0)
    """

    def __init__(self, model_name: str = "small", device: str = "cpu", beam_size: int = 5):
        if WhisperModel is None:
            raise RuntimeError("faster-whisper not installed. Install with `pip install faster-whisper`")

        self.model_name = model_name
        self.device = device
        self.beam_size = beam_size

        # Load model once and reuse
        self.model = WhisperModel(self.model_name, device=self.device)
        
    def configure_for_latency(self):
        """Configure for optimal latency (smaller model, faster processing)"""
        # Use tiny model for faster processing
        self.model_name = "tiny"
        self.beam_size = 1  # Smaller beam size for faster decoding
        # Note: In practice, you'd need to reload the model
        # For simplicity in this example, we'll just update settings

    def transcribe_file(self, path: str) -> str:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        segments, info = self.model.transcribe(path, beam_size=self.beam_size)
        text = "".join(segment.text for segment in segments)
        return text.strip()

    def _record_temp_wav(self, duration: float = 5.0, samplerate: int = 16000) -> str:
        try:
            import sounddevice as sd
            import soundfile as sf
        except Exception as e:
            raise RuntimeError("sounddevice and soundfile are required for recording: pip install sounddevice soundfile") from e

        channels = 1
        print(f"🎙️ Recording {duration:.1f}s @ {samplerate}Hz...")
        data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='int16')
        sd.wait()

        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp_path = tmp.name
        tmp.close()

        sf.write(tmp_path, data, samplerate=samplerate)
        return tmp_path

    def _record_temp_wav_vad(self, max_duration: float = 5.0, samplerate: int = 16000, 
                              silence_threshold: float = 2.0, aggressiveness: int = 1,
                              trailing_buffer_ms: float = 300) -> str:
        """Record using Voice Activity Detection (VAD) - stops when speech ends.
        
        Args:
            max_duration: Maximum recording duration in seconds
            samplerate: Audio sample rate
            silence_threshold: Seconds of silence before stopping
            aggressiveness: VAD aggressiveness 0-3 (0=least aggressive, 3=most aggressive)
            trailing_buffer_ms: Keep this many ms of audio before the silence (for whisper context)
        """
        try:
            import sounddevice as sd
            import soundfile as sf
            import numpy as np
        except Exception as e:
            raise RuntimeError("sounddevice, soundfile, and numpy are required for VAD recording") from e
        
        try:
            import webrtcvad
            vad = webrtcvad.Vad(aggressiveness)  # Configurable aggressiveness
        except ImportError:
            print("⚠️ webrtcvad not installed. Falling back to fixed recording.")
            return self._record_temp_wav(max_duration, samplerate)

        channels = 1
        frames = []
        print(f"🎙️ VAD Recording (max {max_duration:.1f}s) - speak now...")
        
        frame_duration_ms = 30  # WebRTC VAD works best with 10, 20, or 30ms frames
        frame_samples = int(samplerate * frame_duration_ms / 1000)
        trailing_buffer_frames = int(trailing_buffer_ms / frame_duration_ms)
        
        # Record until silence or max duration
        start_time = time.time()
        silent_frames = 0
        required_silent_frames = int(silence_threshold * 1000 / frame_duration_ms)
        speech_detected = False
        
        with sd.InputStream(samplerate=samplerate, channels=channels, dtype='int16') as stream:
            while (time.time() - start_time) < max_duration:
                data, _ = stream.read(frame_samples)
                if data is None:
                    break
                    
                frames.append(data)
                
                # Check if frame contains speech
                is_speech_frame = False
                try:
                    if vad.is_speech(data.tobytes(), samplerate):
                        silent_frames = 0  # Reset silence counter
                        speech_detected = True
                        is_speech_frame = True
                    else:
                        silent_frames += 1
                except Exception:
                    # VAD may fail for very short or malformed frames
                    silent_frames += 1
                
                # Only stop after we've detected speech AND have enough silence
                if speech_detected and silent_frames >= required_silent_frames:
                    # Keep trailing buffer: remove some trailing silent frames but keep context
                    if trailing_buffer_frames > 0 and len(frames) > trailing_buffer_frames:
                        frames = frames[:-trailing_buffer_frames]
                    print(f"🔇 Detected {silence_threshold}s of silence after speech - stopping")
                    break
        
        if not frames:
            raise RuntimeError("No audio captured")
        
        audio_data = np.concatenate(frames, axis=0)
        actual_duration = len(audio_data) / samplerate
        print(f"⏱️ Recorded {actual_duration:.2f}s (VAD stopped recording)")
        
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp_path = tmp.name
        tmp.close()
        
        sf.write(tmp_path, audio_data, samplerate=samplerate)
        return tmp_path

    def transcribe_from_mic(self, duration: float = 5.0, samplerate: int = 16000) -> Optional[str]:
        """Record from mic then transcribe; returns the transcribed text or None."""
        wav_path = None
        try:
            wav_path = self._record_temp_wav(duration, samplerate)
            text = self.transcribe_file(wav_path)
            return text
        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

    def transcribe_from_mic_vad(self, max_duration: float = 5.0, samplerate: int = 16000, 
                                 silence_threshold: float = 2.0, aggressiveness: int = 1,
                                 trailing_buffer_ms: float = 300) -> Optional[str]:
        """Record from mic using VAD then transcribe; returns the transcribed text or None.
        
        Args:
            max_duration: Maximum recording duration in seconds
            samplerate: Audio sample rate
            silence_threshold: Seconds of silence before stopping
            aggressiveness: VAD aggressiveness 0-3 (1=least aggressive/recommended, 3=most aggressive)
            trailing_buffer_ms: Keep this many ms of audio before silence for better transcription
        """
        wav_path = None
        try:
            wav_path = self._record_temp_wav_vad(max_duration, samplerate, silence_threshold,
                                                  aggressiveness, trailing_buffer_ms)
            text = self.transcribe_file(wav_path)
            return text
        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception:
                    pass


if __name__ == '__main__':
    # Quick interactive test (if faster-whisper + sounddevice installed)
    try:
        engine = FasterWhisperASREngine(model_name='small')
    except Exception as e:
        print('Initialization error:', e)
    else:
        print('Recording 4s and transcribing...')
        txt = engine.transcribe_from_mic(4.0)
        print('Transcription:', txt)
        
        print('\n--- Testing VAD recording ---')
        try:
            txt_vad = engine.transcribe_from_mic_vad(max_duration=4.0)
            print('VAD Transcription:', txt_vad)
        except Exception as e:
            print(f'VAD test error: {e}')
