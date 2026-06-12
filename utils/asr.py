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
import threading
from typing import Optional

import numpy as np

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
        if self.model_name != "tiny":
            self.model_name = "tiny"
            self.beam_size = 1
            self.model = WhisperModel(self.model_name, device=self.device)

    def transcribe_file(self, path: str) -> str:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        segments, info = self.model.transcribe(path, beam_size=self.beam_size)
        text = "".join(segment.text for segment in segments)
        return text.strip()

    def _record_temp_wav(self, duration: float = 5.0, samplerate: int = 16000,
                          show_volume: bool = True) -> str:
        """Record audio for fixed duration with optional volume visualization.

        Args:
            duration: Recording duration in seconds
            samplerate: Audio sample rate
            show_volume: Whether to show real-time volume visualization
        """
        try:
            import sounddevice as sd
            import soundfile as sf
            import numpy as np
        except Exception as e:
            raise RuntimeError("sounddevice and soundfile are required for recording: pip install sounddevice soundfile") from e

        channels = 1
        print(f"🎙️ Recording {duration:.1f}s @ {samplerate}Hz...")

        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp_path = tmp.name
        tmp.close()

        if show_volume:
            # Record with volume visualization
            frames = []
            chunk_duration = 0.05  # 50ms chunks for smooth visualization
            chunk_samples = int(samplerate * chunk_duration)
            total_chunks = int(duration / chunk_duration)
            stream = None

            try:
                stream = sd.InputStream(samplerate=samplerate, channels=channels, dtype='int16')
                stream.start()

                for _ in range(total_chunks):
                    data, _ = stream.read(chunk_samples)
                    frames.append(data)

                    # Calculate and display volume
                    rms = self._calculate_rms(data)
                    bar = self._draw_volume_bar(rms)
                    print(f"\r{bar}", end='', flush=True)

                print()  # New line after volume bar
                data = np.concatenate(frames, axis=0)
                sf.write(tmp_path, data, samplerate=samplerate)
            finally:
                if stream is not None:
                    try:
                        stream.stop()
                        stream.close()
                    except Exception:
                        pass
        else:
            # Simple recording without visualization
            try:
                data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='int16')
                sd.wait()
                sf.write(tmp_path, data, samplerate=samplerate)
            except Exception:
                # Ensure any partial recording is cleaned up
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except:
                    pass
                raise

        return tmp_path

    @staticmethod
    def _calculate_rms(audio_data: np.ndarray) -> float:
        """Calculate RMS (Root Mean Square) of audio data for volume level."""
        return np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))

    @staticmethod
    def _draw_volume_bar(rms: float, max_width: int = 30) -> str:
        """Draw ASCII volume bar based on RMS level.

        Args:
            rms: RMS value (0-32767 for int16 audio)
            max_width: Maximum width of the bar in characters

        Returns:
            ASCII bar string like "[████████░░░░] -45dB"
        """
        # Convert RMS to approximate dB (relative to full scale)
        # 32767 is max for int16, so dB = 20*log10(rms/32767)
        if rms < 1:
            db = -60
        else:
            db = 20 * np.log10(rms / 32767)

        # Normalize to 0-1 range (-60dB to 0dB)
        normalized = max(0, min(1, (db + 60) / 60))
        filled = int(normalized * max_width)

        # Create bar with different characters for visual interest
        bar_chars = "█" * filled + "░" * (max_width - filled)

        return f"[{bar_chars}] {db:+.1f}dB"

    def _record_temp_wav_vad(self, max_duration: float = 5.0, samplerate: int = 16000,
                              silence_threshold: float = 2.0, aggressiveness: int = 1,
                              trailing_buffer_ms: float = 300,
                              show_volume: bool = True,
                              stop_event: threading.Event | None = None) -> str:
        """Record using Voice Activity Detection (VAD) - stops when speech ends.

        Args:
            max_duration: Maximum recording duration in seconds
            samplerate: Audio sample rate
            silence_threshold: Seconds of silence before stopping
            aggressiveness: VAD aggressiveness 0-3 (0=least aggressive, 3=most aggressive)
            trailing_buffer_ms: Keep this many ms of audio before the silence (for whisper context)
            show_volume: Whether to show real-time volume visualization
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
        stream = None

        try:
            stream = sd.InputStream(samplerate=samplerate, channels=channels, dtype='int16')
            stream.start()

            while (time.time() - start_time) < max_duration:
                if stop_event is not None and stop_event.is_set():
                    if show_volume:
                        print()
                    print("🔇 Recording cancelled by stop event")
                    break
                data, _ = stream.read(frame_samples)
                if data is None:
                    break

                frames.append(data)

                # Calculate and display volume if enabled
                if show_volume:
                    rms = self._calculate_rms(data)
                    bar = self._draw_volume_bar(rms)
                    # Use \r to return to start of line, \033[K to clear to end
                    print(f"\r{bar}", end='', flush=True)

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
                    if show_volume:
                        print()  # New line after volume bar
                    print(f"🔇 Detected {silence_threshold}s of silence after speech - stopping")
                    break
        finally:
            # Ensure stream is properly stopped and closed
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass

        if not frames:
            raise RuntimeError("No audio captured")

        # Clear volume bar line if we were showing it
        if show_volume:
            print()

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

    def close(self) -> None:
        """Cleanup audio resources."""
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass

    def transcribe_from_mic_vad(self, max_duration: float = 5.0, samplerate: int = 16000,
                                 silence_threshold: float = 2.0, aggressiveness: int = 1,
                                 trailing_buffer_ms: float = 300,
                                 show_volume: bool = True,
                                 stop_event: threading.Event | None = None) -> Optional[str]:
        """Record from mic using VAD then transcribe; returns the transcribed text or None.

        Args:
            max_duration: Maximum recording duration in seconds
            samplerate: Audio sample rate
            silence_threshold: Seconds of silence before stopping
            aggressiveness: VAD aggressiveness 0-3 (1=least aggressive/recommended, 3=most aggressive)
            trailing_buffer_ms: Keep this many ms of audio before silence for better transcription
            show_volume: Whether to show real-time volume visualization
            stop_event: Optional event that, when set, cancels the recording early
        """
        wav_path = None
        try:
            wav_path = self._record_temp_wav_vad(max_duration, samplerate, silence_threshold,
                                                  aggressiveness, trailing_buffer_ms, show_volume,
                                                  stop_event=stop_event)
            if not wav_path:
                return None
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
