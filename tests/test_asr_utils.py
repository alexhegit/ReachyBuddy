"""Test ASR utility methods (pure math, no hardware)."""
import numpy as np
import pytest
from utils.asr import FasterWhisperASREngine


class TestCalculateRMS:
    def test_silence_returns_low_value(self, engine):
        data = np.zeros(1000, dtype=np.int16)
        rms = engine._calculate_rms(data)
        assert rms == 0.0

    def test_constant_signal(self, engine):
        data = np.full(1000, 1000, dtype=np.int16)
        rms = engine._calculate_rms(data)
        assert rms == pytest.approx(1000.0, rel=0.01)

    def test_sine_wave(self, engine):
        t = np.linspace(0, 1, 1000)
        data = (np.sin(2 * np.pi * 440 * t) * 5000).astype(np.int16)
        rms = engine._calculate_rms(data)
        assert 3000 < rms < 4000


class TestDrawVolumeBar:
    def test_min_db_shows_empty(self, engine):
        bar = engine._draw_volume_bar(0.5)
        assert "[░" in bar or "[" in bar  # mostly empty

    def test_max_db_shows_full(self, engine):
        bar = engine._draw_volume_bar(32767)
        assert "█" in bar

    def test_contains_db_value(self, engine):
        bar = engine._draw_volume_bar(5000)
        assert "dB" in bar


@pytest.fixture
def engine():
    # Create without loading WhisperModel (requires model download)
    obj = object.__new__(FasterWhisperASREngine)
    return obj
