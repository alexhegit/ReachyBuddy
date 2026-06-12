"""Test CheeseModeApp state machine transitions (with mocked dependencies)."""
import time
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from modes.cheese.app import CheeseModeApp, RCState
from modes.cheese.config import CheeseConfig


@pytest.fixture
def app():
    cfg = CheeseConfig(debug=False)
    a = CheeseModeApp(cfg)
    # Mock out all hardware dependencies
    a.aligner = MagicMock()
    a.voice = MagicMock()
    a.gui = MagicMock()
    a.gui.get_events.return_value = []
    a.runtime = MagicMock()
    a._robot_runtime = None
    # Default aligner response: face visible, not aligned, no tracking
    a.aligner.update.return_value = {
        "has_face": True, "aligned": False, "bbox": (0, 0, 80, 80),
        "center": (40, 40), "dx": 50, "dy": 30, "stable_frames": 0,
    }
    return a


def frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestInitialState:
    def test_starts_in_sleep(self, app):
        assert app.state == RCState.SLEEP


class TestWakeTransition:
    def test_wake_word_enters_tracking(self, app):
        app._asr_queue.put("reachy")
        app.run_frame(frame())
        assert app.state == RCState.TRACKING

    def test_random_voice_does_not_wake(self, app):
        app._asr_queue.put("hello world")
        app.run_frame(frame())
        assert app.state == RCState.SLEEP

    def test_wake_clears_no_face_timer(self, app):
        app._asr_queue.put("reachy")
        app.run_frame(frame())
        assert app._tracking_no_face_since == 0.0


class TestSleepCommand:
    def test_sleep_from_tracking(self, app):
        app.state = RCState.TRACKING
        app._asr_queue.put("sleep")
        app.run_frame(frame())
        assert app.state == RCState.SLEEP

    def test_sleep_from_armed(self, app):
        app.state = RCState.ARMED
        app._asr_queue.put("go to sleep")
        app.run_frame(frame())
        assert app.state == RCState.SLEEP

    def test_sleep_not_from_sleep(self, app):
        """Saying 'sleep' while already sleeping should stay in sleep."""
        app.state = RCState.SLEEP
        app._asr_queue.put("sleep")
        app.run_frame(frame())
        assert app.state == RCState.SLEEP


class TestCaptureCommand:
    def test_capture_from_armed(self, app):
        app.state = RCState.ARMED
        app._asr_queue.put("cheese")
        app.run_frame(frame())
        assert app.state == RCState.COUNTDOWN

    def test_capture_not_from_tracking(self, app):
        app.state = RCState.TRACKING
        app._asr_queue.put("cheese")
        app.run_frame(frame())
        assert app.state == RCState.TRACKING


class TestTrackingNoFaceTimeout:
    def test_timeout_triggers_sleep(self, app):
        app.state = RCState.TRACKING
        app.aligner.update.return_value = {
            "has_face": False, "aligned": False, "bbox": None,
            "center": None, "dx": 0, "dy": 0, "stable_frames": 0,
        }
        app._tracking_no_face_since = 0.0
        app.run_frame(frame())
        # First frame without face sets the timer
        assert app._tracking_no_face_since > 0
        assert app.state == RCState.TRACKING

        # Simulate 31 seconds passing
        app._tracking_no_face_since -= 31.0
        app.run_frame(frame())
        assert app.state == RCState.SLEEP

    def test_face_reappears_resets_timer(self, app):
        app.state = RCState.TRACKING
        app._tracking_no_face_since = 10.0
        app.aligner.update.return_value = {
            "has_face": True, "aligned": False, "bbox": (0, 0, 50, 50),
            "center": (25, 25), "dx": 1, "dy": 1, "stable_frames": 0,
        }
        app.run_frame(frame())
        assert app._tracking_no_face_since == 0.0


class TestArmedTimeout:
    def test_armed_timeout_returns_to_sleep(self, app):
        app.state = RCState.ARMED
        app._armed_since = 0.0
        app.aligner.update.return_value = {
            "has_face": True, "aligned": True, "bbox": (0, 0, 50, 50),
            "center": (25, 25), "dx": 1, "dy": 1, "stable_frames": 20,
        }
        # Move _armed_since far enough back
        app._armed_since = time.time() - app.cfg.command_timeout_s - 1.0
        app.run_frame(frame())
        assert app.state == RCState.SLEEP

    def test_armed_loses_face_goes_to_tracking(self, app):
        app.state = RCState.ARMED
        app.aligner.update.return_value = {
            "has_face": False, "aligned": False, "bbox": None,
            "center": None, "dx": 0, "dy": 0, "stable_frames": 0,
        }
        app.run_frame(frame())
        assert app.state == RCState.TRACKING
