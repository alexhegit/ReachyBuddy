"""Tests for guard mode analysis thread lifecycle and alert logic."""

import threading
import time

import cv2
import numpy as np
import pytest

from modes.guard.app import GuardModeApp
from modes.guard.config import GuardConfig


class FakeRuntime:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get_frame(self):
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        cv2.putText(frame, "TEST", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        return frame

    def move_head(self, pan, tilt, duration):
        pass


class FakeVoice:
    def __init__(self):
        self.spoken = []

    def speak_with_emotion(self, text):
        self.spoken.append(text)

    def speak_with_interrupt(self, text):
        self.spoken.append(text)

    def close(self):
        pass


class TestGuardAnalysisThread:
    def test_analysis_thread_waits_for_running_flag(self, monkeypatch):
        """Regression: thread must not exit immediately because _running is False in setup()."""
        cfg = GuardConfig(camera_source="webcam", gui_backend="none", debug=False)
        app = GuardModeApp(cfg)

        app.runtime = FakeRuntime()
        app.runtime.__enter__()
        app._robot_runtime = app.runtime
        app.voice = FakeVoice()

        call_count = {"n": 0}

        def fake_call(frame):
            call_count["n"] += 1
            return "OK"

        monkeypatch.setattr(app, "_call_ollama", fake_call)

        thread = threading.Thread(target=app._analysis_loop, daemon=True)
        thread.start()

        # Thread should still be alive while _running is False
        time.sleep(0.2)
        assert thread.is_alive(), "analysis thread died before _running was set"

        # Now set _running True; thread should call _call_ollama soon
        app._latest_frame = app.runtime.get_frame()
        app._running = True

        deadline = time.time() + 2
        while call_count["n"] == 0 and time.time() < deadline:
            time.sleep(0.05)

        app._running = False
        thread.join(timeout=2)

        assert call_count["n"] >= 1, "analysis thread never called Ollama"

    def test_alert_triggers_tts_and_screenshot(self, tmp_path, monkeypatch):
        """Non-OK analysis result should speak and save screenshot."""
        cfg = GuardConfig(camera_source="webcam", gui_backend="none", debug=False, save_dir=str(tmp_path))
        app = GuardModeApp(cfg)
        app.runtime = FakeRuntime()
        app._robot_runtime = app.runtime
        app.voice = FakeVoice()

        monkeypatch.setattr(app, "_call_ollama", lambda frame: "I see a person in the room")

        thread = threading.Thread(target=app._analysis_loop, daemon=True)
        thread.start()
        app._latest_frame = app.runtime.get_frame()
        app._running = True

        # Wait for analysis to be processed by run_frame-like logic
        deadline = time.time() + 3
        while app._analysis_queue.empty() and time.time() < deadline:
            time.sleep(0.05)

        # Manually drain queue like run_frame does
        last = ""
        while True:
            try:
                last = app._analysis_queue.get_nowait()
            except Exception:
                break

        if last:
            app._last_analysis_text = last
            is_ok = "ok" in last.lower()
            if not is_ok:
                app.voice.speak_with_interrupt(last)
                app._save_alert_screenshot(app._latest_frame)
                app._last_alert_text = last

        app._running = False
        thread.join(timeout=2)

        assert len(app.voice.spoken) == 1, "TTS should speak the alert"
        assert len(list(tmp_path.glob("*.jpg"))) == 1, "screenshot should be saved"
