"""Guard mode - multi-modal security monitoring via Ollama VLM."""

from __future__ import annotations

import base64
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests

from core.base_app import BaseModeApp
from core.runtime import create_runtime, find_reachy_camera

from .config import GuardConfig
from .gui import GuardGUI


class GuardModeApp(BaseModeApp):
    """Guard mode: periodic VLM analysis with head scanning and voice alerts."""

    def __init__(self, config: GuardConfig):
        super().__init__(config)
        self.cfg: GuardConfig = config

        # Thread-safe frame buffer
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None

        # Analysis state
        self._analysis_queue: "queue.Queue[str]" = queue.Queue()
        self._last_alert_text = ""
        self._last_alert_at = 0.0
        self._last_analysis_text = ""

        # Scanning state
        self._scan_pan = 0.0
        self._scan_dir = 1
        self._last_scan_at = 0.0
        self._last_head_at = 0.0

        # Screenshot counter
        self._shot_count = 0

        # Runtime tracking (Reachy robot vs camera-only)
        self._robot_runtime = None
        self._camera_runtime = None

    # ── BaseModeApp interface ──────────────────────────────────────

    def get_mode_name(self) -> str:
        return "guard"

    def get_requirements(self) -> list:
        return [
            "numpy",
            "opencv-python",
            "requests",
            "pillow",
        ]

    def setup(self) -> None:
        print("🔍 Initializing Guard mode...")

        # Camera/runtime — try Reachy first, fall back to webcam for frames
        device = find_reachy_camera()
        self.runtime = create_runtime(
            camera_source=self.cfg.camera_source,
            camera_index=self.cfg.camera_index,
            reachy_device_path=device,
            reachy_host=self.cfg.reachy_host,
            reachy_port=self.cfg.reachy_port,
        )

        self._reachy_connected = False
        self._camera_runtime = None

        if self.cfg.camera_source == "reachy":
            try:
                self.runtime.__enter__()
                self._reachy_connected = True
                self._robot_runtime = self.runtime
                print("✅ Reachy daemon connected (robot control available)")
            except Exception as e:
                print(f"⚠️ Reachy init failed: {e}")
                print("   ↪ Falling back to webcam-only mode")
                self.runtime = create_runtime("webcam", self.cfg.camera_index, reachy_device_path=device)
                self.runtime.__enter__()

            # If Reachy is connected but media is unavailable, open camera directly
            if self._reachy_connected:
                try:
                    frame = self.runtime.get_frame()
                except Exception:
                    frame = None
                if frame is None:
                    cam_dev = find_reachy_camera()
                    if cam_dev:
                        print(f"ℹ️ Reachy media unavailable; using camera device {cam_dev} for frames")
                        camera_rt = create_runtime("webcam", self.cfg.camera_index, reachy_device_path=cam_dev)
                        camera_rt.__enter__()
                        self._camera_runtime = camera_rt
                        self.runtime = camera_rt
                        print("✅ Camera runtime initialized for frames; robot control kept")
                        # Apply basic V4L2 tuning for Reachy camera (no ISP → dark default)
                        self._tune_camera(cam_dev)
                    else:
                        print("⚠️ No camera device found; frames may be unavailable")
        else:
            # Webcam mode
            try:
                self.runtime.__enter__()
            except Exception as e:
                print(f"❌ Webcam init failed: {e}")
                raise

        # Save directory
        save_path = Path(self.cfg.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Alerts saved to: {save_path}")

        # TTS
        from utils.tts_engine import PiperTTSEngine
        self.voice = PiperTTSEngine(
            model_path=self.cfg.piper_model,
            config_path=self.cfg.piper_config,
            speaker_id=self.cfg.speaker_id,
        )
        self.voice.speak_with_emotion("Guard mode activated.")

        # GUI
        self.gui = GuardGUI(self.cfg)

        # Start analysis thread
        self._analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self._analysis_thread.start()

        print(f"🤖 Guard running — Ollama: {self.cfg.ollama_url} model: {self.cfg.ollama_model}")

        # Quick connectivity/model sanity check
        self._verify_ollama()

    def run_frame(self, frame: np.ndarray) -> bool:
        # Store latest frame for analysis thread
        with self._frame_lock:
            self._latest_frame = frame.copy()

        # Debug: log frame stats on first frame (and every 100th)
        if self.cfg.debug:
            frame_mean = frame.mean()
            if getattr(self, '_frame_debug_count', 0) % 100 == 0:
                print(f"   📷 Frame: shape={frame.shape} mean={frame_mean:.1f} min={frame.min()} max={frame.max()}")
            self._frame_debug_count = getattr(self, '_frame_debug_count', 0) + 1

        # Head scanning
        self._update_scan(frame.shape[:2])

        # Check for alerts
        last_result = ""
        try:
            while True:
                last_result = self._analysis_queue.get_nowait()
        except queue.Empty:
            pass

        if last_result:
            self._last_analysis_text = last_result
            is_ok = "ok" in last_result.lower()
            if is_ok:
                print(f"   ✅ Scene OK")
            else:
                now = time.time()
                same_alert = (last_result == self._last_alert_text
                              and now - self._last_alert_at < self.cfg.alert_cooldown)
                if same_alert:
                    print(f"   🔕 Alert on cooldown")
                else:
                    print(f"   🚨 Alert: {last_result[:80]}")
                    self.voice.speak_with_interrupt(last_result)
                    self._save_alert_screenshot(frame)
                    self._last_alert_text = last_result
                    self._last_alert_at = now

        # Build status line
        status = self._last_analysis_text[:60] if self._last_analysis_text else "Waiting for analysis..."

        # Render GUI
        if self.gui and self.gui.available:
            self.gui.draw(frame, status, self._scan_pan)

        return True

    def cleanup(self) -> None:
        if self._camera_runtime:
            self._camera_runtime.__exit__(None, None, None)
        if self.runtime:
            self.runtime.__exit__(None, None, None)
        if self.gui:
            self.gui.close()
        if self.voice:
            self.voice.close()

    # ── Internal helpers ───────────────────────────────────────────

    def _update_scan(self, frame_shape):
        """Periodically move head in a sweeping motion."""
        if not self.cfg.scan_enabled:
            return
        now = time.time()
        if now - self._last_scan_at < self.cfg.scan_interval:
            return
        self._last_scan_at = now

        self._scan_pan += self._scan_dir * self.cfg.scan_speed
        if abs(self._scan_pan) > self.cfg.scan_range:
            self._scan_dir *= -1
            self._scan_pan = max(-self.cfg.scan_range,
                                 min(self.cfg.scan_range, self._scan_pan))

        robot_rt = self._robot_runtime or self.runtime
        if robot_rt:
            try:
                robot_rt.move_head(pan=self._scan_pan, tilt=0.0,
                                   duration=self.cfg.scan_interval + 0.05)
            except Exception as e:
                if self.cfg.debug:
                    print(f"   ⚠️ Scan move error: {e}")

    def _verify_ollama(self):
        """Check that Ollama is reachable and the configured model exists.

        Uses /api/tags so no model is loaded into VRAM.
        """
        proxies = {"http": None, "https": None}
        try:
            resp = requests.get(f"{self.cfg.ollama_url}/api/tags", timeout=10, proxies=proxies)
            resp.raise_for_status()
            models = {m.get("name") for m in resp.json().get("models", [])}
            if self.cfg.ollama_model not in models:
                print(f"   ⚠️ Model '{self.cfg.ollama_model}' not found in Ollama. Run: ollama pull {self.cfg.ollama_model}")
                return
            print(f"   ✅ Ollama model '{self.cfg.ollama_model}' is available (will load on first analysis)")
        except Exception as e:
            print(f"   ⚠️ Cannot reach Ollama at {self.cfg.ollama_url}: {e}")
            if "127.0.0.1" in str(e) or "localhost" in self.cfg.ollama_url:
                print("      If you have HTTP_PROXY set, try: export NO_PROXY=localhost,127.0.0.1")
            return

        # Optional: lightweight text ping to confirm the model runs (debug only).
        if self.cfg.debug:
            try:
                resp = requests.post(
                    f"{self.cfg.ollama_url}/api/chat",
                    json={
                        "model": self.cfg.ollama_model,
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": False,
                        "keep_alive": 0,
                    },
                    timeout=60,
                    proxies=proxies,
                )
                if resp.status_code != 200:
                    print(f"   ⚠️ Ollama text chat returned HTTP {resp.status_code}: {resp.text[:120]}")
            except Exception as e:
                print(f"   ⚠️ Ollama text chat failed: {e}")

    def _analysis_loop(self):
        """Background thread: periodically send frame to Ollama."""
        print("   🧵 Analysis thread started, waiting for app to run...")
        while not getattr(self, '_running', False):
            time.sleep(0.05)
        print("   🧵 Analysis thread active")

        first = True
        while getattr(self, '_running', False):
            time.sleep(0.5 if first else self.cfg.analysis_interval)
            first = False

            with self._frame_lock:
                if self._latest_frame is None:
                    if self.cfg.debug:
                        print("   ⏳ No frame available yet, skipping analysis")
                    continue
                frame = self._latest_frame.copy()

            try:
                if self.cfg.debug:
                    print(f"   🧠 Sending frame to Ollama ({self.cfg.ollama_model})...")
                text = self._call_ollama(frame)
                if text:
                    self._analysis_queue.put(text)
                    print(f"   🧠 Analysis: {text[:100]}")
            except requests.exceptions.ConnectionError:
                print(f"   ⚠️ Cannot reach Ollama at {self.cfg.ollama_url}")
            except Exception as e:
                # Always print analysis errors so users can diagnose VLM issues
                print(f"   ⚠️ Analysis error: {e}")

        print("   🧵 Analysis thread stopped")

    def _call_ollama(self, frame: np.ndarray) -> str:
        """Send frame to Ollama VLM and return text response."""
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf).decode('utf-8')

        # Bypass HTTP(S)_PROXY for local Ollama; proxies break localhost calls.
        proxies = {"http": None, "https": None}
        resp = requests.post(
            f"{self.cfg.ollama_url}/api/chat",
            json={
                "model": self.cfg.ollama_model,
                "messages": [{
                    "role": "user",
                    "content": self.cfg.prompt,
                    "images": [b64],
                }],
                "stream": False,
            },
            timeout=60,
            proxies=proxies,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "").strip()

    def _save_alert_screenshot(self, frame: np.ndarray):
        """Save a screenshot when an alert triggers."""
        save_path = Path(self.cfg.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._shot_count += 1
        fname = save_path / f"alert_{ts}_{self._shot_count:03d}.jpg"
        cv2.imwrite(str(fname), frame)
        if self.cfg.debug:
            print(f"📸 Alert screenshot saved: {fname}")

    @staticmethod
    def _tune_camera(device: str):
        """Apply basic V4L2 tuning for Reachy camera (no hardware ISP)."""
        import subprocess
        # Load saved profile if available
        from core.runtime import load_camera_profile, PROFILE_DIR
        default_profile = PROFILE_DIR / "default.json"
        if default_profile.exists():
            if load_camera_profile("default", device):
                print(f"✅ Loaded camera profile: default")
                return
        # No profile found; set aggressive defaults for the dark Reachy camera
        params = {
            "brightness": 20,
            "contrast": 10,
            "gain": 120,
            "saturation": 60,
            "gamma": 140,
        }
        ctrl_str = ",".join(f"{k}={v}" for k, v in params.items())
        try:
            result = subprocess.run(
                ["v4l2-ctl", "-d", device, "--set-ctrl", ctrl_str],
                capture_output=True, timeout=5,
            )
            if result.returncode == 0:
                print(f"📷 Applied camera tuning: {ctrl_str}")
            else:
                print(f"⚠️ Camera tuning returned {result.returncode}: {result.stderr.decode().strip()}")
        except Exception as e:
            print(f"⚠️ Camera tuning failed: {e}")
        # Show current values
        try:
            for param in ["brightness", "gain", "exposure_time_absolute"]:
                r = subprocess.run(
                    ["v4l2-ctl", "-d", device, "--get-ctrl", param],
                    capture_output=True, timeout=5, text=True,
                )
                if r.returncode == 0:
                    print(f"   {param}: {r.stdout.strip()}")
        except Exception:
            pass
