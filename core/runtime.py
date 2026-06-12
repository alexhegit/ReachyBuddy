"""Robot/Webcam runtime abstractions."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

try:
    from reachy_mini import ReachyMini
    from reachy_mini.utils import create_head_pose
except Exception:
    ReachyMini = None

    def create_head_pose(*args, **kwargs):
        return None


# Default camera device and profile directory
DEFAULT_CAMERA_DEVICE = "/dev/video0"
PROFILE_DIR = Path.home() / ".config" / "reachy_mini"

# Reachy camera identifiers (name patterns and VID:PID)
REACHY_CAMERA_PATTERNS = ["reachy", "arducam"]
REACHY_CAMERAS_VID_PID = [
    (0x38fb, 0x1002),  # Reachy Mini Lite
    (0x1bcf, 0x28c4),  # Older RPi Camera
    (0x0c45, 0x636d),  # Arducam
]


def find_reachy_camera() -> Optional[str]:
    """Auto-detect Reachy camera device path.

    Returns:
        Device path (e.g., '/dev/video0') or None if not found
    """
    try:
        # Method 1: Check by device name using v4l2-ctl
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            reachy_section = False
            for line in lines:
                line_stripped = line.strip()
                # Check if this is a Reachy camera section header
                if any(pattern in line_stripped.lower() for pattern in REACHY_CAMERA_PATTERNS):
                    reachy_section = True
                    continue
                # If we're in Reachy section and find a video device
                if reachy_section and "/dev/video" in line_stripped:
                    device = line_stripped.split("(")[0].strip()
                    # Verify this device supports controls
                    try:
                        test_result = subprocess.run(
                            ["v4l2-ctl", "-d", device, "--get-ctrl", "brightness"],
                            capture_output=True,
                            timeout=2
                        )
                        if test_result.returncode == 0:
                            return device
                    except Exception:
                        pass
                    continue
                # Empty line or new section ends Reachy section
                if reachy_section and (not line_stripped or ":" in line_stripped):
                    reachy_section = False

        # Method 2: Check via /sys/class/video4linux
        sys_video_path = Path("/sys/class/video4linux")
        if sys_video_path.exists():
            for device_dir in sorted(sys_video_path.glob("video*")):
                name_file = device_dir / "name"
                if name_file.exists():
                    name = name_file.read_text().strip()
                    if any(pattern in name.lower() for pattern in REACHY_CAMERA_PATTERNS):
                        device = f"/dev/{device_dir.name}"
                        # Verify it supports controls
                        try:
                            test_result = subprocess.run(
                                ["v4l2-ctl", "-d", device, "--get-ctrl", "brightness"],
                                capture_output=True,
                                timeout=2
                            )
                            if test_result.returncode == 0:
                                return device
                        except Exception:
                            pass
    except Exception:
        pass
    return None


class RobotRuntime:
    """Abstract base for robot/webcam runtime."""

    def get_frame(self) -> Optional[np.ndarray]:
        """Get current camera frame."""
        raise NotImplementedError

    def look_at_image(self, x: int, y: int, duration: float = 0.2,
                       frame_width: int = 640, frame_height: int = 480,
                       pan_gain: float = 0.3, tilt_gain: float = 0.2,
                       pan_invert: bool = False) -> None:
        """Make robot look at image coordinates."""
        raise NotImplementedError

    def goto_body_yaw(self, yaw: float, duration: float = 0.35) -> None:
        """Set body yaw."""
        raise NotImplementedError

    def move_head(self, pan: float = 0.0, tilt: float = 0.0, duration: float = 0.5) -> None:
        """Set head pan/tilt in radians directly."""
        raise NotImplementedError

    def reset_head(self, duration: float = 0.2) -> None:
        """Reset head to neutral position."""
        raise NotImplementedError

    def set_automatic_body_yaw(self, enabled: bool) -> None:
        """Enable/disable automatic body yaw."""
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def load_camera_profile(profile_name: str, device: Optional[str] = None) -> bool:
    """Load camera parameters from a saved profile.

    Args:
        profile_name: Name of the profile to load
        device: Camera device path (auto-detected if None)

    Returns:
        True if profile was loaded successfully
    """
    # Auto-detect Reachy camera if device not specified
    if device is None or device == DEFAULT_CAMERA_DEVICE:
        detected = find_reachy_camera()
        if detected:
            device = detected
        else:
            device = DEFAULT_CAMERA_DEVICE

    profile_path = PROFILE_DIR / f"{profile_name}.json"
    if not profile_path.exists():
        return False

    try:
        with open(profile_path, "r") as f:
            data = json.load(f)

        params = data.get("params", {})
        if not params:
            return False

        # Use v4l2-ctl to apply parameters
        ctrl_str = ",".join([f"{k}={v}" for k, v in params.items()])
        result = subprocess.run(
            ["v4l2-ctl", "-d", device, "--set-ctrl", ctrl_str],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


class ReachyRuntime(RobotRuntime):
    """Runtime for physical Reachy Mini robot."""

    def __init__(self, camera_profile: Optional[str] = None, host: Optional[str] = None, port: Optional[int] = None):
        self._ctx = None
        self._reachy = None
        self._camera_profile = camera_profile
        # Host/port for Reachy daemon connection (defaults handled by ReachyMini)
        self._host = host
        self._port = port

    @staticmethod
    def _normalize_host(host: Optional[str]) -> Optional[str]:
        """Normalize common daemon host inputs for client use."""
        if host == "0.0.0.0":
            return "127.0.0.1"
        return host

    @staticmethod
    def _connection_mode_for_host(host: Optional[str]) -> str:
        """Select ReachyMini connection mode based on the target host."""
        if host in (None, "localhost", "127.0.0.1", "::1"):
            return "localhost_only"
        return "network"

    def __enter__(self):
        if ReachyMini is None:
            raise RuntimeError(
                "reachy_mini is not available. "
                "Use --camera-source webcam for local test, "
                "or install: pip install 'reachy-mini[mujoco]'"
            )

        # Load camera profile if specified (before SDK initializes camera)
        if self._camera_profile:
            print(f"📷 Loading camera profile: {self._camera_profile}")
            if load_camera_profile(self._camera_profile):
                print(f"✅ Camera profile '{self._camera_profile}' loaded")
            else:
                print(f"⚠️ Failed to load camera profile: {self._camera_profile}")

        # Try to initialize Reachy with media backend. If media backend fails
        # (e.g., WebRTC server not running), fall back to a no-media backend so
        # robot control is still available.
        normalized_host = self._normalize_host(self._host)
        connection_mode = self._connection_mode_for_host(normalized_host)
        print(
            f"🔌 Connecting to Reachy daemon at {normalized_host or 'reachy-mini.local'}:"
            f"{self._port or 8000} ({connection_mode})"
        )
        try:
            # Pass host/port to ReachyMini when provided so remote daemons are reachable
            kwargs = {}
            if normalized_host is not None:
                kwargs["host"] = normalized_host
            if self._port is not None:
                kwargs["port"] = self._port
            kwargs["connection_mode"] = connection_mode
            self._ctx = ReachyMini(media_backend="default", **kwargs)
        except Exception as exc:
            print(f"⚠️ reachy_mini init with media backend failed: {exc}")
            print("   ↪ Retrying with media_backend='no_media' to enable control-only mode")
            try:
                kwargs = {}
                if normalized_host is not None:
                    kwargs["host"] = normalized_host
                if self._port is not None:
                    kwargs["port"] = self._port
                kwargs["connection_mode"] = connection_mode
                self._ctx = ReachyMini(media_backend="no_media", **kwargs)
            except Exception as exc2:
                print(f"❌ reachy_mini init failed (no-media fallback): {exc2}")
                raise

        self._reachy = self._ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._ctx is not None:
            return self._ctx.__exit__(exc_type, exc_val, exc_tb)
        return False

    def get_frame(self) -> Optional[np.ndarray]:
        if self._reachy and hasattr(self._reachy, "media") and self._reachy.media:
            return self._reachy.media.get_frame()
        return None

    def look_at_image(self, x: int, y: int, duration: float = 0.2,
                       frame_width: int = 640, frame_height: int = 480,
                       pan_gain: float = 0.3, tilt_gain: float = 0.2,
                       pan_invert: bool = False) -> None:
        """Make Reachy look at image coordinates.

        Maps pixel coordinates to head pan/tilt and calls
        goto_target(head=pose).  Always uses the manual fallback mapping
        (the SDK's built-in look_at_image tries to access camera media
        which is None in --no-media daemon mode).
        """
        if not self._reachy:
            return

        w, h = frame_width, frame_height
        nx = (x - (w / 2.0)) / (w / 2.0)
        ny = (y - (h / 2.0)) / (h / 2.0)

        pan = float(nx * pan_gain)
        if pan_invert:
            pan = -pan
        tilt = float(ny * tilt_gain)

        try:
            pose = create_head_pose(pan=pan, tilt=tilt)
        except TypeError:
            try:
                pose = create_head_pose(pan, tilt)
            except Exception:
                pose = None

        if pose is not None:
            self._reachy.goto_target(head=pose, duration=duration)

    def goto_body_yaw(self, yaw: float, duration: float = 0.35) -> None:
        if self._reachy:
            self._reachy.goto_target(body_yaw=yaw, duration=duration)

    def move_head(self, pan: float = 0.0, tilt: float = 0.0, duration: float = 0.5) -> None:
        if self._reachy:
            try:
                pose = create_head_pose(pan=pan, tilt=tilt)
            except TypeError:
                try:
                    pose = create_head_pose(pan, tilt)
                except Exception:
                    pose = None
            if pose is not None:
                self._reachy.goto_target(head=pose, duration=duration)

    def reset_head(self, duration: float = 0.2) -> None:
        if self._reachy:
            self._reachy.goto_target(head=create_head_pose(), duration=duration)

    def set_automatic_body_yaw(self, enabled: bool) -> None:
        if self._reachy:
            self._reachy.set_automatic_body_yaw(enabled)


class WebcamRuntime(RobotRuntime):
    """Runtime for local webcam (testing without robot).

    Supports both integer indices (0, 1, 2...) and device paths (/dev/video4).
    """

    def __init__(self, camera_index: int = 0, device_path: Optional[str] = None):
        self._camera_index = camera_index
        self._device_path = device_path
        self._cap: Optional[cv2.VideoCapture] = None

    def __enter__(self):
        # Use device path if provided, otherwise use index
        if self._device_path:
            self._cap = cv2.VideoCapture(self._device_path, cv2.CAP_V4L2)
            device_str = self._device_path
        else:
            self._cap = cv2.VideoCapture(self._camera_index, cv2.CAP_V4L2)
            device_str = f"index {self._camera_index}"

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open webcam {device_str}")

        # Prefer MJPG (Motion-JPEG) format — YUYV raw often yields black/dark frames
        # on cameras without hardware ISP (e.g., Reachy Mini Camera).
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self._cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        # Set resolution for better performance
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._cap is not None:
            self._cap.release()
        return False

    def get_frame(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok:
            return None
        # Camera may ignore the 640x480 set() call (e.g. Reachy MJPG only
        # supports 1920x1080+).  Resize to a consistent resolution so all
        # pixel-based parameters (deadzone, etc.) behave as
        # designed regardless of the camera's native output.
        if frame.shape[1] != 640 or frame.shape[0] != 480:
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
        return frame

    def look_at_image(self, x: int, y: int, duration: float = 0.2,
                       frame_width: int = 640, frame_height: int = 480,
                       pan_gain: float = 0.3, tilt_gain: float = 0.2,
                       pan_invert: bool = False) -> None:
        pass

    def goto_body_yaw(self, yaw: float, duration: float = 0.35) -> None:
        # No-op for webcam
        pass

    def move_head(self, pan: float = 0.0, tilt: float = 0.0, duration: float = 0.5) -> None:
        # No-op for webcam
        pass

    def reset_head(self, duration: float = 0.2) -> None:
        # No-op for webcam
        pass

    def set_automatic_body_yaw(self, enabled: bool) -> None:
        # No-op for webcam
        pass


def create_runtime(
    camera_source: str,
    camera_index: int = 0,
    camera_profile: Optional[str] = None,
    reachy_device_path: Optional[str] = None,
    reachy_host: Optional[str] = None,
    reachy_port: Optional[int] = None,
) -> RobotRuntime:
    """Factory function to create appropriate runtime.

    Args:
        camera_source: "reachy" or "webcam"
        camera_index: Camera device index for webcam mode
        camera_profile: Name of camera profile to load (for reachy mode)
        reachy_device_path: Device path for Reachy camera (e.g., '/dev/video4')
    """
    if camera_source == "reachy":
        return ReachyRuntime(camera_profile=camera_profile, host=reachy_host, port=reachy_port)
    else:
        return WebcamRuntime(camera_index, device_path=reachy_device_path)
