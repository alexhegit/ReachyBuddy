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

    def look_at_image(self, x: int, y: int, duration: float = 0.2) -> None:
        """Make robot look at image coordinates."""
        raise NotImplementedError

    def goto_body_yaw(self, yaw: float, duration: float = 0.35) -> None:
        """Set body yaw."""
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

    def __init__(self, camera_profile: Optional[str] = None):
        self._ctx = None
        self._reachy = None
        self._camera_profile = camera_profile

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

        self._ctx = ReachyMini(media_backend="default")
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

    def look_at_image(self, x: int, y: int, duration: float = 0.2) -> None:
        if self._reachy:
            self._reachy.look_at_image(x, y, duration=duration)

    def goto_body_yaw(self, yaw: float, duration: float = 0.35) -> None:
        if self._reachy:
            self._reachy.goto_target(body_yaw=yaw, duration=duration)

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
            self._cap = cv2.VideoCapture(self._device_path)
            device_str = self._device_path
        else:
            self._cap = cv2.VideoCapture(self._camera_index)
            device_str = f"index {self._camera_index}"

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open webcam {device_str}")
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
        return frame

    def look_at_image(self, x: int, y: int, duration: float = 0.2) -> None:
        # No-op for webcam
        pass

    def goto_body_yaw(self, yaw: float, duration: float = 0.35) -> None:
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
    reachy_device_path: Optional[str] = None
) -> RobotRuntime:
    """Factory function to create appropriate runtime.

    Args:
        camera_source: "reachy" or "webcam"
        camera_index: Camera device index for webcam mode
        camera_profile: Name of camera profile to load (for reachy mode)
        reachy_device_path: Device path for Reachy camera (e.g., '/dev/video4')
    """
    if camera_source == "reachy":
        return ReachyRuntime(camera_profile=camera_profile)
    else:
        return WebcamRuntime(camera_index, device_path=reachy_device_path)
