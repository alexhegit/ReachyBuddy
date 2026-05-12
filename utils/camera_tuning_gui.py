#!/usr/bin/env python3
"""Camera tuning GUI with real-time preview.

Usage:
    python utils/camera_tuning_gui.py [OPTIONS]

Controls:
    - Adjust trackbars to change camera parameters in real-time
    - Press 's' to save current profile
    - Press 'l' to load profile
    - Press 'r' to reset to defaults
    - Press 'q' or ESC to quit
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


DEFAULT_DEVICE = "/dev/video0"
PROFILE_DIR = Path.home() / ".config" / "reachy_mini"

# Reachy camera identifiers (VID:PID)
REACHY_CAMERAS = [
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
                if "Reachy" in line_stripped or "Arducam" in line_stripped:
                    reachy_section = True
                    continue
                # If we're in Reachy section and find a video device
                if reachy_section and "/dev/video" in line_stripped:
                    device = line_stripped.split("(")[0].strip()
                    # Verify this device supports controls by testing brightness
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
                    # This device doesn't support controls, continue to next
                    continue
                # Empty line or new section ends Reachy section
                if reachy_section and (not line_stripped or ":" in line_stripped):
                    reachy_section = False

        # Method 2: Check via /sys/class/video4linux for device names
        sys_video_path = Path("/sys/class/video4linux")
        if sys_video_path.exists():
            for device_dir in sorted(sys_video_path.glob("video*")):
                name_file = device_dir / "name"
                if name_file.exists():
                    name = name_file.read_text().strip()
                    if "reachy" in name.lower() or "arducam" in name.lower():
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

    except Exception as e:
        print(f"Warning: Failed to auto-detect Reachy camera: {e}")

    return None


def list_all_cameras() -> List[Tuple[str, str]]:
    """List all available cameras.

    Returns:
        List of (device_path, name) tuples
    """
    cameras = []
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            current_name = "Unknown"
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if "/dev/video" in line:
                    device = line.split("(")[0].strip()
                    cameras.append((device, current_name))
                else:
                    current_name = line.split(":")[0].strip()
    except Exception:
        pass

    # Fallback: scan /dev/video*
    if not cameras:
        for i in range(10):
            device = f"/dev/video{i}"
            if Path(device).exists():
                cameras.append((device, f"Camera {i}"))

    return cameras


def get_camera_specs():
    """Get camera specifications."""
    return {
        "brightness": {"min": -64, "max": 64, "default": 0, "step": 1},
        "contrast": {"min": 0, "max": 95, "default": 1, "step": 1},
        "saturation": {"min": 0, "max": 100, "default": 48, "step": 1},
        "hue": {"min": -2000, "max": 2000, "default": 0, "step": 1},
        "gamma": {"min": 80, "max": 160, "default": 100, "step": 1},
        "gain": {"min": 0, "max": 255, "default": 32, "step": 1},
        "sharpness": {"min": 0, "max": 7, "default": 2, "step": 1},
        "backlight_compensation": {"min": 0, "max": 10, "default": 2, "step": 1},
    }


def v4l2_get(device: str, param: str) -> Optional[int]:
    """Get a single parameter value using v4l2-ctl."""
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", device, "--get-ctrl", param],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Parse output: "param: value"
            line = result.stdout.strip().split("\n")[0]
            if ":" in line:
                return int(line.split(":", 1)[1].strip())
    except Exception:
        pass
    return None


def v4l2_set(device: str, params: Dict[str, int]) -> bool:
    """Set parameters using v4l2-ctl."""
    try:
        ctrl_str = ",".join([f"{k}={v}" for k, v in params.items()])
        result = subprocess.run(
            ["v4l2-ctl", "-d", device, "--set-ctrl", ctrl_str],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


class CameraTuningGUI:
    """GUI for camera parameter tuning with real-time preview."""

    # Trackbar range for OpenCV (0-100 mapped to actual range)
    TRACKBAR_MAX = 100

    def __init__(self, device: str = DEFAULT_DEVICE, profile_dir: Path = PROFILE_DIR):
        self.device = device
        self.profile_dir = profile_dir
        self.profile_dir.mkdir(parents=True, exist_ok=True)

        self.specs = get_camera_specs()
        self.params = list(self.specs.keys())
        self.current_values = {}

        # Camera capture
        self.cap = None

        # Window settings
        self.window_name = "Camera Tuning GUI"
        self.preview_width = 640
        self.preview_height = 480

        # Button states for visual feedback
        self.button_pressed = None
        self.button_press_time = 0

    def _map_to_trackbar(self, param: str, value: int) -> int:
        """Map actual parameter value to trackbar 0-100 range."""
        spec = self.specs[param]
        min_val, max_val = spec["min"], spec["max"]
        # Linear mapping
        trackbar_val = int((value - min_val) / (max_val - min_val) * self.TRACKBAR_MAX)
        return max(0, min(self.TRACKBAR_MAX, trackbar_val))

    def _map_from_trackbar(self, param: str, trackbar_val: int) -> int:
        """Map trackbar 0-100 to actual parameter value."""
        spec = self.specs[param]
        min_val, max_val = spec["min"], spec["max"]
        # Linear mapping
        value = int(min_val + (trackbar_val / self.TRACKBAR_MAX) * (max_val - min_val))
        # Round to step
        step = spec.get("step", 1)
        value = round(value / step) * step
        return max(min_val, min(max_val, value))

    def _read_camera_values(self) -> Dict[str, int]:
        """Read current values from camera."""
        values = {}
        for param in self.params:
            val = v4l2_get(self.device, param)
            if val is not None:
                values[param] = val
            else:
                values[param] = self.specs[param]["default"]
        return values

    def _create_trackbars(self):
        """Create OpenCV trackbars for each parameter."""
        for param in self.params:
            current_val = self.current_values.get(param, self.specs[param]["default"])
            trackbar_val = self._map_to_trackbar(param, current_val)

            cv2.createTrackbar(
                param,
                self.window_name,
                trackbar_val,
                self.TRACKBAR_MAX,
                lambda x, p=param: self._on_trackbar_change(p, x)
            )

    def _on_trackbar_change(self, param: str, trackbar_val: int):
        """Handle trackbar position change."""
        new_value = self._map_from_trackbar(param, trackbar_val)
        if self.current_values.get(param) != new_value:
            self.current_values[param] = new_value
            v4l2_set(self.device, {param: new_value})

    def _draw_button(self, canvas: np.ndarray, x: int, y: int, w: int, h: int,
                     label: str, is_pressed: bool = False) -> None:
        """Draw a button on the canvas."""
        # Colors
        if is_pressed:
            bg_color = (100, 150, 200)  # Lighter when pressed
            text_color = (255, 255, 255)
            border_color = (200, 200, 255)
        else:
            bg_color = (60, 100, 150)  # Normal blue
            text_color = (255, 255, 255)
            border_color = (100, 150, 200)

        # Button background
        cv2.rectangle(canvas, (x, y), (x + w, y + h), bg_color, -1)
        # Button border
        cv2.rectangle(canvas, (x, y), (x + w, y + h), border_color, 2)

        # Button text (centered)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(canvas, label, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    def _get_button_regions(self, panel_x: int, start_y: int) -> Dict[str, tuple]:
        """Get button click regions (x, y, w, h) for each button."""
        btn_w = 105
        btn_h = 35
        gap = 10

        return {
            "save": (panel_x + 10, start_y, btn_w, btn_h),
            "load": (panel_x + 10 + btn_w + gap, start_y, btn_w, btn_h),
            "reset": (panel_x + 10, start_y + btn_h + gap, btn_w, btn_h),
            "quit": (panel_x + 10 + btn_w + gap, start_y + btn_h + gap, btn_w, btn_h),
        }

    def _handle_mouse_click(self, event: int, x: int, y: int, flags: int, param: any) -> None:
        """Handle mouse click events for buttons."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        h, w = self.preview_height, self.preview_width
        panel_x = w

        # Button region starts near bottom
        start_y = h - 140
        buttons = self._get_button_regions(panel_x, start_y)

        for btn_name, (bx, by, bw, bh) in buttons.items():
            if bx <= x < bx + bw and by <= y < by + bh:
                self.button_pressed = btn_name
                self.button_press_time = cv2.getTickCount()

                # Execute action
                if btn_name == "save":
                    self._save_profile()
                elif btn_name == "load":
                    self._load_profile()
                elif btn_name == "reset":
                    self._reset_defaults()
                elif btn_name == "quit":
                    self._should_quit = True
                break

    def _draw_info_panel(self, frame: np.ndarray) -> np.ndarray:
        """Draw parameter info panel on the right side of frame."""
        h, w = frame.shape[:2]
        panel_width = 250

        # Create canvas with panel
        canvas = np.zeros((h, w + panel_width, 3), dtype=np.uint8)
        canvas[:h, :w] = frame

        # Draw panel background
        canvas[:h, w:] = (40, 40, 40)

        # Title
        cv2.putText(canvas, "Parameters", (w + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.line(canvas, (w + 10, 40), (w + panel_width - 10, 40), (100, 100, 100), 1)

        # Draw each parameter
        y = 70
        for param in self.params:
            spec = self.specs[param]
            current = self.current_values.get(param, spec["default"])
            default = spec["default"]

            # Parameter name
            color = (0, 255, 0) if current == default else (0, 200, 255)
            cv2.putText(canvas, param, (w + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            # Current value
            text = f"{current} [{spec['min']}..{spec['max']}]"
            cv2.putText(canvas, text, (w + 10, y + 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # Visual bar
            bar_width = 230
            bar_y = y + 25
            ratio = (current - spec["min"]) / (spec["max"] - spec["min"])
            filled_width = int(bar_width * ratio)

            cv2.rectangle(canvas, (w + 10, bar_y), (w + 10 + bar_width, bar_y + 8), (60, 60, 60), -1)
            cv2.rectangle(canvas, (w + 10, bar_y), (w + 10 + filled_width, bar_y + 8), (0, 150, 255), -1)

            y += 55

        # Draw buttons at the bottom
        start_y = h - 140
        buttons = self._get_button_regions(w, start_y)

        # Check if button is pressed (for visual feedback)
        import time
        is_pressed = lambda name: (self.button_pressed == name and
                                   (cv2.getTickCount() - self.button_press_time) / cv2.getTickFrequency() < 0.2)

        # Draw buttons
        for btn_name, (bx, by, bw, bh) in buttons.items():
            label = btn_name.capitalize()
            self._draw_button(canvas, bx, by, bw, bh, label, is_pressed(btn_name))

        # Draw keyboard shortcuts hint
        hint_y = h - 25
        cv2.putText(canvas, "Hotkeys: s=save, l=load, r=reset, q=quit",
                   (w + 10, hint_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

        return canvas

    def _save_profile(self):
        """Save current settings as a profile."""
        profile_name = input("\nEnter profile name to save: ").strip()
        if not profile_name:
            print("Save cancelled")
            return

        profile_path = self.profile_dir / f"{profile_name}.json"
        try:
            with open(profile_path, "w") as f:
                json.dump({
                    "name": profile_name,
                    "device": self.device,
                    "params": self.current_values
                }, f, indent=2)
            print(f"✅ Saved profile: {profile_name}")
        except Exception as e:
            print(f"❌ Failed to save: {e}")

    def _load_profile(self):
        """Load a profile."""
        profiles = list(self.profile_dir.glob("*.json"))
        if not profiles:
            print("\nNo saved profiles found")
            return

        print("\nAvailable profiles:")
        for i, p in enumerate(profiles, 1):
            print(f"  {i}. {p.stem}")

        try:
            choice = input("Enter profile name or number: ").strip()
            # Try as number first
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(profiles):
                    profile_path = profiles[idx]
                else:
                    print("Invalid selection")
                    return
            except ValueError:
                # Try as name
                profile_path = self.profile_dir / f"{choice}.json"

            if not profile_path.exists():
                print(f"Profile not found: {choice}")
                return

            with open(profile_path, "r") as f:
                data = json.load(f)

            # Apply loaded values
            loaded_params = data.get("params", {})
            if v4l2_set(self.device, loaded_params):
                self.current_values.update(loaded_params)
                # Update trackbars
                for param, value in loaded_params.items():
                    if param in self.params:
                        trackbar_val = self._map_to_trackbar(param, value)
                        cv2.setTrackbarPos(param, self.window_name, trackbar_val)
                print(f"✅ Loaded profile: {data.get('name', profile_path.stem)}")
            else:
                print("❌ Failed to apply profile")
        except Exception as e:
            print(f"❌ Error loading profile: {e}")

    def _reset_defaults(self):
        """Reset all parameters to defaults."""
        defaults = {p: self.specs[p]["default"] for p in self.params}
        if v4l2_set(self.device, defaults):
            self.current_values = defaults.copy()
            # Update trackbars
            for param, value in defaults.items():
                trackbar_val = self._map_to_trackbar(param, value)
                cv2.setTrackbarPos(param, self.window_name, trackbar_val)
            print("✅ Reset to defaults")
        else:
            print("❌ Failed to reset")

    def run(self):
        """Run the GUI main loop."""
        # Auto-detect Reachy camera if using default device
        if self.device == DEFAULT_DEVICE:
            detected = find_reachy_camera()
            if detected and detected != self.device:
                print(f"🔍 Auto-detected Reachy camera: {detected}")
                self.device = detected
            elif detected is None:
                # List available cameras and let user choose
                cameras = list_all_cameras()
                if len(cameras) > 1:
                    print("\n📷 Multiple cameras detected:")
                    for i, (dev, name) in enumerate(cameras, 1):
                        marker = " ← Reachy?" if "reachy" in name.lower() or "arducam" in name.lower() else ""
                        print(f"  {i}. {dev} - {name}{marker}")
                    print(f"  Using default: {self.device}")
                    print(f"  (Use --device to specify another camera)\n")
                elif len(cameras) == 1:
                    print(f"📷 Found camera: {cameras[0][0]} - {cameras[0][1]}")

        # Open camera
        print(f"Opening camera: {self.device}")
        self.cap = cv2.VideoCapture(self.device)
        if not self.cap.isOpened():
            print(f"❌ Failed to open camera: {self.device}")
            return 1

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.preview_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.preview_height)

        # Read current camera values
        self.current_values = self._read_camera_values()
        print("Current camera values:", self.current_values)

        # Initialize GUI state
        self._should_quit = False
        self.button_pressed = None
        self.button_press_time = 0

        # Create window and set up mouse callback
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 900, 600)
        cv2.setMouseCallback(self.window_name, self._handle_mouse_click)

        # Create trackbars
        self._create_trackbars()

        print("\n🎛️  Camera Tuning GUI started")
        print("Adjust trackbars or click buttons to change parameters")
        print("Press 's'=save, 'l'=load, 'r'=reset, 'q'=quit\n")

        while not self._should_quit:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame")
                break

            # Resize to preview size
            frame = cv2.resize(frame, (self.preview_width, self.preview_height))

            # Draw info panel with buttons
            display = self._draw_info_panel(frame)

            # Show
            cv2.imshow(self.window_name, display)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                break
            elif key == ord('s'):
                self._save_profile()
            elif key == ord('l'):
                self._load_profile()
            elif key == ord('r'):
                self._reset_defaults()

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nCamera tuner exited")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Camera tuning GUI with real-time preview"
    )
    parser.add_argument(
        "--device", "-d",
        default=DEFAULT_DEVICE,
        help=f"Camera device (default: {DEFAULT_DEVICE})"
    )

    args = parser.parse_args()

    gui = CameraTuningGUI(device=args.device)
    return gui.run()


if __name__ == "__main__":
    sys.exit(main())
