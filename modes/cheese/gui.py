"""GUI for cheese mode - OpenCV only."""

import queue
from typing import List, Optional

import cv2
import numpy as np

from .config import CheeseConfig


class CheeseGUI:
    """GUI for cheese photo mode using OpenCV."""

    def __init__(self, config: CheeseConfig):
        self.cfg = config
        self._event_queue: "queue.Queue[str]" = queue.Queue()
        self._ready = True
        self._backend = "none" if config.gui_backend == "none" else "cv2"
        self._window_name = "ReachyCheese"

        # Button layout
        self._button_height = 44
        self._buttons = [
            ("Wake", "manual_wake"),
            ("Take Photo", "manual_capture"),
            ("Sleep", "manual_sleep"),
        ]

        # Initialize
        if self._backend == "cv2":
            self._init_cv2()

    def _init_cv2(self) -> None:
        """Initialize OpenCV GUI."""
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            self._window_name,
            self.cfg.preview_width,
            self.cfg.preview_height + self._button_height + 60,
        )
        cv2.setMouseCallback(self._window_name, self._on_mouse)

    @property
    def available(self) -> bool:
        return self._ready

    def is_running(self) -> bool:
        """Check if GUI is still running."""
        if not self._ready:
            return False
        if self._backend == "none":
            return True  # Headless mode always runs
        # OpenCV - check if window exists and is visible
        try:
            visible = cv2.getWindowProperty(self._window_name, cv2.WND_PROP_VISIBLE)
            if visible < 1:
                return False
            autosize = cv2.getWindowProperty(self._window_name, cv2.WND_PROP_AUTOSIZE)
            if autosize < 0:
                return False
            return True
        except cv2.error:
            return False
        except Exception:
            return False

    def get_events(self) -> List[str]:
        """Get pending GUI events."""
        events = []
        while not self._event_queue.empty():
            try:
                events.append(self._event_queue.get_nowait())
            except queue.Empty:
                break

        # Check OpenCV key presses and window state
        if self._backend == "cv2":
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                events.append("quit")

            # Check if window was closed (click X button)
            try:
                visible = cv2.getWindowProperty(self._window_name, cv2.WND_PROP_VISIBLE)
                if visible < 1:
                    events.append("quit")
            except cv2.error:
                events.append("quit")

        return events

    def draw(self, frame: np.ndarray, state, status: Optional[dict], last_saved: str) -> None:
        """Draw the GUI."""
        if self._backend == "cv2":
            self._draw_cv2(frame, state, status, last_saved)
        # "none" backend: no drawing

    def _draw_cv2(self, frame: np.ndarray, state, status: Optional[dict], last_saved: str) -> None:
        """Draw using OpenCV."""
        # Get original dimensions
        src_h, src_w = frame.shape[:2]

        # Resize frame
        frame_resized = cv2.resize(frame, (self.cfg.preview_width, self.cfg.preview_height))

        # Calculate scale factors
        scale_x = self.cfg.preview_width / float(src_w)
        scale_y = self.cfg.preview_height / float(src_h)

        # Draw overlays with scaling
        self._draw_overlays(frame_resized, state, status, scale_x, scale_y)

        # Create canvas with button panel
        panel_h = self._button_height + 60
        canvas = np.zeros(
            (self.cfg.preview_height + panel_h, self.cfg.preview_width, 3),
            dtype=np.uint8,
        )
        canvas[:self.cfg.preview_height, :, :] = frame_resized

        # Draw status text
        cv2.putText(
            canvas,
            f"State: {state.value.upper()}",
            (10, self.cfg.preview_height + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (240, 240, 240),
            2,
        )

        if status and status.get("has_face"):
            face_text = f"Face stable={status['stable_frames']}, aligned={status['aligned']}"
        else:
            face_text = "Face: not detected"

        cv2.putText(
            canvas,
            face_text[:50],
            (10, self.cfg.preview_height + 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (200, 200, 200),
            1,
        )

        # Draw buttons
        spacing = 8
        top = self.cfg.preview_height + 10
        available_width = self.cfg.preview_width - spacing * (len(self._buttons) + 1)
        btn_w = available_width // len(self._buttons)

        for i, (label, _) in enumerate(self._buttons):
            bx = spacing + i * (btn_w + spacing)
            by = top
            cv2.rectangle(
                canvas,
                (bx, by),
                (bx + btn_w, by + self._button_height),
                (70, 70, 70),
                -1,
            )
            cv2.rectangle(
                canvas,
                (bx, by),
                (bx + btn_w, by + self._button_height),
                (130, 130, 130),
                1,
            )
            cv2.putText(
                canvas,
                label,
                (bx + 10, by + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (235, 235, 235),
                2,
            )

        cv2.imshow(self._window_name, canvas)

    def _draw_overlays(
        self,
        frame: np.ndarray,
        state,
        status: Optional[dict],
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ) -> None:
        """Draw overlays on frame."""
        h, w = frame.shape[:2]

        # Draw center crosshair
        cv2.drawMarker(
            frame,
            (w // 2, h // 2),
            (0, 220, 220),
            markerType=cv2.MARKER_CROSS,
            markerSize=22,
            thickness=2,
        )

        # Draw face box (with scaling)
        if status and status.get("bbox"):
            x, y, bw, bh = status["bbox"]
            px = int(x * scale_x)
            py = int(y * scale_y)
            pw = int(bw * scale_x)
            ph = int(bh * scale_y)
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), (50, 230, 50), 2)

        # Draw state indicator
        state_colors = {
            "sleep": (100, 100, 100),
            "tracking": (0, 200, 200),
            "armed": (0, 255, 0),
            "countdown": (0, 0, 255),
        }
        color = state_colors.get(state.value, (200, 200, 200))
        cv2.circle(frame, (30, 30), 15, color, -1)

        # Draw tracking info
        if status and status.get("has_face"):
            dx = status.get("dx", 0)
            dy = status.get("dy", 0)
            info_text = f"dx={dx:+.0f} dy={dy:+.0f}"
            cv2.putText(
                frame,
                info_text,
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

            # Draw alignment target box (deadzone)
            cx, cy = w // 2, h // 2
            dz_x = int(25 * scale_x)
            dz_y = int(20 * scale_y)
            cv2.rectangle(
                frame,
                (cx - dz_x, cy - dz_y),
                (cx + dz_x, cy + dz_y),
                (0, 200, 200),
                1,
            )

    def _on_mouse(self, event, x, y, flags, param) -> None:
        """Handle mouse events for OpenCV GUI."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        top = self.cfg.preview_height + 10
        if y < top:
            return

        spacing = 8
        available_width = self.cfg.preview_width - spacing * (len(self._buttons) + 1)
        btn_w = available_width // len(self._buttons)

        for i, (_, action) in enumerate(self._buttons):
            bx = spacing + i * (btn_w + spacing)
            by = top
            if bx <= x <= bx + btn_w and by <= y <= by + self._button_height:
                self._event_queue.put(action)
                return

    def close(self) -> None:
        """Cleanup GUI resources."""
        if self._backend == "cv2":
            try:
                cv2.destroyWindow(self._window_name)
            except Exception:
                pass
        self._ready = False
