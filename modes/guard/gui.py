"""GUI for guard mode - OpenCV only."""

from typing import List

import cv2
import numpy as np

from .config import GuardConfig


class GuardGUI:
    """Simple OpenCV GUI for guard mode."""

    def __init__(self, config: GuardConfig):
        self.cfg = config
        self._ready = True
        self._backend = "none" if config.gui_backend == "none" else "cv2"
        self._window_name = "ReachyGuard"
        self._first_frame = True

    @property
    def available(self) -> bool:
        return self._ready

    def is_running(self) -> bool:
        if not self._ready:
            return False
        if self._backend == "none":
            return True
        try:
            visible = cv2.getWindowProperty(self._window_name, cv2.WND_PROP_VISIBLE)
            return visible >= 1
        except (cv2.error, Exception):
            return False

    def get_events(self) -> List[str]:
        events = []
        if self._backend == "cv2":
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                events.append("quit")
            try:
                if cv2.getWindowProperty(self._window_name, cv2.WND_PROP_VISIBLE) < 1:
                    events.append("quit")
            except cv2.error:
                events.append("quit")
        return events

    def draw(self, frame: np.ndarray, status_text: str, scan_pan: float):
        if self._backend == "none":
            return

        pw, ph = self.cfg.preview_width, self.cfg.preview_height

        canvas = cv2.resize(frame, (pw, ph))

        # Status overlay
        if status_text:
            cv2.putText(canvas, status_text[:60], (8, ph - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2)

        # Scan indicator
        bar_w, bar_h = 100, 6
        bar_x = pw // 2 - bar_w // 2
        cv2.rectangle(canvas, (bar_x, 6), (bar_x + bar_w, 6 + bar_h), (50, 50, 50), -1)
        pos = bar_x + int((scan_pan / self.cfg.scan_range + 1) * 0.5 * bar_w)
        pos = max(bar_x + 1, min(bar_x + bar_w - 1, pos))
        cv2.rectangle(canvas, (pos - 2, 6), (pos + 2, 6 + bar_h), (0, 220, 0), -1)

        # Simple info bar at bottom
        cv2.rectangle(canvas, (0, ph - 24), (pw, ph), (30, 30, 30), -1)
        cv2.putText(canvas, status_text[:55], (6, ph - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
        cv2.putText(canvas, f"Scan: {scan_pan:+.2f}", (pw - 120, ph - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

        cv2.imshow(self._window_name, canvas)
        cv2.waitKey(1)

    def close(self):
        if self._backend == "cv2":
            try:
                cv2.destroyWindow(self._window_name)
            except Exception:
                pass
        self._ready = False
