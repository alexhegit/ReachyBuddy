"""GUI for chat mode - OpenCV only."""

from typing import List

import cv2
import numpy as np

from .config import ChatConfig


class ChatGUI:
    """Simple OpenCV GUI for chat mode."""

    def __init__(self, config: ChatConfig):
        self.cfg = config
        self._backend = "none" if config.gui_backend == "none" else "cv2"
        self._window_name = "ReachyChat"
        self._ready = True

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

    def draw(self, frame: np.ndarray, status: str, subtitle: str = ""):
        if self._backend == "none":
            return

        pw, ph = self.cfg.preview_width, self.cfg.preview_height
        canvas = cv2.resize(frame, (pw, ph))

        # Status bar at bottom
        cv2.rectangle(canvas, (0, ph - 28), (pw, ph), (30, 30, 30), -1)
        cv2.putText(canvas, status[:55], (6, ph - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1)

        # Subtitle at top
        if subtitle:
            cv2.rectangle(canvas, (0, 0), (pw, 28), (30, 30, 30), -1)
            cv2.putText(canvas, subtitle[:55], (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)

        cv2.imshow(self._window_name, canvas)
        cv2.waitKey(1)

    def close(self):
        if self._backend == "cv2":
            try:
                cv2.destroyWindow(self._window_name)
            except Exception:
                pass
        self._ready = False
