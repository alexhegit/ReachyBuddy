"""GUI for cheese mode."""

import queue
from typing import TYPE_CHECKING, List, Optional, Tuple

import cv2
import numpy as np

try:
    from dearpygui import dearpygui as dpg
except Exception:
    dpg = None

from .config import CheeseConfig


class CheeseGUI:
    """GUI for cheese photo mode."""
    
    def __init__(self, config: CheeseConfig):
        self.cfg = config
        self._event_queue: "queue.Queue[str]" = queue.Queue()
        self._ready = True
        self._backend = self._detect_backend()
        self._window_name = "ReachyCheese"
        
        # Button layout
        self._button_height = 44
        self._buttons = [
            ("Wake", "manual_wake"),
            ("Take Photo", "manual_capture"),
            ("Sleep", "manual_sleep"),
        ]
        
        # DPG specific
        self._texture_tag = "preview_texture"
        self._status_tag = "status_text"
        self._face_tag = "face_text"
        
        # Initialize
        if self._backend == "dpg":
            self._init_dpg()
        else:
            self._init_cv2()
    
    def _detect_backend(self) -> str:
        """Detect best available GUI backend."""
        if self.cfg.gui_backend == "none":
            return "none"
        if self.cfg.gui_backend in ("auto", "dpg") and dpg is not None:
            return "dpg"
        if self.cfg.gui_backend in ("auto", "cv2", "none"):
            return "cv2"
        return "cv2"
    
    def _init_dpg(self) -> None:
        """Initialize DearPyGui."""
        try:
            dpg.create_context()
            
            with dpg.texture_registry():
                blank = np.zeros((self.cfg.preview_height, self.cfg.preview_width, 3), dtype=np.float32)
                dpg.add_raw_texture(
                    width=self.cfg.preview_width,
                    height=self.cfg.preview_height,
                    default_value=blank.flatten(),
                    format=dpg.mvFormat_Float_rgb,
                    tag=self._texture_tag,
                )
            
            with dpg.window(
                label="ReachyCheese",
                width=self.cfg.preview_width + 28,
                height=self.cfg.preview_height + 180,
            ):
                dpg.add_image(self._texture_tag)
                dpg.add_text("State: SLEEP", tag=self._status_tag)
                dpg.add_text("Face: --", tag=self._face_tag)
                
                with dpg.group(horizontal=True):
                    for label, action in self._buttons:
                        dpg.add_button(
                            label=label,
                            callback=lambda s, a, u=action: self._event_queue.put(u),
                            user_data=action,
                        )
            
            dpg.create_viewport(
                title="ReachyCheese",
                width=self.cfg.preview_width + 32,
                height=self.cfg.preview_height + 200,
            )
            dpg.setup_dearpygui()
            dpg.show_viewport()
        except Exception as e:
            print(f"⚠️ DPG init failed: {e}, falling back to OpenCV")
            self._backend = "cv2"
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
        if self._backend == "dpg":
            return dpg.is_dearpygui_running()
        if self._backend == "none":
            return True  # Headless mode always runs
        # OpenCV - check if window exists and is visible
        try:
            # WND_PROP_VISIBLE returns -1 if window is closed
            visible = cv2.getWindowProperty(self._window_name, cv2.WND_PROP_VISIBLE)
            if visible < 1:
                return False
            # Also check WND_PROP_AUTOSIZE as additional verification
            autosize = cv2.getWindowProperty(self._window_name, cv2.WND_PROP_AUTOSIZE)
            if autosize < 0:
                return False
            return True
        except cv2.error:
            # Window was closed
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
                # Window was closed
                events.append("quit")
        
        return events
    
    def draw(self, frame: np.ndarray, state, status: Optional[dict], last_saved: str) -> None:
        """Draw the GUI."""
        if self._backend == "dpg":
            self._draw_dpg(frame, state, status, last_saved)
        elif self._backend == "cv2":
            self._draw_cv2(frame, state, status, last_saved)
        # "none" backend: no drawing
    
    def _draw_dpg(self, frame: np.ndarray, state, status: Optional[dict], last_saved: str) -> None:
        """Draw using DearPyGui."""
        # Get original dimensions
        src_h, src_w = frame.shape[:2]
        
        # Resize and convert
        frame_resized = cv2.resize(frame, (self.cfg.preview_width, self.cfg.preview_height))
        
        # Calculate scale factors
        scale_x = self.cfg.preview_width / float(src_w)
        scale_y = self.cfg.preview_height / float(src_h)
        
        # Draw overlays with scaling
        self._draw_overlays(frame_resized, state, status, scale_x, scale_y)
        
        # Update texture
        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        dpg.set_value(self._texture_tag, rgb.flatten())
        
        # Update text
        dpg.set_value(self._status_tag, f"State: {state.value.upper()}")
        if status and status.get("has_face"):
            face_text = f"Face: stable={status['stable_frames']}, aligned={status['aligned']}"
        else:
            face_text = "Face: not detected"
        dpg.set_value(self._face_tag, face_text)
        
        dpg.render_dearpygui_frame()
    
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
        """Draw overlays on frame.
        
        Args:
            frame: Frame to draw on (may be resized)
            state: Current state
            status: Face tracking status
            scale_x: X scale factor from original to current frame
            scale_y: Y scale factor from original to current frame
        """
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
            # Scale bbox coordinates to match resized frame
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
            dz_x = int(25 * scale_x)  # deadzone_x
            dz_y = int(20 * scale_y)  # deadzone_y
            # Use LINE_8 (solid) instead of LINE_DASHED for compatibility
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
        if self._backend == "dpg":
            try:
                dpg.destroy_context()
            except Exception:
                pass
        elif self._backend == "cv2":
            try:
                cv2.destroyWindow(self._window_name)
            except Exception:
                pass
        self._ready = False
