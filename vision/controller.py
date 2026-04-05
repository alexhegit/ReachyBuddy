"""Vision controller for Reachy Mini Chat.

Main interface for vision capabilities. Manages camera access and
provides high-level vision events to the chat application.
"""

import time
import threading
from typing import Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

from .face_tracker import FaceTracker, FaceTrackerThread


class VisionState(Enum):
    """Vision system states."""
    IDLE = "idle"
    TRACKING = "tracking"
    PERSON_PRESENT = "person_present"
    NO_PERSON = "no_person"


@dataclass
class VisionConfig:
    """Configuration for vision controller.
    
    Attributes:
        enabled: Master switch for vision features
        face_tracking: Enable face detection and tracking
        target_fps: Target camera processing FPS
        detection_timeout: Seconds before considering person left
        auto_wake: Wake robot when person detected
        track_while_speaking: Continue tracking during TTS
    """
    enabled: bool = True
    face_tracking: bool = True
    target_fps: float = 15.0
    detection_timeout: float = 2.0
    auto_wake: bool = True
    track_while_speaking: bool = True


class VisionController:
    """Main controller for vision capabilities.
    
    Manages camera access, face tracking, and provides event callbacks
    for integration with the chat application.
    
    Args:
        reachy: ReachyMini instance for camera access
        config: VisionConfig settings
    
    Example:
        >>> controller = VisionController(reachy)
        >>> controller.start()
        >>> # In animation loop:
        >>> if pos := controller.get_face_position():
        ...     reachy.look_at_image(pos[0], pos[1])
    """
    
    def __init__(self, reachy, config: Optional[VisionConfig] = None):
        self.reachy = reachy
        self.config = config or VisionConfig()
        
        # State
        self.state = VisionState.IDLE
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Face tracking
        self.face_tracker: Optional[FaceTracker] = None
        self._face_position: Optional[Tuple[int, int]] = None
        self._last_detection_time: float = 0.0
        
        # Camera access
        self._camera = None
        self._init_camera()
        
        # Callbacks
        self.on_face_detected: Optional[Callable] = None
        self.on_person_enter: Optional[Callable] = None
        self.on_person_leave: Optional[Callable] = None
        
    def _init_camera(self):
        """Initialize camera access via Reachy's media manager."""
        if not self.config.enabled:
            return
            
        try:
            # Access camera through Reachy's media manager
            self._camera = self.reachy.media
            if self._camera is None:
                print("⚠️ Vision: Camera not available")
                self.config.enabled = False
        except Exception as e:
            print(f"⚠️ Vision: Failed to access camera: {e}")
            self.config.enabled = False
    
    def _get_frame(self):
        """Get current camera frame."""
        if self._camera is None:
            return None
        try:
            return self._camera.get_frame()
        except Exception:
            return None
    
    def start(self):
        """Start vision processing."""
        if not self.config.enabled or self._running:
            return
        
        print("👁️  Starting vision controller...")
        
        # Initialize face tracker
        if self.config.face_tracking:
            self.face_tracker = FaceTracker(
                model_selection=0,  # Short range for desktop robot
                min_detection_confidence=0.5,
                smooth_factor=0.3
            )
        
        self._running = True
        self.state = VisionState.TRACKING
        
        # Start background processing thread
        self._thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._thread.start()
        
        print(f"   ✅ Face tracking: {self.config.face_tracking}")
        print(f"   ✅ Target FPS: {self.config.target_fps}")
    
    def stop(self):
        """Stop vision processing."""
        if not self._running:
            return
        
        print("👁️  Stopping vision controller...")
        self._running = False
        
        if self._thread:
            self._thread.join(timeout=2.0)
        
        if self.face_tracker:
            self.face_tracker.close()
        
        self.state = VisionState.IDLE
        print("   ✅ Vision stopped")
    
    def _processing_loop(self):
        """Main vision processing loop."""
        frame_interval = 1.0 / self.config.target_fps
        frame_count = 0
        
        print("   🎥 Vision processing started")
        
        while self._running:
            start_time = time.time()
            
            # Get frame
            frame = self._get_frame()
            if frame is None:
                if frame_count == 0:
                    print("   ⚠️  No frame available from camera")
                time.sleep(frame_interval)
                continue
            
            frame_count += 1
            
            # Log first successful frame capture
            if frame_count == 1:
                print(f"   ✅ First frame captured: {frame.shape}", flush=True)
            
            # Process face tracking
            if self.config.face_tracking and self.face_tracker:
                pos = self.face_tracker.get_face_center(frame)
                
                if pos:
                    self._face_position = pos
                    self._last_detection_time = time.time()
                    
                    # Log first face detection (only first 3 times, then throttle)
                    if frame_count <= 3:
                        print(f"   👤 Face detected at ({pos[0]}, {pos[1]})", flush=True)
                    
                    # Trigger callback
                    if self.on_face_detected:
                        self.on_face_detected(pos)
                    
                    # Check state transition
                    if self.state == VisionState.NO_PERSON:
                        self.state = VisionState.PERSON_PRESENT
                        if self.on_person_enter:
                            self.on_person_enter()
                else:
                    # Check timeout
                    elapsed = time.time() - self._last_detection_time
                    if elapsed > self.config.detection_timeout:
                        if self.state == VisionState.PERSON_PRESENT:
                            self.state = VisionState.NO_PERSON
                            self._face_position = None
                            if self.on_person_leave:
                                self.on_person_leave()
            
            # Maintain FPS
            elapsed = time.time() - start_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def get_face_position(self) -> Optional[Tuple[int, int]]:
        """Get current face position (x, y) in pixel coordinates.
        
        Returns:
            (x, y) tuple or None if no face detected
        """
        return self._face_position
    
    def is_person_present(self) -> bool:
        """Check if a person is currently detected.
        
        Returns True if face position is available (more reliable than state check).
        """
        # Check if we have a recent face position
        if self._face_position is not None:
            elapsed = time.time() - self._last_detection_time
            if elapsed < self.config.detection_timeout:
                return True
        return self.state == VisionState.PERSON_PRESENT
    
    def get_fps(self) -> float:
        """Get current vision processing FPS."""
        if self.face_tracker:
            return self.face_tracker.get_fps()
        return 0.0
    
    def look_at_face(self, duration: float = 0.3):
        """Convenience method: make robot look at detected face.
        
        Args:
            duration: Movement duration in seconds
            
        Returns:
            True if face was detected and movement triggered
        """
        if pos := self.get_face_position():
            try:
                self.reachy.look_at_image(pos[0], pos[1], duration=duration)
                return True
            except Exception as e:
                print(f"⚠️ Vision: Failed to look at face: {e}")
        return False
