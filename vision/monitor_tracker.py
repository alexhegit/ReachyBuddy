"""Security monitoring using camera feed.

Provides motion detection, person presence monitoring, and anomaly detection
for security surveillance purposes.
"""

import time
import threading
from typing import Optional, Callable, List
from dataclasses import dataclass
from datetime import datetime
from collections import deque

import cv2
import numpy as np


@dataclass
class MonitorEvent:
    """Security event record.
    
    Attributes:
        timestamp: When the event occurred
        event_type: 'motion', 'person_enter', 'person_leave', 'anomaly'
        confidence: Detection confidence (0.0-1.0)
        description: Human-readable description
        snapshot: Optional frame snapshot
    """
    timestamp: datetime
    event_type: str
    confidence: float
    description: str
    snapshot: Optional[np.ndarray] = None


class MonitorTracker:
    """Security monitor using motion detection and person detection.
    
    Features:
    - Motion detection using frame differencing
    - Person presence monitoring (using existing face tracker)
    - Anomaly detection (sudden light changes, camera obstruction)
    - Event logging with timestamps
    - Optional video recording of events
    
    Args:
        motion_threshold: Sensitivity for motion detection (default 25)
        min_motion_area: Minimum contour area to trigger motion (default 500)
        cooldown_seconds: Minimum time between events (default 5)
        buffer_seconds: Seconds to buffer before motion event (default 3)
    """
    
    def __init__(
        self,
        motion_threshold: int = 25,
        min_motion_area: int = 500,
        cooldown_seconds: float = 5.0,
        buffer_seconds: float = 3.0,
        enable_recording: bool = False,
        recording_dir: str = "./monitor_recordings"
    ):
        self.motion_threshold = motion_threshold
        self.min_motion_area = min_motion_area
        self.cooldown_seconds = cooldown_seconds
        self.buffer_seconds = buffer_seconds
        self.enable_recording = enable_recording
        self.recording_dir = recording_dir
        
        # State
        self._running = False
        self._background_frame: Optional[np.ndarray] = None
        self._frame_buffer: deque = deque(maxlen=int(buffer_seconds * 10))  # 10 FPS
        
        # Event tracking
        self._last_motion_time: float = 0.0
        self._last_person_time: float = 0.0
        self._person_present: bool = False
        
        # Event callbacks
        self.on_motion: Optional[Callable] = None
        self.on_person_enter: Optional[Callable] = None
        self.on_person_leave: Optional[Callable] = None
        self.on_anomaly: Optional[Callable] = None
        
        # Event history
        self._events: List[MonitorEvent] = []
        self._max_events: int = 1000
        
    def start(self):
        """Start monitoring."""
        self._running = True
        print("   🔒 Security monitoring started")
        
    def stop(self):
        """Stop monitoring."""
        self._running = False
        print("   🔒 Security monitoring stopped")
        
    def detect_motion(self, frame: np.ndarray) -> bool:
        """Detect motion using frame differencing.
        
        Returns True if significant motion detected.
        """
        if frame is None:
            return False
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Initialize background
        if self._background_frame is None:
            self._background_frame = gray
            return False
        
        # Frame differencing
        delta = cv2.absdiff(self._background_frame, gray)
        thresh = cv2.threshold(delta, self.motion_threshold, 255, cv2.THRESH_BINARY)[1]
        
        # Dilate to fill holes
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check for significant motion
        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > self.min_motion_area:
                motion_detected = True
                break
        
        # Update background (slow adaptation)
        self._background_frame = cv2.addWeighted(
            self._background_frame, 0.95, gray, 0.05, 0
        )
        
        return motion_detected
    
    def detect_anomaly(self, frame: np.ndarray) -> Optional[str]:
        """Detect anomalies like sudden light changes or camera obstruction.
        
        Returns anomaly type or None.
        """
        if frame is None:
            return None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        # Check for sudden darkness (obstruction/disconnect)
        if mean_brightness < 20:
            return "darkness"
        
        # Check for sudden brightness (light flash)
        if hasattr(self, '_last_brightness'):
            brightness_change = abs(mean_brightness - self._last_brightness)
            if brightness_change > 80:
                return "light_change"
        
        self._last_brightness = mean_brightness
        return None
    
    def process_frame(
        self, 
        frame: np.ndarray, 
        person_detected: bool = False
    ) -> Optional[MonitorEvent]:
        """Process a frame and detect security events.
        
        Args:
            frame: Current camera frame
            person_detected: Whether a person was detected (from face tracker)
            
        Returns:
            MonitorEvent if event detected, None otherwise
        """
        if not self._running:
            return None
        
        current_time = time.time()
        self._frame_buffer.append((current_time, frame.copy()))
        
        event: Optional[MonitorEvent] = None
        
        # Check for motion
        if self.detect_motion(frame):
            if current_time - self._last_motion_time > self.cooldown_seconds:
                self._last_motion_time = current_time
                event = MonitorEvent(
                    timestamp=datetime.now(),
                    event_type='motion',
                    confidence=0.8,
                    description='Motion detected in monitored area',
                    snapshot=frame.copy()
                )
                if self.on_motion:
                    self.on_motion(event)
        
        # Check for person enter/leave
        if person_detected != self._person_present:
            self._person_present = person_detected
            if person_detected:
                event = MonitorEvent(
                    timestamp=datetime.now(),
                    event_type='person_enter',
                    confidence=0.9,
                    description='Person entered monitored area',
                    snapshot=frame.copy()
                )
                if self.on_person_enter:
                    self.on_person_enter(event)
            else:
                event = MonitorEvent(
                    timestamp=datetime.now(),
                    event_type='person_leave',
                    confidence=0.9,
                    description='Person left monitored area',
                    snapshot=frame.copy()
                )
                if self.on_person_leave:
                    self.on_person_leave(event)
        
        # Check for anomalies
        anomaly = self.detect_anomaly(frame)
        if anomaly:
            event = MonitorEvent(
                timestamp=datetime.now(),
                event_type='anomaly',
                confidence=0.7,
                description=f'Anomaly detected: {anomaly}',
                snapshot=frame.copy()
            )
            if self.on_anomaly:
                self.on_anomaly(event)
        
        # Store event
        if event:
            self._events.append(event)
            if len(self._events) > self._max_events:
                self._events.pop(0)
        
        return event
    
    def get_recent_events(self, count: int = 10) -> List[MonitorEvent]:
        """Get recent security events."""
        return self._events[-count:]
    
    def get_event_stats(self) -> dict:
        """Get event statistics."""
        stats = {
            'total_events': len(self._events),
            'motion_count': sum(1 for e in self._events if e.event_type == 'motion'),
            'person_enter_count': sum(1 for e in self._events if e.event_type == 'person_enter'),
            'person_leave_count': sum(1 for e in self._events if e.event_type == 'person_leave'),
            'anomaly_count': sum(1 for e in self._events if e.event_type == 'anomaly'),
        }
        return stats
    
    def save_snapshot(self, event: MonitorEvent, filepath: str):
        """Save event snapshot to file."""
        if event.snapshot is not None:
            cv2.imwrite(filepath, event.snapshot)
