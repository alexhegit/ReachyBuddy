"""Hand pointing tracker using MediaPipe Hands.

Detects index finger pointing gestures and maps to screen coordinates
for robot gaze control.
"""

import time
from typing import Optional, Tuple, List
from dataclasses import dataclass

import cv2


@dataclass
class PointingResult:
    """Result of pointing detection.
    
    Attributes:
        index_tip: (x, y) pixel coordinates of index finger tip
        index_pip: (x, y) pixel coordinates of index finger PIP joint
        hand_side: 'Left' or 'Right'
        confidence: Detection confidence (0.0-1.0)
        pointing_vector: Normalized direction vector (dx, dy)
    """
    index_tip: Tuple[int, int]
    index_pip: Tuple[int, int]
    hand_side: str
    confidence: float
    pointing_vector: Tuple[float, float]


class PointTracker:
    """Track index finger pointing gesture.
    
    MediaPipe Hand Landmarks (21 points):
        0: Wrist
        5-8: Index finger (MCP, PIP, DIP, TIP)
        9-12: Middle finger
        13-16: Ring finger  
        17-20: Pinky
    
    Index tip is landmark #8, PIP is #6.
    
    Args:
        mode: 'strict' requires only index extended, 'loose' accepts any index extended
        min_detection_confidence: Detection threshold
        tip_extension_ratio: How much longer tip must be vs PIP (default 1.1, lower = easier)
    """
    
    # MediaPipe hand landmark indices
    WRIST = 0
    INDEX_MCP = 5  # Base of index finger
    INDEX_PIP = 6  # First joint
    INDEX_DIP = 7  # Second joint  
    INDEX_TIP = 8  # Finger tip
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    RING_MCP = 13
    RING_PIP = 14
    PINKY_MCP = 17
    PINKY_PIP = 18
    
    def __init__(
        self,
        mode: str = 'loose',  # 'strict' or 'loose'
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        tip_extension_ratio: float = 1.1  # Lower threshold = easier detection
    ):
        self.mode = mode
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.tip_extension_ratio = tip_extension_ratio
        
        self._hands = None
        self._mp_drawing = None
        
        # Tracking state
        self._last_pointing_time: float = 0.0
        self._pointing_timeout: float = 0.3  # seconds
        
    def _init_mediapipe(self):
        """Lazy initialization of MediaPipe Hands."""
        if self._hands is None:
            try:
                import mediapipe as mp
                self._hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,  # Track one hand for pointing
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence
                )
                self._mp_drawing = mp.solutions.drawing_utils
            except ImportError as e:
                raise ImportError(
                    f"MediaPipe not installed: {e}. Run: pip install mediapipe"
                )
    
    def _is_finger_extended(
        self, 
        landmarks, 
        tip_idx: int, 
        pip_idx: int,
        mcp_idx: int
    ) -> bool:
        """Check if a finger is extended.
        
        A finger is extended if tip is farther from wrist than PIP joint.
        """
        wrist = landmarks[self.WRIST]
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]
        
        # Calculate distances from wrist (using squared distance for efficiency)
        def dist_sq(p1, p2):
            return (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2
        
        tip_to_wrist = dist_sq(tip, wrist)
        pip_to_wrist = dist_sq(pip, wrist)
        
        # Safety check: avoid division by zero
        if pip_to_wrist < 0.0001:
            return False
        
        # Tip should be farther than PIP from wrist
        # Using ratio: tip distance / pip distance > threshold
        ratio = tip_to_wrist / pip_to_wrist
        return ratio > self.tip_extension_ratio
    
    def _is_pointing_gesture(self, landmarks, debug: bool = False) -> bool:
        """Check if hand is making pointing gesture.
        
        Returns True if index finger is extended.
        In 'strict' mode, other fingers must be curled.
        """
        # Check index finger is extended
        index_extended = self._is_finger_extended(
            landmarks, self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP
        )
        
        if debug:
            print(f"        Index extended: {index_extended}")
        
        if not index_extended:
            return False
        
        if self.mode == 'strict':
            # In strict mode, other fingers should NOT be extended
            middle_extended = self._is_finger_extended(
                landmarks, 12, 10, 9  # Middle finger
            )
            ring_extended = self._is_finger_extended(
                landmarks, 16, 14, 13  # Ring finger
            )
            pinky_extended = self._is_finger_extended(
                landmarks, 20, 18, 17  # Pinky
            )
            
            if debug:
                print(f"        Middle: {middle_extended}, Ring: {ring_extended}, Pinky: {pinky_extended}")
            
            # Only index should be extended
            return not (middle_extended or ring_extended or pinky_extended)
        
        # Loose mode: just need index extended
        return True
    
    def detect_pointing(self, frame, debug: bool = False) -> Optional[PointingResult]:
        """Detect pointing gesture in frame.
        
        Args:
            frame: OpenCV BGR image
            debug: Print debug info
            
        Returns:
            PointingResult or None if no pointing detected
        """
        self._init_mediapipe()
        
        import mediapipe as mp
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self._hands.process(rgb_frame)
        
        if not results or not results.multi_hand_landmarks:
            if debug:
                print("      🔍 No hand detected by MediaPipe")
            return None
        
        if debug:
            print(f"      🔍 MediaPipe found {len(results.multi_hand_landmarks)} hand(s)")
        
        # Get the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = hand_landmarks.landmark
        
        # Check for pointing gesture
        is_pointing = self._is_pointing_gesture(landmarks, debug=debug)
        
        if not is_pointing:
            return None
        
        # Get hand info
        handedness = results.multi_handedness[0].classification[0]
        hand_side = handedness.label  # 'Left' or 'Right'
        confidence = handedness.score
        
        # Get index finger coordinates
        h, w = frame.shape[:2]
        index_tip = landmarks[self.INDEX_TIP]
        index_pip = landmarks[self.INDEX_PIP]
        
        tip_x = int(index_tip.x * w)
        tip_y = int(index_tip.y * h)
        pip_x = int(index_pip.x * w)
        pip_y = int(index_pip.y * h)
        
        # Calculate pointing direction vector (normalized)
        dx = index_tip.x - index_pip.x
        dy = index_tip.y - index_pip.y
        length = (dx ** 2 + dy ** 2) ** 0.5
        if length > 0:
            dx, dy = dx / length, dy / length
        
        self._last_pointing_time = time.time()
        
        return PointingResult(
            index_tip=(tip_x, tip_y),
            index_pip=(pip_x, pip_y),
            hand_side=hand_side,
            confidence=confidence,
            pointing_vector=(dx, dy)
        )
    
    def is_pointing(self) -> bool:
        """Check if pointing was detected recently."""
        return (time.time() - self._last_pointing_time) < self._pointing_timeout
    
    def get_pointing_direction(self, result: PointingResult) -> str:
        """Get cardinal direction of pointing.
        
        Returns: 'up', 'down', 'left', 'right', 'up-left', etc.
        """
        dx, dy = result.pointing_vector
        
        # Determine primary direction
        directions = []
        if dy < -0.5:
            directions.append('up')
        elif dy > 0.5:
            directions.append('down')
        if dx < -0.5:
            directions.append('left')
        elif dx > 0.5:
            directions.append('right')
        
        return '-'.join(directions) if directions else 'center'
    
    def draw_debug(
        self, 
        frame, 
        result: Optional[PointingResult] = None
    ):
        """Draw debug visualization on frame."""
        if result is None or self._hands is None:
            return frame
        
        import mediapipe as mp
        
        # This would need the original hand_landmarks which we don't store
        # For now, just draw a circle at index tip
        cv2.circle(frame, result.index_tip, 10, (0, 255, 0), -1)
        cv2.circle(frame, result.index_pip, 8, (255, 0, 0), -1)
        
        # Draw pointing direction
        tip_x, tip_y = result.index_tip
        dx, dy = result.pointing_vector
        end_x = int(tip_x + dx * 50)
        end_y = int(tip_y + dy * 50)
        cv2.arrowedLine(frame, (tip_x, tip_y), (end_x, end_y), (0, 0, 255), 3)
        
        # Label
        label = f"{result.hand_side} hand pointing"
        cv2.putText(frame, label, (tip_x + 15, tip_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def close(self):
        """Release resources."""
        if self._hands is not None:
            self._hands.close()
