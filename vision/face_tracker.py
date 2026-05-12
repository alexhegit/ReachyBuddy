"""Face detection and tracking using MediaPipe.

Provides real-time face detection with smoothing for stable head tracking.
"""

import os
import time
import threading
import urllib.request
from typing import Optional, Tuple
from collections import deque

import cv2


# Model download URLs for MediaPipe Tasks API
MODEL_URLS = {
    "short_range": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
    "full_range": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/1/blaze_face_full_range.tflite",
}


def get_model_path(model_selection: int = 0) -> str:
    """Get or download the face detection model.

    Args:
        model_selection: 0 for short-range (2m), 1 for long-range (5m)

    Returns:
        Path to the model file
    """
    model_key = "short_range" if model_selection == 0 else "full_range"
    model_filename = f"blaze_face_{model_key}.tflite"

    # Use ~/.cache/mediapipe for model storage
    cache_dir = os.path.expanduser("~/.cache/mediapipe")
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(cache_dir, model_filename)

    if not os.path.exists(model_path):
        print(f"      📥 Downloading face detection model ({model_key})...")
        urllib.request.urlretrieve(MODEL_URLS[model_key], model_path)
        print(f"      ✅ Model downloaded to {model_path}")

    return model_path


class FaceTracker:
    """Face detection and position tracking.

    Uses MediaPipe Face Detection for efficient CPU-based face tracking.
    Provides smoothed face center coordinates for robot head control.

    Args:
        model_selection: 0 for short-range (2m), 1 for long-range (5m)
        min_detection_confidence: Detection threshold (0.0-1.0)
        smooth_factor: EMA smoothing factor (0.0-1.0, higher = more responsive)
    """

    def __init__(
        self,
        model_selection: int = 0,
        min_detection_confidence: float = 0.5,
        smooth_factor: float = 0.25,  # High smoothness
        multi_face_strategy: str = "largest"  # "largest", "center", "leftmost"
    ):
        self.model_selection = model_selection
        self.min_detection_confidence = min_detection_confidence
        self.smooth_factor = smooth_factor
        self.multi_face_strategy = multi_face_strategy

        # MediaPipe will be imported on first use to avoid startup overhead
        self._face_detector = None
        self._mp_drawing = None
        self._Image = None
        self._ImageFormat = None

        # Tracking state
        self._current_position: Optional[Tuple[int, int]] = None
        self._last_detection_time: float = 0.0
        self._detection_timeout: float = 1.0  # seconds

        # Smoothing
        self._ema_x: Optional[float] = None
        self._ema_y: Optional[float] = None

        # Statistics
        self._fps_history = deque(maxlen=30)
        self._last_frame_time: float = 0.0

    def _init_mediapipe(self):
        """Lazy initialization of MediaPipe."""
        if self._face_detector is None:
            try:
                from mediapipe.tasks.python.core.base_options import BaseOptions
                from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions, RunningMode
                from mediapipe import Image, ImageFormat
                import mediapipe as mp

                print(f"      📦 Initializing MediaPipe FaceDetection (model={self.model_selection}, conf={self.min_detection_confidence})")

                # Get or download model
                model_path = get_model_path(self.model_selection)

                # Create detector options
                options = FaceDetectorOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    running_mode=RunningMode.IMAGE,
                    min_detection_confidence=self.min_detection_confidence,
                )

                # Create detector
                self._face_detector = FaceDetector.create_from_options(options)

                # Store Image classes for later use
                self._Image = Image
                self._ImageFormat = ImageFormat

                # Drawing utils (still available in tasks.vision)
                from mediapipe.tasks.python import vision as mp_vision
                self._mp_drawing = mp_vision.drawing_utils

                print(f"      ✅ MediaPipe initialized")
            except ImportError as e:
                raise ImportError(
                    f"MediaPipe not installed: {e}. "
                    "Run: pip install mediapipe"
                )

    def detect(self, frame) -> Optional[Tuple[int, int, int, int]]:
        """Detect face in frame and return bounding box.

        Args:
            frame: OpenCV BGR image (numpy array)

        Returns:
            Tuple of (x, y, width, height) or None if no face detected
        """
        self._init_mediapipe()

        # Validate frame
        if frame is None or frame.size == 0:
            print("      ⚠️  Invalid frame (None or empty)")
            return None

        h, w = frame.shape[:2]
        if h < 100 or w < 100:
            print(f"      ⚠️  Frame too small: {w}x{h}")
            return None

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = self._Image(image_format=self._ImageFormat.SRGB, data=rgb_frame)

        # Detect faces
        results = self._face_detector.detect(mp_image)

        if not results or not results.detections:
            return None

        # Log multiple faces
        num_faces = len(results.detections)
        if num_faces > 1:
            print(f"      👥 {num_faces} faces detected, using '{self.multi_face_strategy}' strategy")

        # Select face based on strategy
        if num_faces == 1:
            detection = results.detections[0]
        else:
            # Multiple faces - apply selection strategy
            detection = self._select_face(results.detections, w, h)

        # Get bounding box from detection result
        bbox = detection.bounding_box

        # Convert to absolute pixels
        x = int(bbox.origin_x)
        y = int(bbox.origin_y)
        width = int(bbox.width)
        height = int(bbox.height)

        return (x, y, width, height)

    def _select_face(self, detections, frame_w, frame_h):
        """Select which face to track when multiple detected.

        Strategies:
        - "largest": Biggest face (closest person)
        - "center": Face closest to image center (main subject)
        - "leftmost": Leftmost face (reading order)
        """
        if self.multi_face_strategy == "largest":
            # Select largest face by area
            largest = None
            max_area = 0
            for det in detections:
                area = det.bounding_box.width * det.bounding_box.height
                if area > max_area:
                    max_area = area
                    largest = det
            return largest

        elif self.multi_face_strategy == "center":
            # Select face closest to image center
            center_x, center_y = frame_w / 2, frame_h / 2
            closest = None
            min_dist = float('inf')
            for det in detections:
                bbox = det.bounding_box
                face_cx = bbox.origin_x + bbox.width / 2
                face_cy = bbox.origin_y + bbox.height / 2
                dist = ((face_cx - center_x) ** 2 + (face_cy - center_y) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    closest = det
            return closest

        else:  # "leftmost" or default
            # Select leftmost face
            leftmost = None
            min_x = float('inf')
            for det in detections:
                bbox = det.bounding_box
                x = bbox.origin_x
                if x < min_x:
                    min_x = x
                    leftmost = det
            return leftmost

    def get_face_center(self, frame) -> Optional[Tuple[int, int]]:
        """Get smoothed face center coordinates.

        Args:
            frame: OpenCV BGR image

        Returns:
            Tuple of (x, y) pixel coordinates or None
        """
        bbox = self.detect(frame)

        if bbox is None:
            # Check if we should clear stale position
            if time.time() - self._last_detection_time > self._detection_timeout:
                self._current_position = None
                self._ema_x = None
                self._ema_y = None
            return self._current_position

        x, y, w, h = bbox
        center_x = x + w // 2
        center_y = y + h // 2

        # Apply EMA smoothing
        if self._ema_x is None:
            self._ema_x = center_x
            self._ema_y = center_y
        else:
            self._ema_x = (
                self.smooth_factor * center_x +
                (1 - self.smooth_factor) * self._ema_x
            )
            self._ema_y = (
                self.smooth_factor * center_y +
                (1 - self.smooth_factor) * self._ema_y
            )

        self._current_position = (int(self._ema_x), int(self._ema_y))
        self._last_detection_time = time.time()

        return self._current_position

    def get_position(self) -> Optional[Tuple[int, int]]:
        """Get last known face position without processing new frame.

        Returns:
            Last smoothed (x, y) or None if no recent detection
        """
        # Check for timeout
        if time.time() - self._last_detection_time > self._detection_timeout:
            self._current_position = None
        return self._current_position

    def is_face_detected(self) -> bool:
        """Check if face is currently tracked."""
        return self.get_position() is not None

    def update_fps(self):
        """Update FPS calculation. Call once per frame."""
        current_time = time.time()
        if self._last_frame_time > 0:
            fps = 1.0 / (current_time - self._last_frame_time)
            self._fps_history.append(fps)
        self._last_frame_time = current_time

    def get_fps(self) -> float:
        """Get average FPS over last 30 frames."""
        if not self._fps_history:
            return 0.0
        return sum(self._fps_history) / len(self._fps_history)

    def close(self):
        """Release resources."""
        if self._face_detector is not None:
            self._face_detector.close()


class FaceTrackerThread(threading.Thread):
    """Background thread for continuous face tracking.

    Runs face detection in a separate thread to avoid blocking
    the main chat loop.

    Args:
        camera: Callable that returns current frame
        tracker: FaceTracker instance
        callback: Optional callback(face_position) on each detection
    """

    def __init__(
        self,
        camera,
        tracker: FaceTracker,
        callback=None,
        fps_target: float = 15.0
    ):
        super().__init__(daemon=True)
        self.camera = camera
        self.tracker = tracker
        self.callback = callback
        self.fps_target = fps_target
        self._running = False
        self._frame_interval = 1.0 / fps_target

    def run(self):
        """Main tracking loop."""
        self._running = True

        while self._running:
            start_time = time.time()

            # Get frame from camera
            frame = self.camera()
            if frame is not None:
                # Track face
                pos = self.tracker.get_face_center(frame)
                self.tracker.update_fps()

                # Notify callback
                if self.callback and pos:
                    self.callback(pos)

            # Maintain target FPS
            elapsed = time.time() - start_time
            sleep_time = self._frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self):
        """Stop the tracking thread."""
        self._running = False
