"""Vision module for Reachy Mini Chat v9.

This module provides computer vision capabilities including:
- Face detection and tracking
- Gesture recognition
- Emotion analysis
- Visual question answering

Example:
    >>> from vision import VisionController, VisionConfig
    >>> config = VisionConfig(enabled=True, face_tracking=True)
    >>> controller = VisionController(reachy, config)
    >>> controller.start()
    >>> face_pos = controller.get_face_position()
"""

from .controller import VisionController, VisionConfig, VisionState
from .face_tracker import FaceTracker, FaceTrackerThread

__all__ = [
    'VisionController',
    'VisionConfig',
    'VisionState',
    'FaceTracker',
    'FaceTrackerThread',
]
