"""Vision module for Reachy Mini Chat v9.

This module provides computer vision capabilities including:
- Face detection and tracking
- Gesture recognition
- Emotion analysis
- Visual question answering

Example:
    >>> from vision import VisionController
    >>> controller = VisionController(reachy, enabled=True)
    >>> controller.start()
    >>> face_pos = controller.face_tracker.get_position()
"""

from .controller import VisionController
from .face_tracker import FaceTracker

__all__ = ['VisionController', 'FaceTracker']
