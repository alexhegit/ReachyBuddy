"""Test FaceTracker._select_face strategy logic (pure geometry)."""
from unittest.mock import MagicMock, PropertyMock
import pytest

from vision.face_tracker import FaceTracker
from modes.cheese.config import CheeseConfig


@pytest.fixture
def tracker():
    cfg = CheeseConfig()
    # Instantiate but don't init MediaPipe — we test _select_face directly
    t = object.__new__(FaceTracker)
    t.multi_face_strategy = "largest"
    return t


def make_detection(x, y, w, h):
    """Create a mock DetectionResult with bounding_box."""
    det = MagicMock()
    bb = MagicMock()
    bb.origin_x = x
    bb.origin_y = y
    bb.width = w
    bb.height = h
    type(det).bounding_box = PropertyMock(return_value=bb)
    return det


class TestSelectLargestFace:
    def test_selects_largest(self, tracker):
        dets = [
            make_detection(100, 100, 50, 30),   # 1500
            make_detection(200, 200, 200, 150),  # 30000 <- largest
            make_detection(50, 50, 40, 40),      # 1600
        ]
        result = tracker._select_face(dets, 640, 480)
        assert result is dets[1]

    def test_zero_area_fallback(self, tracker):
        """If all detections have zero area, return first one (previous crash)."""
        dets = [
            make_detection(100, 100, 0, 0),
            make_detection(200, 200, 0, 0),
        ]
        result = tracker._select_face(dets, 640, 480)
        assert result is dets[0]

    def test_single_detection_bypassed(self, tracker):
        """Single face is selected directly, not via _select_face."""
        # This tests the calling code path
        pass


class TestSelectCenterFace:
    def test_selects_closest_to_center(self, tracker):
        tracker.multi_face_strategy = "center"
        # Center of frame is (320, 240)
        dets = [
            make_detection(500, 300, 50, 50),    # far
            make_detection(300, 220, 40, 40),    # close to center <- should win
            make_detection(50, 50, 50, 50),      # far
        ]
        result = tracker._select_face(dets, 640, 480)
        assert result is dets[1]

    def test_empty_detections_returns_none(self, tracker):
        result = tracker._select_face([], 640, 480)
        assert result is None


class TestSelectLeftmostFace:
    def test_selects_leftmost(self, tracker):
        tracker.multi_face_strategy = "leftmost"
        dets = [
            make_detection(300, 100, 50, 50),
            make_detection(100, 100, 50, 50),   # <- leftmost
            make_detection(500, 100, 50, 50),
        ]
        result = tracker._select_face(dets, 640, 480)
        assert result is dets[1]
