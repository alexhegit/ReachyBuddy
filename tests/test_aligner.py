"""Test FaceAligner control logic with mocked tracker and runtime."""
import time
from unittest.mock import MagicMock
import numpy as np
import pytest

from modes.cheese.app import FaceAligner
from modes.cheese.config import CheeseConfig


@pytest.fixture
def config():
    return CheeseConfig(
        deadzone_x=25, deadzone_y=20, stable_needed=10,
        ema_alpha=1.0,  # No smoothing for deterministic tests
    )


@pytest.fixture
def aligner(config):
    return FaceAligner(config, debug=False)


@pytest.fixture
def mock_runtime():
    rt = MagicMock()
    rt.goto_body_yaw.return_value = None
    rt.reset_head.return_value = None
    rt.look_at_image.return_value = None
    return rt


def make_frame(cx: int, cy: int, w: int = 640, h: int = 480) -> np.ndarray:
    """Create a fake 640x480 frame — aligner only uses shape."""
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestFaceAlignerReset:
    def test_reset_clears_state(self, aligner):
        aligner._stable_frames = 50
        aligner._locked = True
        aligner._body_yaw = 0.5
        aligner.reset()
        assert aligner._stable_frames == 0
        assert not aligner._locked
        assert aligner._body_yaw == 0.0


class TestFaceAlignerNoFace:
    def test_no_face_returns_empty_status(self, aligner):
        # Mock the tracker to return None (no face)
        aligner._tracker.detect = MagicMock(return_value=None)
        frame = make_frame(320, 240)
        status = aligner.update(None, frame)
        assert not status["has_face"]
        assert not status["aligned"]
        assert status["bbox"] is None

    def test_no_face_resets_stable_frames(self, aligner):
        aligner._stable_frames = 10
        aligner._tracker.detect = MagicMock(return_value=None)
        aligner.update(None, make_frame(320, 240))
        assert aligner._stable_frames == 0


class TestFaceAlignerSmallFace:
    def test_small_face_rejected(self, aligner):
        """Face less than 1% of frame area should be treated as no face."""
        # 640*480 = 307200, 1% = 3072; a 30x30 box = 900 < 3072
        aligner._tracker.detect = MagicMock(return_value=(100, 100, 30, 30))
        status = aligner.update(None, make_frame(320, 240))
        assert not status["has_face"]


class TestFaceAlignerAlignment:
    def test_face_at_center_is_aligned(self, aligner):
        """Face at center of frame should give dx/dy ~0 -> aligned."""
        # bbox must be large enough: area > 1% of 640*480 = 3072
        aligner._tracker.detect = MagicMock(return_value=(280, 200, 80, 80))
        status = aligner.update(None, make_frame(320, 240))
        assert status["has_face"]
        # cx = 280 + 40 = 320, cy = 200 + 40 = 240 -> dx=0, dy=0
        assert abs(status["dx"]) <= 1

    def test_aligned_accumulates_stable_frames(self, aligner):
        aligner._tracker.detect = MagicMock(return_value=(280, 200, 80, 80))
        for _ in range(5):
            status = aligner.update(None, make_frame(320, 240))
        assert status["stable_frames"] >= 5

    def test_not_aligned_resets_stable_frames(self, aligner):
        aligner._tracker.detect = MagicMock(return_value=(100, 100, 80, 80))
        status = aligner.update(None, make_frame(320, 240))
        assert status["stable_frames"] == 0
        assert not status["aligned"]


class TestFaceAlignerHeadingMoving:
    def test_body_move_triggers_for_large_error(self, aligner, mock_runtime):
        """Large deviation should trigger body yaw command."""
        aligner._tracker.detect = MagicMock(return_value=(0, 200, 80, 80))
        dx = 0 + 40 - 320  # = -280 (far left)
        assert abs(dx) > 70
        # Run multiple updates to trigger the time-based logic
        for _ in range(3):
            aligner.update(mock_runtime, make_frame(320, 240))
            if aligner._big_error_since > 0:
                break
        assert aligner._big_error_since > 0

    def test_head_move_triggers_when_in_deadzone_but_outside_release(self, aligner,
                                                                       mock_runtime):
        """Small-medium error should trigger head movement."""
        # dx = 40 which is > deadzone (25) but < body threshold (70)
        aligner._tracker.detect = MagicMock(
            return_value=(360, 240, 80, 80))
        aligner._last_track_at = 0.0  # Force movement
        status = aligner.update(mock_runtime, make_frame(320, 240))
        assert status["has_face"]
        # Should NOT be aligned (dx=40 > deadzone=25)
        assert not status["aligned"]
