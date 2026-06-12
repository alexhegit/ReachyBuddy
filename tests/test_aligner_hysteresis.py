"""Test FaceAligner hysteresis (lock-hold/release) logic."""
from unittest.mock import MagicMock
import numpy as np
import pytest

from modes.cheese.app import FaceAligner
from modes.cheese.config import CheeseConfig


@pytest.fixture
def config():
    return CheeseConfig(
        deadzone_x=25, deadzone_y=20, stable_needed=5,
        lock_hold_x=35, lock_hold_y=28,
        release_x=45, release_y=35,
        ema_alpha=1.0,
    )


@pytest.fixture
def aligner(config):
    a = FaceAligner(config, debug=False)
    a._tracker.detect = MagicMock()
    return a


def make_frame(cx: int, cy: int) -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestHysteresis:
    def test_stable_frames_accumulate_to_locked(self, aligner):
        """After stable_needed frames within deadzone, aligner locks."""
        # Face exactly at center -> dx=0, dy=0. Use bbox large enough (>1% area).
        aligner._tracker.detect.return_value = (280, 200, 80, 80)
        for i in range(aligner._stable_needed):
            status = aligner.update(None, make_frame(320, 240))
        assert aligner._locked
        assert status["aligned"]

    def test_locked_maintained_within_hold_zone(self, aligner):
        """Once locked, face can drift up to lock_hold_x before releasing."""
        aligner._tracker.detect.return_value = (280, 200, 80, 80)
        for _ in range(aligner._stable_needed):
            aligner.update(None, make_frame(320, 240))
        assert aligner._locked

        # Move face right by 30px (within lock_hold_x=35)
        aligner._tracker.detect.return_value = (310, 200, 80, 80)
        # cx = 310 + 40 - 320 = 30 -> abs(30) <= 35
        aligner.update(None, make_frame(320, 240))
        assert aligner._locked

    def test_locked_released_outside_hold_zone(self, aligner):
        """Face moving beyond lock_hold_x should release."""
        aligner._tracker.detect.return_value = (280, 200, 80, 80)
        for _ in range(aligner._stable_needed):
            aligner.update(None, make_frame(320, 240))
        assert aligner._locked

        # Move face right by 50px (beyond lock_hold_x=35)
        aligner._tracker.detect.return_value = (330, 200, 80, 80)
        # cx = 330 + 40 - 320 = 50 -> abs(50) > 35
        aligner.update(None, make_frame(320, 240))
        assert not aligner._locked

    def test_locked_release_does_not_reattach_in_same_frame(self, aligner):
        """After release, stable_frames resets, alignment is false."""
        aligner._tracker.detect.return_value = (280, 200, 80, 80)
        for _ in range(aligner._stable_needed):
            aligner.update(None, make_frame(320, 240))
        assert aligner._locked

        # Large move
        aligner._tracker.detect.return_value = (400, 200, 80, 80)
        status = aligner.update(None, make_frame(320, 240))
        assert not aligner._locked
        assert not status["aligned"]
        assert status["stable_frames"] == 0
