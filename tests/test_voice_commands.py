"""Test voice command phrase matching (pure string logic)."""
import pytest
from modes.cheese.app import CheeseModeApp
from modes.cheese.config import CheeseConfig


@pytest.fixture
def app():
    cfg = CheeseConfig(debug=False)
    return CheeseModeApp(cfg)


class TestIsWakePhrase:
    def test_wake_word_reachy(self, app):
        assert app._is_wake_phrase("hey reachy")

    def test_wake_word_ricky(self, app):
        assert app._is_wake_phrase("ricky come here")

    def test_wake_word_richie(self, app):
        assert app._is_wake_phrase("richie")

    def test_wake_word_reaching(self, app):
        assert app._is_wake_phrase("reaching")

    def test_no_false_positive(self, app):
        assert not app._is_wake_phrase("hello world")

    def test_punctuation_ignored(self, app):
        assert app._is_wake_phrase("reachy! come here")

    def test_mixed_case(self, app):
        assert app._is_wake_phrase("ReAcHy")

    def test_substring_not_enough(self, app):
        """'reach' should not match 'reachy'."""
        assert not app._is_wake_phrase("breach")

    def test_custom_wake_word(self, app):
        app.cfg.wake_word = "buddy"
        assert app._is_wake_phrase("hey buddy")
        assert not app._is_wake_phrase("hey reachy")


class TestIsSleepPhrase:
    def test_sleep(self, app):
        assert app._is_sleep_phrase("go to sleep")

    def test_stop(self, app):
        assert app._is_sleep_phrase("stop")

    def test_cancel(self, app):
        assert app._is_sleep_phrase("cancel that")

    def test_no_false_positive(self, app):
        assert not app._is_sleep_phrase("take photo")


class TestIsCapturePhrase:
    def test_cheese(self, app):
        assert app._is_capture_phrase("say cheese")

    def test_take_photo(self, app):
        assert app._is_capture_phrase("take photo now")

    def test_photo(self, app):
        assert app._is_capture_phrase("photo time")

    def test_picture(self, app):
        assert app._is_capture_phrase("take a picture")

    def test_no_false_positive(self, app):
        assert not app._is_capture_phrase("stop")
