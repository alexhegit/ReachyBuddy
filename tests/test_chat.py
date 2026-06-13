"""Tests for chat mode emotion controller."""

import threading
import time

import pytest

from modes.chat.emotion_controller import EmotionAnalyzer


class FakeReachy:
    """Minimal fake for testing move selection without real robot."""
    pass


class FakeLib:
    def __init__(self, moves):
        self._moves = moves

    def list_moves(self):
        return list(self._moves.keys())

    def get(self, name):
        return self._moves[name]


class TestEmotionAnalyzer:
    def test_positive(self):
        a = EmotionAnalyzer()
        emotion, intensity, level = a.analyze("I am so happy and excited today!")
        assert emotion == "positive"
        assert intensity == "medium"
        assert 0 <= level <= 1

    def test_negative(self):
        a = EmotionAnalyzer()
        emotion, intensity, _ = a.analyze("I feel sad and angry about this.")
        assert emotion == "negative"

    def test_question(self):
        a = EmotionAnalyzer()
        emotion, intensity, _ = a.analyze("Why is the sky blue?")
        assert emotion == "question"

    def test_activity(self):
        a = EmotionAnalyzer()
        emotion, intensity, _ = a.analyze("Let's dance and play!")
        assert emotion == "activity"

    def test_neutral(self):
        a = EmotionAnalyzer()
        emotion, intensity, _ = a.analyze("The object is on the table.")
        assert emotion == "neutral"
        assert intensity == "low"

    def test_emoji(self):
        a = EmotionAnalyzer()
        emotion, _, _ = a.analyze("😊👍")
        assert emotion == "positive"


class TestSpeakingActor:
    def test_start_stop(self, monkeypatch):
        from modes.chat.emotion_controller import EmotionMovePlayer, SpeakingActor

        # Patch RecordedMoves to avoid HF downloads during tests
        class FakeRecordedMoves:
            def __init__(self, *args, **kwargs):
                pass

            def list_moves(self):
                return []

            def get(self, name):
                return None

        monkeypatch.setattr(
            "modes.chat.emotion_controller.EmotionMovePlayer.__init__",
            lambda self, reachy, gentle_mode=False, debug=False: None,
        )
        monkeypatch.setattr(
            "modes.chat.emotion_controller.EmotionMovePlayer._categorize",
            lambda self: setattr(self, "emotion_to_moves", {"positive": [("emotions", "happy1")]}),
        )

        class FakeMovePlayer:
            def __init__(self):
                self.emotion_to_moves = {"positive": [("emotions", "happy1")]}
                self.gentle_mode = False
                self.debug = False
                self.calls = []

            def play(self, emotion, intensity):
                self.calls.append((emotion, intensity))

        player = FakeMovePlayer()
        actor = SpeakingActor(player, debug=False)

        actor.start("positive", "medium")
        time.sleep(0.3)
        actor.stop()

        assert len(player.calls) >= 1
        assert player.calls[0][0] == "positive"
