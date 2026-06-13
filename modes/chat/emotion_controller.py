"""Lightweight emotion analysis and robot actions for chat mode.

Extracted and refactored from ReachyMiniChat's EmotionControllerV6 to keep the
chat mode codebase small and focused.
"""

from __future__ import annotations

import random
import threading
import time
from typing import Dict, List, Optional, Tuple


# Maps recorded move names from Pollen libraries to emotion categories.
_EMOTION_CATEGORY_MAP: Dict[str, str] = {
    # positive
    "amazed1": "positive",
    "cheerful1": "positive",
    "dance1": "positive",
    "dance2": "positive",
    "dance3": "positive",
    "enthusiastic1": "positive",
    "enthusiastic2": "positive",
    "grateful1": "positive",
    "helpful1": "positive",
    "helpful2": "positive",
    "laughing1": "positive",
    "laughing2": "positive",
    "loving1": "positive",
    "proud1": "positive",
    "proud2": "positive",
    "proud3": "positive",
    "relief1": "positive",
    "relief2": "positive",
    "success1": "positive",
    "success2": "positive",
    "welcoming1": "positive",
    "welcoming2": "positive",
    "yes1": "positive",
    "understanding2": "positive",
    "electric1": "positive",
    # negative
    "anxiety1": "negative",
    "boredom1": "negative",
    "boredom2": "negative",
    "contempt1": "negative",
    "displeased1": "negative",
    "displeased2": "negative",
    "downcast1": "negative",
    "disgusted1": "negative",
    "dying1": "negative",
    "exhausted1": "negative",
    "fear1": "negative",
    "frustrated1": "negative",
    "furious1": "negative",
    "go_away1": "negative",
    "impatient1": "negative",
    "impatient2": "negative",
    "irritated1": "negative",
    "irritated2": "negative",
    "lonely1": "negative",
    "no1": "negative",
    "no_sad1": "negative",
    "rage1": "negative",
    "sad1": "negative",
    "sad2": "negative",
    "scared1": "negative",
    "tired1": "negative",
    "reprimand1": "negative",
    "reprimand2": "negative",
    "reprimand3": "negative",
    "calming1": "negative",
    "yes_sad1": "negative",
    "resigned1": "negative",
    # question
    "confused1": "question",
    "curious1": "question",
    "incomprehensible2": "question",
    "inquiring1": "question",
    "inquiring2": "question",
    "inquiring3": "question",
    "lost1": "question",
    "thoughtful1": "question",
    "thoughtful2": "question",
    "uncertain1": "question",
    "uncomfortable1": "question",
    # activity
    "no_excited1": "activity",
    "serenity1": "activity",
    # neutral
    "attentive1": "neutral",
    "attentive2": "neutral",
    "come1": "neutral",
    "indifferent1": "neutral",
    "understanding1": "neutral",
    "oops1": "neutral",
    "oops2": "neutral",
    "shy1": "neutral",
    "sleep1": "neutral",
    "proud1": "neutral",
    "surprised1": "neutral",
    "surprised2": "neutral",
}

_GENTLE_EMOTIONS = {
    "attentive1", "attentive2", "understanding1", "understanding2",
    "shy1", "come1", "indifferent1",
    "grateful1", "helpful1", "helpful2", "relief1", "relief2",
    "yes1", "welcoming1",
    "thoughtful1", "thoughtful2", "curious1", "inquiring1",
    "serenity1", "calming1",
}


class EmotionAnalyzer:
    """Analyze text and return emotion category, intensity, and level."""

    POSITIVE_WORDS = [
        '开心', '快乐', '高兴', '喜欢', '爱', '谢谢', '感谢', '好', '棒', '完美',
        'excited', 'happy', 'joy', 'love', 'thanks', 'good', 'great', 'awesome',
    ]
    NEGATIVE_WORDS = [
        '伤心', '难过', '悲伤', '生气', '失望', '抱歉', '对不起', '不好', '坏',
        'sad', 'angry', 'sorry', 'disappointed', 'bad', 'wrong', 'hate',
    ]
    QUESTION_WORDS = ['吗', '？', '?', '为什么', '怎么', '如何', 'what', 'why', 'how', 'when']
    ACTIVITY_WORDS = ['跳舞', '舞蹈', '运动', '活动', '动起来', 'dance', 'move', 'action', 'play']

    POSITIVE_EMOJI = ['😊', '😄', '😍', '👍', '🥰', '😎', '🎉', '❤️', '😂', '🤗']
    NEGATIVE_EMOJI = ['😢', '😭', '😡', '👎', '😔', '😞', '😤', '💔']
    QUESTION_EMOJI = ['🤔', '❓', '⁉️', '💭', '🧐', '🔍']
    ACTIVITY_EMOJI = ['💃', '🕺', '🎵', '🎶', '⚽', '🏀', '🎮']

    def analyze(self, text: str) -> Tuple[str, str, float]:
        text_lower = text.lower()
        emotion_level = min(len(text) / 200, 1.0)

        pos_count = sum(1 for word in self.POSITIVE_WORDS if word in text_lower)
        neg_count = sum(1 for word in self.NEGATIVE_WORDS if word in text_lower)
        ques_count = sum(1 for word in self.QUESTION_WORDS if word in text_lower)
        act_count = sum(1 for word in self.ACTIVITY_WORDS if word in text_lower)

        pos_count += sum(1 for emoji in self.POSITIVE_EMOJI if emoji in text)
        neg_count += sum(1 for emoji in self.NEGATIVE_EMOJI if emoji in text)
        ques_count += sum(1 for emoji in self.QUESTION_EMOJI if emoji in text)
        act_count += sum(1 for emoji in self.ACTIVITY_EMOJI if emoji in text)

        scores = {
            'positive': pos_count,
            'negative': neg_count,
            'question': ques_count,
            'activity': act_count,
        }
        emotion_type = max(scores, key=scores.get)
        total_score = sum(scores.values())

        # If no emotional cues at all, force neutral regardless of tie-breaking
        if total_score == 0:
            return 'neutral', 'low', emotion_level

        if total_score >= 3:
            intensity = 'high'
        elif total_score >= 1:
            intensity = 'medium'
        else:
            intensity = 'low'

        if emotion_type == 'positive':
            emotion_level = min(emotion_level * 1.2, 1.0)
        elif emotion_type == 'negative':
            emotion_level *= 0.8

        return emotion_type, intensity, emotion_level


class EmotionMovePlayer:
    """Select and play recorded moves from Pollen emotion/dance libraries."""

    def __init__(self, reachy, gentle_mode: bool = False, debug: bool = False):
        from reachy_mini.motion.recorded_move import RecordedMoves

        self.reachy = reachy
        self.gentle_mode = gentle_mode
        self.debug = debug
        self._emotions_lib = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
        self._dances_lib = RecordedMoves("pollen-robotics/reachy-mini-dances-library")
        self._categorize()

    def _categorize(self):
        self.emotion_to_moves: Dict[str, List[Tuple[str, str]]] = {
            'positive': [], 'negative': [], 'question': [],
            'activity': [], 'neutral': [],
        }

        for move_name in self._emotions_lib.list_moves():
            category = _EMOTION_CATEGORY_MAP.get(move_name)
            if category is None:
                category = self._keyword_category(move_name, self._emotions_lib)
            self.emotion_to_moves[category].append(('emotions', move_name))

        for move_name in self._dances_lib.list_moves():
            category = self._keyword_category(move_name, self._dances_lib)
            self.emotion_to_moves[category].append(('dances', move_name))

    def _keyword_category(self, move_name: str, lib) -> str:
        try:
            move = lib.get(move_name)
            desc = (move.description or "").lower()
        except Exception:
            desc = move_name.lower()

        scores = {
            'positive': sum(1 for w in [
                'happy', 'joy', 'love', 'excited', 'great', 'awesome',
                'good', 'thanks', 'celebrate', 'dance', 'cheer', 'proud',
            ] if w in desc),
            'negative': sum(1 for w in [
                'sad', 'angry', 'sorry', 'disappoint', 'bad', 'wrong',
                'hate', 'fear', 'bored', 'frustrat', 'rage', 'tired',
            ] if w in desc),
            'question': sum(1 for w in [
                'what', 'why', 'how', 'when', 'curious', 'wonder',
                'think', 'question', 'unsure', 'confused',
            ] if w in desc),
            'activity': sum(1 for w in [
                'dance', 'move', 'action', 'play', 'energy', 'wiggle',
            ] if w in desc),
        }
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else 'neutral'

    def _get_move(self, lib_tag: str, move_name: str):
        lib = self._emotions_lib if lib_tag == 'emotions' else self._dances_lib
        return lib.get(move_name)

    def _filter_gentle(self, moves: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        if not self.gentle_mode:
            return moves
        gentle = [(lib, name) for lib, name in moves if name in _GENTLE_EMOTIONS]
        return gentle if gentle else moves

    def play(self, emotion_type: str, intensity: str = 'medium'):
        available = self.emotion_to_moves.get(emotion_type, []) or self.emotion_to_moves['neutral']
        available = self._filter_gentle(available)
        if not available:
            return

        if intensity == 'high' and len(available) > 1:
            lib_tag, move_name = available[-1]
        elif intensity == 'low' and len(available) > 1:
            lib_tag, move_name = available[0]
        else:
            lib_tag, move_name = random.choice(available)

        duration_map = {'high': 0.8, 'medium': 1.0, 'low': 1.2}
        if self.gentle_mode:
            duration_map = {'high': 1.0, 'medium': 1.3, 'low': 1.5}
        duration = duration_map.get(intensity, 1.0)

        if self.debug:
            print(f"🎬 Playing {lib_tag}/{move_name} ({emotion_type}, {intensity})")
        try:
            move = self._get_move(lib_tag, move_name)
            self.reachy.play_move(move, initial_goto_duration=duration)
        except Exception as e:
            if self.debug:
                print(f"⚠️ Move error ({lib_tag}/{move_name}): {e}")


class SpeakingActor:
    """Play continuous robot actions while TTS is speaking."""

    def __init__(self, move_player: EmotionMovePlayer, debug: bool = False):
        self.move_player = move_player
        self.debug = debug
        self._is_speaking = False
        self._thread: Optional[threading.Thread] = None

    def start(self, emotion_type: str, intensity: str = 'medium'):
        self._is_speaking = True
        self._thread = threading.Thread(
            target=self._loop,
            args=(emotion_type, intensity),
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._is_speaking = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self, emotion_type: str, intensity: str):
        pause_map = {'high': 1.0, 'medium': 1.5, 'low': 2.0}
        pause = pause_map.get(intensity, 1.5)
        if self.move_player.gentle_mode:
            pause *= 1.5

        while self._is_speaking:
            try:
                self.move_player.play(emotion_type, intensity)
            except Exception as e:
                if self.debug:
                    print(f"⚠️ Speaking action error: {e}")
            time.sleep(pause)


class ChatEmotionController:
    """High-level facade: analyze emotion + play moves + speak with actions."""

    def __init__(self, reachy, gentle_mode: bool = False, debug: bool = False):
        self.analyzer = EmotionAnalyzer()
        self.move_player = EmotionMovePlayer(reachy, gentle_mode=gentle_mode, debug=debug)
        self.speaking_actor = SpeakingActor(self.move_player, debug=debug)
        self.debug = debug

    def analyze(self, text: str) -> Tuple[str, str, float]:
        return self.analyzer.analyze(text)

    def react(self, text: str, emotion: str, intensity: str):
        """Play an immediate reaction move for the given emotion."""
        self.move_player.play(emotion, intensity)

    def speak_with_actions(self, speak_fn, text: str, emotion: str, intensity: str):
        """Call speak_fn (blocking TTS) while robot performs actions.

        Args:
            speak_fn: Callable[[str], None] that blocks until speech finishes.
        """
        if not text.strip():
            return
        if self.debug:
            print(f"🗣️ Speaking with {emotion} emotion")

        self.speaking_actor.start(emotion, intensity)
        try:
            speak_fn(text)
        finally:
            self.speaking_actor.stop()
