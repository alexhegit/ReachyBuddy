"""Configuration for chat mode."""

from dataclasses import dataclass, field
from typing import Optional

from core.base_app import ModeConfig


@dataclass
class ChatConfig(ModeConfig):
    """Configuration for chat mode."""

    # Ollama API
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:0.8b"

    # ASR
    use_asr: bool = True
    asr_model: str = "base"
    asr_language: Optional[str] = None  # None = auto
    asr_device: str = "cpu"

    # Conversation
    system_prompt: str = (
        "You are a cute desktop robot assistant. Respond with enthusiasm and warmth. "
        "Always respond in the same language as the user's message. Keep answers concise."
    )
    max_history: int = 5
    temperature: float = 0.8
    max_tokens: int = 200

    # Emotion actions
    gentle_mode: bool = False

    # Thinking animation
    thinking_duration: float = 10.0

    # Connection info for daemon
    reachy_host: str = "127.0.0.1"
    reachy_port: int = 8000

    def __post_init__(self):
        pass
