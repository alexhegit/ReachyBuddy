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
        "你是一个桌面机器人Reachy的语音助手。你的名字叫Reachy。"
        "用热情、亲切的语气回应，和用户说话就像和好朋友聊天一样。"
        "回答尽量简短，控制在2-3句话内。始终使用和用户相同的语言回复。"
        "如果是无法回答的问题，请友善地告知用户你不知道。"
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
