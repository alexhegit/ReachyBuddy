"""Configuration for guard mode."""

from dataclasses import dataclass, field
from typing import Optional

from core.base_app import ModeConfig


@dataclass
class GuardConfig(ModeConfig):
    """Configuration for guard security monitoring mode."""

    # Ollama API
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "gemma4:12b"

    # Analysis interval (seconds between VLM calls)
    analysis_interval: float = 8.0

    # Head scanning
    scan_enabled: bool = True
    scan_range: float = 0.6       # ±radians for pan sweep
    scan_speed: float = 0.015     # radians per scan step
    scan_interval: float = 0.3    # seconds between scan steps

    # Alert cooldown (seconds) — same alert won't repeat within this window
    alert_cooldown: float = 30.0

    # Prompt sent to the VLM
    prompt: str = (
        "You are a security camera. Describe what you see briefly in 1-2 sentences. "
        "If the scene is normal and empty, say exactly 'OK'. "
        "If you see people, describe their position and action. "
        "If you see anything unusual, describe it."
    )

    # Save alert screenshots
    save_dir: Optional[str] = None

    # Connection info for daemon
    reachy_host: str = "127.0.0.1"
    reachy_port: int = 8000

    def __post_init__(self):
        if self.save_dir is None:
            from pathlib import Path
            self.save_dir = str(Path.home() / "Pictures" / "ReachyGuard")
