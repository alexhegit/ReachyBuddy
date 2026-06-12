"""Configuration for cheese mode."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from core.base_app import ModeConfig


@dataclass
class CheeseConfig(ModeConfig):
    """Configuration for cheese photo mode."""

    # Save settings
    save_dir: Path = field(default_factory=lambda: Path.home() / "Pictures" / "ReachyMiniPhoto")

    # Wake word
    wake_word: str = "reachy"
    command_timeout_s: float = 12.0

    # Face tracking
    min_detection_confidence: float = 0.50
    smooth_factor: float = 0.20

    # Alignment thresholds
    deadzone_x: int = 25
    deadzone_y: int = 20
    stable_needed: int = 10

    # FaceAligner control parameters
    lock_hold_x: int = 35
    lock_hold_y: int = 28
    release_x: int = 45
    release_y: int = 35
    reacquire_x: int = 130
    reacquire_y: int = 110
    cmd_max_step_x: int = 95
    cmd_max_step_y: int = 70
    min_cmd_delta_px: int = 24
    max_body_yaw: float = 0.8
    ema_alpha: float = 0.30
    body_step: float = 0.12
    body_duration: float = 0.42
    head_reset_duration: float = 0.28
    body_cooldown: float = 0.6
    settle_duration: float = 0.35
    move_interval_soft: float = 0.28
    move_interval_reacquire: float = 0.22
    move_interval_normal: float = 0.16
    head_gain_x: float = 0.55
    head_gain_y: float = 0.50
    head_duration_soft: float = 0.34
    head_duration_normal: float = 0.24
    big_error_delay_reacquire: float = 0.35
    big_error_delay_normal: float = 0.5

    # Camera profile for hardware parameter tuning (reachy mode only)
    camera_profile: Optional[str] = None

    # Host/port of Reachy daemon (for remote robot control)
    reachy_host: str = "reachy-mini.local"
    reachy_port: int = 8000

    # Tracking enabled (head/body movements). Use CLI flag --track on/off (default: on)
    track_enabled: bool = True

    def __post_init__(self):
        """Ensure save_dir is a Path object."""
        if isinstance(self.save_dir, str):
            self.save_dir = Path(self.save_dir).expanduser()
