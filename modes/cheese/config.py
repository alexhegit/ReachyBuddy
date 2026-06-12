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

    # Hierarchical tracking control
    #
    # Two layers:
    #   COARSE (body yaw) — large horizontal errors, slow 0.8s interval.
    #     Body rotates to bring face into the head's tracking range.
    #     Head does NOT reset after body move — it keeps tracking through
    #     the rotation so the face is never lost.
    #   FINE (head pan/tilt) — residual errors, fast 0.4s interval.
    #     Head handles fine horizontal centering + vertical alignment.
    #
    # Alignment (ARMED state) is determined by the fine deadzone only.
    #
    # Signal chain:
    #   FaceAligner: ema_dx * head_gain_x → target_x (pixel offset)
    #   Fallback: (target_x - center)/(w/2) * pan_gain → pan (radians)
    #   Effective gain = head_gain_x * pan_gain / (w/2)
    #
    # Fine deadzone (for ARMED transition)
    deadzone_x: int = 30
    deadzone_y: int = 25
    stable_needed: int = 10

    # Head tracking: proportional gain from pixel error.
    # With gain=1.5 and 60° HFOV, head nearly tracks face 1:1.
    head_gain_x: float = 1.5
    head_gain_y: float = 1.0

    # Head movement intervals
    head_interval: float = 0.25
    head_duration_normal: float = 0.30
    head_duration_soft: float = 0.40

    # Body-follows-head: when the head looks beyond this offset (px)
    # the body rotates to recenter the head.
    body_follow_threshold_px: int = 150
    body_max_step: float = 0.20   # rad per body movement
    body_duration: float = 0.60
    body_interval: float = 1.0
    max_body_yaw: float = 0.8

    ema_alpha: float = 0.25

    # Head deadzone (hysteresis for lock release)
    head_deadzone_x: int = 40
    head_deadzone_y: int = 30

    # Fallback look_at_image gains
    # Normalised pixel offset → radians: pan = nx * pan_gain, tilt = ny * tilt_gain
    pan_gain: float = 0.40
    tilt_gain: float = 0.35
    pan_invert: bool = False

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
