"""Configuration for cheese mode."""

from dataclasses import dataclass, field
from pathlib import Path

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
    
    # Image adjustment
    brightness: float = 0.0  # -50 to +50, 0 is no change
    contrast: float = 1.0    # 0.5 to 2.0, 1.0 is no change
    
    def __post_init__(self):
        """Ensure save_dir is a Path object."""
        if isinstance(self.save_dir, str):
            self.save_dir = Path(self.save_dir).expanduser()
