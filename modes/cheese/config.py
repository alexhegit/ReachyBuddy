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
    
    # Software image adjustment (applied after frame capture)
    # Note: These improve display but don't fix hardware-level image quality
    # Default values are tuned for Reachy camera (vs Webcam with hardware ISP):
    # - Reachy: brightness=15, contrast=1.3, saturation=1.4
    # - Webcam: brightness=0, contrast=1.0, saturation=1.0
    brightness: float = 0.0   # -50 to +50, 0 is no change
    contrast: float = 1.0     # 0.5 to 2.0, 1.0 is no change  
    saturation: float = 1.0   # 0.0 to 2.0, 1.0 is no change
    
    # Hardware camera parameters (Reachy Mini only, if supported by SDK)
    # These would need to be set through reachy_mini SDK camera controls
    auto_exposure: bool = True
    exposure: int = -1  # -1 for auto, or specific value
    gain: int = -1      # -1 for auto, or specific value
    
    def __post_init__(self):
        """Ensure save_dir is a Path object."""
        if isinstance(self.save_dir, str):
            self.save_dir = Path(self.save_dir).expanduser()
