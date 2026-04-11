"""Base application class for all modes."""

from __future__ import annotations

import atexit
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import cv2
import numpy as np

if TYPE_CHECKING:
    from .runtime import RobotRuntime


@dataclass
class ModeConfig:
    """Base configuration for all modes."""
    
    # Camera settings
    camera_source: str = "reachy"  # "reachy" or "webcam"
    camera_index: int = 0
    
    # Preview settings
    preview_width: int = 640
    preview_height: int = 480
    preview_fps: float = 20.0
    
    # ASR settings
    asr_model: str = "base"  # tiny, base, small, medium, large
    vad_silence: float = 0.7
    vad_aggressive: int = 1
    
    # TTS settings
    piper_model: str = "models/en-us-ryan-medium.onnx"
    piper_config: Optional[str] = None
    speaker_id: int = 0
    
    # GUI settings
    gui_backend: str = "auto"  # "auto", "dpg", "cv2", "none"
    
    # Debug
    debug: bool = False
    
    # Mode-specific settings (subclass override)
    mode_specific: dict = field(default_factory=dict)


class BaseModeApp(ABC):
    """Base class for all mode applications.
    
    Lifecycle:
        1. __init__(config)
        2. check_requirements() -> bool
        3. setup()
        4. run() -> loop run_frame()
        5. cleanup()
    """
    
    def __init__(self, config: ModeConfig):
        self.config = config
        self._running = False
        self._frame_count = 0
        
        # Components (initialized in setup)
        self.runtime: Optional[RobotRuntime] = None
        self.voice: Optional[object] = None
        self.gui: Optional[object] = None
        
        # Timing
        self._frame_interval = 1.0 / max(config.preview_fps, 1.0)
    
    @abstractmethod
    def get_mode_name(self) -> str:
        """Return mode name (e.g., 'cheese', 'guard')."""
        raise NotImplementedError
    
    @abstractmethod
    def get_requirements(self) -> List[str]:
        """Return list of required Python packages for this mode."""
        raise NotImplementedError
    
    def check_requirements(self) -> Tuple[bool, List[str]]:
        """Check if all requirements are installed.
        
        Returns:
            (ok, missing_packages)
        """
        import importlib
        missing = []
        for pkg in self.get_requirements():
            # Handle package name mapping (e.g., "opencv-python" -> "cv2")
            module_name = pkg.split("[")[0].replace("-", "_").lower()
            if module_name in ("opencv_python",):
                module_name = "cv2"
            elif module_name == "pillow":
                module_name = "PIL"
            try:
                importlib.import_module(module_name)
            except ImportError:
                missing.append(pkg)
        return len(missing) == 0, missing
    
    @abstractmethod
    def setup(self) -> None:
        """Initialize mode-specific components.
        
        Called once before run() starts.
        Should initialize: runtime, voice, gui, etc.
        """
        raise NotImplementedError
    
    @abstractmethod
    def run_frame(self, frame: np.ndarray) -> bool:
        """Process one frame.
        
        Args:
            frame: Camera frame (BGR format)
            
        Returns:
            True to continue, False to exit
        """
        raise NotImplementedError
    
    @abstractmethod
    def cleanup(self) -> None:
        """Mode-specific cleanup.
        
        Called when run() ends (normally or on exception).
        Release mode-specific resources here.
        """
        pass
    
    def _cleanup_without_exit(self) -> None:
        """Cleanup without calling os._exit (for exception propagation)."""
        print("\n🧹 Cleaning up...", flush=True)
        
        # Stop running
        self._running = False
        
        # Cleanup mode-specific resources
        try:
            self.cleanup()
        except Exception as e:
            if self.config.debug:
                print(f"⚠️ Mode cleanup error: {e}")
        
        # Cleanup GUI
        if self.gui:
            try:
                self.gui.close()
            except Exception:
                pass
        
        # Cleanup voice
        if self.voice:
            try:
                self.voice.close()
            except Exception:
                pass
        
        # Stop sounddevice
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
                pass
        
        print("👋 Goodbye!", flush=True)
    
    def _cleanup_base(self) -> None:
        """Base cleanup (called automatically)."""
        import sys
        
        self._cleanup_without_exit()
        sys.stdout.flush()
        
        # Prevent segfault on exit
        time.sleep(0.1)
        os._exit(0)
    
    def run(self) -> None:
        """Main run loop."""
        # Check requirements
        ok, missing = self.check_requirements()
        if not ok:
            print(f"❌ Missing dependencies for {self.get_mode_name()} mode:")
            for pkg in missing:
                print(f"   - {pkg}")
            print(f"\nInstall with: pip install -r requirements/{self.get_mode_name()}.txt")
            return
        
        # Register cleanup
        atexit.register(self._cleanup_base)
        
        exception_to_reraise = None
        
        try:
            # Setup
            print(f"🚀 Starting {self.get_mode_name().upper()} mode...")
            self.setup()
            self._running = True
            
            print(f"🤖 Running (Press 'q' in GUI or Ctrl+C to exit)\n")
            
            # Main loop
            while self._running:
                tick = time.time()
                
                # Get frame
                if self.runtime:
                    frame = self.runtime.get_frame()
                    if frame is None:
                        time.sleep(0.02)
                        continue
                else:
                    time.sleep(0.02)
                    continue
                
                self._frame_count += 1
                
                # Process frame
                try:
                    should_continue = self.run_frame(frame)
                    if not should_continue:
                        break
                except Exception as e:
                    if self.config.debug:
                        import traceback
                        traceback.print_exc()
                    else:
                        print(f"⚠️ Frame error: {e}")
                
                # Maintain FPS
                elapsed = time.time() - tick
                if elapsed < self._frame_interval:
                    time.sleep(self._frame_interval - elapsed)
                    
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        except NotImplementedError as e:
            # Store to re-raise after cleanup
            exception_to_reraise = e
        except Exception as e:
            print(f"\n❌ Error: {e}")
            if self.config.debug:
                import traceback
                traceback.print_exc()
        finally:
            atexit.unregister(self._cleanup_base)
            # Cleanup without os._exit if we need to re-raise
            if exception_to_reraise is not None:
                self._cleanup_without_exit()
                raise exception_to_reraise
            else:
                self._cleanup_base()
    
    def stop(self) -> None:
        """Request stop (can be called from signal handler or UI)."""
        self._running = False
