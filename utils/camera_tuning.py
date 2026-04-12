#!/usr/bin/env python3
"""Camera tuning utility for Reachy Mini and webcam.

This tool helps adjust camera hardware parameters (V4L2 controls) and
manage profiles for different lighting conditions.

Usage:
    python utils/camera_tuning.py --list
    python utils/camera_tuning.py --save default
    python utils/camera_tuning.py --set brightness=10,contrast=15
    python utils/camera_tuning.py --load indoor
    python utils/camera_tuning.py --reset
    python utils/camera_tuning.py --interactive
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Default camera device
DEFAULT_DEVICE = "/dev/video0"

# Profile storage directory
PROFILE_DIR = Path.home() / ".config" / "reachy_mini"

# Parameters that can be adjusted
# Note: white_balance_temperature is read-only when auto WB is enabled
TUNABLE_PARAMS = [
    "brightness",      # -64 to 64, default 0
    "contrast",        # 0 to 95, default 1
    "saturation",      # 0 to 100, default 48
    "hue",             # -2000 to 2000, default 0
    "gamma",           # 80 to 160, default 100
    "gain",            # 0 to 255, default 32
    "sharpness",       # 0 to 7, default 2
    "backlight_compensation",  # 0 to 10, default 2
]

# Default values from camera specs
# Note: white_balance_temperature is omitted as it's read-only when auto WB is enabled
DEFAULT_VALUES = {
    "brightness": 0,
    "contrast": 1,
    "saturation": 48,
    "hue": 0,
    "gamma": 100,
    "gain": 32,
    "sharpness": 2,
    "backlight_compensation": 2,
}


@dataclass
class CameraProfile:
    """Camera parameter profile."""
    name: str
    description: str
    params: Dict[str, int]
    device: str = DEFAULT_DEVICE
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "params": self.params,
            "device": self.device,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CameraProfile":
        return cls(
            name=data["name"],
            description=data["description"],
            params=data["params"],
            device=data.get("device", DEFAULT_DEVICE),
        )


class CameraTuner:
    """Camera tuning utility using v4l2-ctl."""
    
    def __init__(self, device: str = DEFAULT_DEVICE):
        self.device = device
        self.profile_dir = PROFILE_DIR
        self.profile_dir.mkdir(parents=True, exist_ok=True)
    
    def _run_v4l2_ctl(self, args: List[str]) -> Tuple[bool, str]:
        """Run v4l2-ctl command and return success status and output."""
        cmd = ["v4l2-ctl", "-d", self.device] + args
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr
        except FileNotFoundError:
            return False, "v4l2-ctl not found. Install with: sudo apt install v4l-utils"
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, f"Error: {e}"
    
    def get_current_params(self) -> Dict[str, int]:
        """Get current camera parameters."""
        params = {}
        # v4l2-ctl --get-ctrl takes comma-separated list
        ctrl_list = ",".join(TUNABLE_PARAMS)
        success, output = self._run_v4l2_ctl(["--get-ctrl", ctrl_list])
        if not success:
            print(f"Warning: Failed to get parameters: {output}")
            return params
        
        for line in output.strip().split("\n"):
            if ":" in line:
                key, value = line.strip().split(":", 1)
                key = key.strip()
                try:
                    params[key] = int(value.strip())
                except ValueError:
                    pass
        return params
    
    def set_params(self, params: Dict[str, int]) -> bool:
        """Set camera parameters."""
        ctrl_str = ",".join([f"{k}={v}" for k, v in params.items()])
        success, output = self._run_v4l2_ctl(["--set-ctrl", ctrl_str])
        if not success:
            print(f"Error setting parameters: {output}")
            return False
        return True
    
    def list_params(self) -> None:
        """List current camera parameters with default values."""
        current = self.get_current_params()
        
        print(f"\n📷 Camera: {self.device}")
        print("-" * 60)
        print(f"{'Parameter':<25} {'Current':<10} {'Default':<10} {'Status'}")
        print("-" * 60)
        
        for param in TUNABLE_PARAMS:
            current_val = current.get(param, "N/A")
            default_val = DEFAULT_VALUES.get(param, "N/A")
            
            if current_val != default_val:
                status = "🔧 MODIFIED"
            else:
                status = "✓ default"
            
            print(f"{param:<25} {current_val:<10} {default_val:<10} {status}")
        
        print("-" * 60)
    
    def save_profile(self, name: str, description: str = "") -> bool:
        """Save current camera settings as a profile."""
        params = self.get_current_params()
        profile = CameraProfile(
            name=name,
            description=description or f"Profile saved on {os.popen('date').read().strip()}",
            params=params,
            device=self.device,
        )
        
        profile_path = self.profile_dir / f"{name}.json"
        try:
            with open(profile_path, "w") as f:
                json.dump(profile.to_dict(), f, indent=2)
            print(f"✅ Profile saved: {profile_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to save profile: {e}")
            return False
    
    def load_profile(self, name: str) -> bool:
        """Load camera settings from a profile."""
        profile_path = self.profile_dir / f"{name}.json"
        
        if not profile_path.exists():
            print(f"❌ Profile not found: {profile_path}")
            print(f"Available profiles: {self.list_profile_names()}")
            return False
        
        try:
            with open(profile_path, "r") as f:
                data = json.load(f)
            profile = CameraProfile.from_dict(data)
            
            print(f"📂 Loading profile: {profile.name}")
            print(f"   Description: {profile.description}")
            
            if self.set_params(profile.params):
                print(f"✅ Profile '{name}' loaded successfully")
                self.list_params()
                return True
            return False
        except Exception as e:
            print(f"❌ Failed to load profile: {e}")
            return False
    
    def list_profiles(self) -> None:
        """List all saved profiles."""
        profiles = list(self.profile_dir.glob("*.json"))
        
        if not profiles:
            print("No saved profiles found.")
            return
        
        print("\n📂 Saved Profiles:")
        print("-" * 60)
        for profile_path in sorted(profiles):
            try:
                with open(profile_path, "r") as f:
                    data = json.load(f)
                profile = CameraProfile.from_dict(data)
                print(f"  • {profile.name:<15} - {profile.description}")
            except Exception:
                print(f"  • {profile_path.stem:<15} - (error reading profile)")
        print("-" * 60)
    
    def list_profile_names(self) -> List[str]:
        """Get list of profile names."""
        return [p.stem for p in self.profile_dir.glob("*.json")]
    
    def reset_to_defaults(self) -> bool:
        """Reset all parameters to camera defaults."""
        print("🔄 Resetting camera to default parameters...")
        if self.set_params(DEFAULT_VALUES):
            print("✅ Camera reset to defaults")
            self.list_params()
            return True
        return False
    
    def parse_param_string(self, param_str: str) -> Dict[str, int]:
        """Parse parameter string like 'brightness=10,contrast=15'."""
        params = {}
        for pair in param_str.split(","):
            if "=" not in pair:
                print(f"Warning: Ignoring invalid parameter: {pair}")
                continue
            key, value = pair.split("=", 1)
            key = key.strip()
            try:
                params[key] = int(value.strip())
            except ValueError:
                print(f"Warning: Invalid value for {key}: {value}")
        return params
    
    def interactive_tune(self) -> None:
        """Interactive parameter tuning."""
        print("\n🎛️  Interactive Camera Tuning")
        print("Commands: 's' = save, 'l' = load, 'r' = reset, 'q' = quit, 'h' = help")
        print("-" * 60)
        
        while True:
            self.list_params()
            
            try:
                cmd = input("\nEnter command or 'param=value' (h for help): ").strip()
                
                if not cmd:
                    continue
                
                if cmd == "q":
                    print("Exiting...")
                    break
                
                elif cmd == "h":
                    print("\nCommands:")
                    print("  h              - Show this help")
                    print("  q              - Quit")
                    print("  s [name]       - Save current settings as profile")
                    print("  l [name]       - Load profile")
                    print("  r              - Reset to defaults")
                    print("  list           - List saved profiles")
                    print("  param=value    - Set parameter (e.g., brightness=10)")
                    print("\nParameters: " + ", ".join(TUNABLE_PARAMS))
                
                elif cmd == "r":
                    self.reset_to_defaults()
                
                elif cmd == "list":
                    self.list_profiles()
                
                elif cmd.startswith("s "):
                    name = cmd[2:].strip() or "custom"
                    desc = input("Enter description (optional): ").strip()
                    self.save_profile(name, desc)
                
                elif cmd.startswith("l "):
                    name = cmd[2:].strip()
                    if name:
                        self.load_profile(name)
                    else:
                        self.list_profiles()
                
                elif "=" in cmd:
                    params = self.parse_param_string(cmd)
                    if params:
                        print(f"Setting: {params}")
                        self.set_params(params)
                
                else:
                    print(f"Unknown command: {cmd}. Type 'h' for help.")
            
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except EOFError:
                break


def main():
    parser = argparse.ArgumentParser(
        description="Camera tuning utility for Reachy Mini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List current parameters
  python utils/camera_tuning.py --list
  
  # Save current settings as "default" profile
  python utils/camera_tuning.py --save default
  
  # Adjust specific parameters
  python utils/camera_tuning.py --set brightness=5,contrast=10,saturation=55
  
  # Load a saved profile
  python utils/camera_tuning.py --load indoor
  
  # Reset to camera defaults
  python utils/camera_tuning.py --reset
  
  # Interactive tuning mode
  python utils/camera_tuning.py --interactive
        """
    )
    
    parser.add_argument(
        "--device", "-d",
        default=DEFAULT_DEVICE,
        help=f"Camera device (default: {DEFAULT_DEVICE})"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List current camera parameters"
    )
    
    parser.add_argument(
        "--save", "-s",
        metavar="NAME",
        help="Save current settings as a profile"
    )
    
    parser.add_argument(
        "--load",
        metavar="NAME",
        help="Load settings from a profile"
    )
    
    parser.add_argument(
        "--set",
        metavar="PARAMS",
        help="Set parameters (format: param1=value1,param2=value2)"
    )
    
    parser.add_argument(
        "--reset", "-r",
        action="store_true",
        help="Reset all parameters to defaults"
    )
    
    parser.add_argument(
        "--profiles",
        action="store_true",
        help="List saved profiles"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive tuning mode"
    )
    
    parser.add_argument(
        "--desc",
        default="",
        help="Description for saved profile"
    )
    
    args = parser.parse_args()
    
    tuner = CameraTuner(device=args.device)
    
    # Default action: list parameters
    if not any([args.list, args.save, args.load, args.set, args.reset, 
                args.profiles, args.interactive]):
        tuner.list_params()
        return
    
    if args.list:
        tuner.list_params()
    
    if args.save:
        tuner.save_profile(args.save, args.desc)
    
    if args.load:
        tuner.load_profile(args.load)
    
    if args.set:
        params = tuner.parse_param_string(args.set)
        if params:
            print(f"Setting parameters: {params}")
            if tuner.set_params(params):
                print("✅ Parameters set successfully")
                tuner.list_params()
    
    if args.reset:
        tuner.reset_to_defaults()
    
    if args.profiles:
        tuner.list_profiles()
    
    if args.interactive:
        tuner.interactive_tune()


if __name__ == "__main__":
    main()
