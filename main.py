#!/usr/bin/env python3
"""ReachyBuddy - Multi-mode robot application.

Modes:
  --cheese    Voice photo capture with face tracking
  --guard     Multi-modal security monitoring (placeholder)
  --chat      Voice conversation with LLM (placeholder)
  --agent     Voice-controlled AI agent (placeholder)

Examples:
  python main.py --cheese --camera-source webcam
  python main.py --guard --guard-endpoint http://localhost:8000
  python main.py --chat --ollama-model qwen3:0.6b
  python main.py --agent --tools config/tools.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with all mode options."""
    parser = argparse.ArgumentParser(
        prog="ReachyBuddy",
        description="Multi-mode robot application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available modes:
  --cheese    Voice photo capture with face tracking (implemented)
  --guard     Multi-modal security monitoring (placeholder)
  --chat      Voice conversation with LLM via Ollama (placeholder)
  --agent     Voice-controlled AI agent with tools (placeholder)

For mode-specific help:
  python main.py --cheese --help
        """,
    )

    # Mode selection group (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--cheese",
        action="store_true",
        help="Photo capture mode with voice control",
    )
    mode_group.add_argument(
        "--guard",
        action="store_true",
        help="Security monitoring mode (placeholder)",
    )
    mode_group.add_argument(
        "--chat",
        action="store_true",
        help="Voice chat mode with LLM (placeholder)",
    )
    mode_group.add_argument(
        "--agent",
        action="store_true",
        help="AI agent mode with tools (placeholder)",
    )

    # Global options (all modes)
    global_group = parser.add_argument_group("Global options")
    global_group.add_argument(
        "--camera-source",
        choices=["reachy", "webcam"],
        default="reachy",
        help="Camera source (default: reachy)",
    )
    global_group.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam index (default: 0)",
    )
    global_group.add_argument(
        "--camera-profile",
        type=str,
        default=None,
        help="Camera profile name to load (for reachy mode). Use camera_tuning_gui.py to create profiles",
    )
    global_group.add_argument(
        "--preview-width",
        type=int,
        default=640,
        help="Preview width (default: 640)",
    )
    global_group.add_argument(
        "--preview-height",
        type=int,
        default=480,
        help="Preview height (default: 480)",
    )
    global_group.add_argument(
        "--preview-fps",
        type=float,
        default=20.0,
        help="Preview FPS (default: 20.0)",
    )
    global_group.add_argument(
        "--gui-backend",
        choices=["cv2", "none"],
        default="cv2",
        help="GUI backend: cv2=window display, none=headless (default: cv2)",
    )
    global_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )

    # ASR options
    asr_group = parser.add_argument_group("ASR options")
    asr_group.add_argument(
        "--asr-model",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="ASR model size (default: base)",
    )
    asr_group.add_argument(
        "--vad-silence",
        type=float,
        default=0.7,
        help="VAD silence threshold seconds (default: 0.7)",
    )
    asr_group.add_argument(
        "--vad-aggressive",
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        help="VAD aggressiveness (default: 1)",
    )

    # TTS options
    tts_group = parser.add_argument_group("TTS options")
    tts_group.add_argument(
        "--piper-model",
        default="models/en-us-ryan-medium.onnx",
        help="Piper TTS model path",
    )
    tts_group.add_argument(
        "--piper-config",
        default=None,
        help="Piper TTS config path (optional)",
    )
    tts_group.add_argument(
        "--speaker",
        type=int,
        default=0,
        help="Speaker ID for multi-speaker models (default: 0)",
    )

    # Cheese mode options
    cheese_group = parser.add_argument_group("Cheese mode options")
    cheese_group.add_argument(
        "--save-dir",
        type=Path,
        default=Path.home() / "Pictures" / "ReachyMiniPhoto",
        help="Photo save directory",
    )
    cheese_group.add_argument(
        "--wake-word",
        default="reachy",
        help="Wake word for photo mode (default: reachy)",
    )
    cheese_group.add_argument(
        "--timeout",
        type=float,
        default=12.0,
        help="Command timeout in seconds (default: 12)",
    )

    # Guard mode options (placeholder)
    guard_group = parser.add_argument_group("Guard mode options (placeholder)")
    guard_group.add_argument(
        "--guard-endpoint",
        default="http://localhost:8000",
        help="Multi-modal model API endpoint",
    )
    guard_group.add_argument(
        "--guard-model",
        choices=["gemma4-e2b", "gemma4-e4b", "minicpm-o-4.5"],
        default="gemma4-e2b",
        help="Vision model to use",
    )

    # Chat mode options (placeholder)
    chat_group = parser.add_argument_group("Chat mode options (placeholder)")
    chat_group.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama API URL",
    )
    chat_group.add_argument(
        "--ollama-model",
        default="qwen3:0.6b",
        help="Ollama model name",
    )
    chat_group.add_argument(
        "--history-size",
        type=int,
        default=5,
        help="Conversation history size",
    )

    # Agent mode options (placeholder)
    agent_group = parser.add_argument_group("Agent mode options (placeholder)")
    agent_group.add_argument(
        "--tools",
        type=Path,
        help="Tools configuration file",
    )
    agent_group.add_argument(
        "--agent-model",
        default="qwen3:0.6b",
        help="LLM for agent",
    )

    return parser


def build_config(args) -> tuple:
    """Build mode-specific config from arguments.

    Returns:
        (mode_name, config_object)
    """
    from modes.cheese.config import CheeseConfig
    from core.base_app import ModeConfig

    # Determine mode
    if args.cheese:
        mode = "cheese"
        config = CheeseConfig(
            camera_source=args.camera_source,
            camera_index=args.camera_index,
            preview_width=args.preview_width,
            preview_height=args.preview_height,
            preview_fps=args.preview_fps,
            gui_backend=args.gui_backend,
            debug=args.debug,
            asr_model=args.asr_model,
            vad_silence=args.vad_silence,
            vad_aggressive=args.vad_aggressive,
            piper_model=args.piper_model,
            piper_config=args.piper_config,
            speaker_id=args.speaker,
            save_dir=args.save_dir,
            wake_word=args.wake_word,
            command_timeout_s=args.timeout,
            camera_profile=args.camera_profile,
        )
    elif args.guard:
        mode = "guard"
        config = ModeConfig(
            camera_source=args.camera_source,
            camera_index=args.camera_index,
            preview_width=args.preview_width,
            preview_height=args.preview_height,
            preview_fps=args.preview_fps,
            gui_backend=args.gui_backend,
            debug=args.debug,
            asr_model=args.asr_model,
            vad_silence=args.vad_silence,
            vad_aggressive=args.vad_aggressive,
            piper_model=args.piper_model,
            piper_config=args.piper_config,
            speaker_id=args.speaker,
        )
        # Store guard-specific in mode_specific dict
        config.mode_specific = {
            "endpoint": args.guard_endpoint,
            "model": args.guard_model,
        }
    elif args.chat:
        mode = "chat"
        config = ModeConfig(
            camera_source=args.camera_source,
            camera_index=args.camera_index,
            preview_width=args.preview_width,
            preview_height=args.preview_height,
            preview_fps=args.preview_fps,
            gui_backend=args.gui_backend,
            debug=args.debug,
            asr_model=args.asr_model,
            vad_silence=args.vad_silence,
            vad_aggressive=args.vad_aggressive,
            piper_model=args.piper_model,
            piper_config=args.piper_config,
            speaker_id=args.speaker,
        )
        config.mode_specific = {
            "ollama_url": args.ollama_url,
            "ollama_model": args.ollama_model,
            "history_size": args.history_size,
        }
    elif args.agent:
        mode = "agent"
        config = ModeConfig(
            camera_source=args.camera_source,
            camera_index=args.camera_index,
            preview_width=args.preview_width,
            preview_height=args.preview_height,
            preview_fps=args.preview_fps,
            gui_backend=args.gui_backend,
            debug=args.debug,
            asr_model=args.asr_model,
            vad_silence=args.vad_silence,
            vad_aggressive=args.vad_aggressive,
            piper_model=args.piper_model,
            piper_config=args.piper_config,
            speaker_id=args.speaker,
        )
        config.mode_specific = {
            "tools": args.tools,
            "agent_model": args.agent_model,
        }
    else:
        raise ValueError("No mode selected")

    return mode, config


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Build config
    try:
        mode_name, config = build_config(args)
    except Exception as e:
        print(f"❌ Config error: {e}")
        return 1

    # Import and run mode
    try:
        from modes import get_mode

        ModeClass = get_mode(mode_name)
        app = ModeClass(config)
        app.run()
        return 0

    except NotImplementedError as e:
        import sys
        print(str(e), flush=True)
        sys.stdout.flush()
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
