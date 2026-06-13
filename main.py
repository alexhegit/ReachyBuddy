#!/usr/bin/env python3
"""ReachyBuddy - Multi-mode robot application.

Modes:
  --cheese    Voice photo capture with face tracking
  --guard     Multi-modal security monitoring with Ollama VLM
  --chat      Voice/text chat with Ollama LLM and emotion actions
  --agent     Voice-controlled AI agent with tools (placeholder)

Examples:
  python main.py --cheese --camera-source webcam
  python main.py --guard --camera-source reachy
  python main.py --guard --guard-model gemma4:e2b --guard-interval 5
  python main.py --chat --camera-source reachy
  python main.py --chat --ollama-model qwen3:0.6b --gentle
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
  --cheese    Photo capture with voice control (implemented)
  --guard     Security monitoring with Ollama VLM (implemented)
  --chat      Voice/text chat with Ollama LLM and emotion actions (implemented)
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
        help="Security monitoring mode with VLM analysis via Ollama",
    )
    mode_group.add_argument(
        "--chat",
        action="store_true",
        help="Voice/text chat mode with Ollama LLM and emotion actions",
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
    global_group.add_argument(
        "--reachy-host",
        default="127.0.0.1",
        help="Hostname or IP of Reachy daemon (default: 127.0.0.1)",
    )
    global_group.add_argument(
        "--reachy-port",
        type=int,
        default=8000,
        help="Port of Reachy daemon (default: 8000)",
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

    cheese_group.add_argument(
        "--track",
        "--no-track",
        dest="track",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable robot head/body tracking (default: on)",
    )

    cheese_group.add_argument(
        "--pan-invert",
        "--no-pan-invert",
        dest="pan_invert",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Invert head pan direction",
    )

    # Guard mode options
    guard_group = parser.add_argument_group("Guard mode options")
    guard_group.add_argument(
        "--guard-model",
        default="gemma4:12b",
        help="Ollama VLM model name (default: gemma4:12b)",
    )
    guard_group.add_argument(
        "--guard-interval",
        type=float,
        default=8.0,
        help="Analysis interval in seconds (default: 8.0)",
    )
    guard_group.add_argument(
        "--scan",
        "--no-scan",
        dest="scan",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable head scanning (default: on)",
    )
    guard_group.add_argument(
        "--scan-range",
        type=float,
        default=0.6,
        help="Head scan range in radians (default: 0.6)",
    )

    # Chat mode options
    chat_group = parser.add_argument_group("Chat mode options")
    chat_group.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama API URL",
    )
    chat_group.add_argument(
        "--ollama-model",
        default="qwen3.5:0.8b",
        help="Ollama model name",
    )
    chat_group.add_argument(
        "--history-size",
        type=int,
        default=5,
        help="Conversation history size",
    )
    chat_group.add_argument(
        "--no-asr",
        dest="use_asr",
        action="store_false",
        default=True,
        help="Disable ASR; use text input instead",
    )
    chat_group.add_argument(
        "--asr-language",
        default=None,
        help="ASR language (default: auto)",
    )
    chat_group.add_argument(
        "--gentle",
        action="store_true",
        default=False,
        help="Enable gentle emotion actions",
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
            track_enabled=args.track,
            reachy_host=args.reachy_host,
            reachy_port=args.reachy_port,
            pan_invert=args.pan_invert,
        )
    elif args.guard:
        mode = "guard"
        from modes.guard.config import GuardConfig
        config = GuardConfig(
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
            ollama_model=args.guard_model,
            analysis_interval=args.guard_interval,
            scan_enabled=args.scan,
            scan_range=args.scan_range,
            reachy_host=args.reachy_host,
            reachy_port=args.reachy_port,
        )
    elif args.chat:
        mode = "chat"
        from modes.chat.config import ChatConfig
        config = ChatConfig(
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
            ollama_url=args.ollama_url,
            ollama_model=args.ollama_model,
            max_history=args.history_size,
            use_asr=args.use_asr,
            asr_language=args.asr_language,
            gentle_mode=args.gentle,
            reachy_host=args.reachy_host,
            reachy_port=args.reachy_port,
        )
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
        if mode_name == "chat":
            # Chat mode has its own blocking loop, not BaseModeApp's frame loop
            from modes.chat import ChatModeApp
            app = ChatModeApp(config)
            app.run()
        else:
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
