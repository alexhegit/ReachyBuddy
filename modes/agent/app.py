"""Agent mode - voice-controlled AI agent with tools (placeholder).

Planned features:
- ASR -> LLM with tool calling -> TTS
- OpenClaw integration for desktop control
- Extensible tool system
- Multi-step task execution
"""

from core.base_app import BaseModeApp, ModeConfig


class AgentModeApp(BaseModeApp):
    """Agent mode placeholder - voice-controlled AI agent.
    
    This mode will provide:
    - Tool-using LLM agent (via OpenClaw or similar)
    - Desktop control capabilities
    - Multi-step task planning and execution
    - Extensible tool registry
    """
    
    def get_mode_name(self) -> str:
        return "agent"
    
    def get_requirements(self) -> list:
        return [
            "numpy",
            "opencv-python",
            "sounddevice",
            "soundfile",
            "faster-whisper",
            "webrtcvad-wheels",
            "piper-tts",
            "requests",
            # "openclaw",  # Future dependency
        ]
    
    def setup(self) -> None:
        """Placeholder - not yet implemented."""
        raise NotImplementedError(
            "\n"
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║  AGENT MODE - COMING SOON                                    ║\n"
            "╠══════════════════════════════════════════════════════════════╣\n"
            "║  Voice-controlled AI agent:                                  ║\n"
            "║    • LLM with tool calling                                   ║\n"
            "║    • Desktop control via OpenClaw                            ║\n"
            "║    • Multi-step task execution                               ║\n"
            "╠══════════════════════════════════════════════════════════════╣\n"
            "║  Usage:                                                      ║\n"
            "║    python main.py --agent --tools config/tools.yaml          ║\n"
            "╚══════════════════════════════════════════════════════════════╝\n"
        )
    
    def run_frame(self, frame) -> bool:
        return False
    
    def cleanup(self) -> None:
        pass
