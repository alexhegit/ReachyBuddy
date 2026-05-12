"""Chat mode - voice conversation with LLM via Ollama (placeholder).

Planned features:
- ASR -> Ollama LLM -> TTS pipeline
- Conversation history management
- Interrupt handling
- Voice activity detection for natural turn-taking
"""

from core.base_app import BaseModeApp, ModeConfig


class ChatModeApp(BaseModeApp):
    """Chat mode placeholder - voice conversation with LLM.

    This mode will provide:
    - Continuous voice chat with Ollama backend
    - Support for any Ollama model (qwen, llama, etc.)
    - Conversation history and context
    - Voice interrupt capability
    """

    def get_mode_name(self) -> str:
        return "chat"

    def get_requirements(self) -> list:
        return [
            "numpy",
            "opencv-python",
            "sounddevice",
            "soundfile",
            "faster-whisper",
            "webrtcvad-wheels",
            "piper-tts",
            "requests",  # For Ollama HTTP API
        ]

    def setup(self) -> None:
        """Placeholder - not yet implemented."""
        raise NotImplementedError(
            "\n"
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║  CHAT MODE - COMING SOON                                     ║\n"
            "╠══════════════════════════════════════════════════════════════╣\n"
            "║  Voice conversation with LLM:                                ║\n"
            "║    • ASR -> Ollama LLM -> TTS pipeline                       ║\n"
            "║    • Conversation history & context                          ║\n"
            "║    • Voice interrupt support                                 ║\n"
            "╠══════════════════════════════════════════════════════════════╣\n"
            "║  Usage:                                                      ║\n"
            "║    python main.py --chat --ollama-model qwen3:0.6b           ║\n"
            "╚══════════════════════════════════════════════════════════════╝\n"
        )

    def run_frame(self, frame) -> bool:
        return False

    def cleanup(self) -> None:
        pass
