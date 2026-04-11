"""Guard mode - multi-modal security monitoring (placeholder).

Planned features:
- Multi-modal analysis using vision-language models (Gemma4, MiniCPM-o)
- HTTP API integration for model inference
- Event-based recording and alerting
- GPU acceleration handled by external model service
"""

from core.base_app import BaseModeApp, ModeConfig


class GuardModeApp(BaseModeApp):
    """Guard mode placeholder - multi-modal security monitoring.
    
    This mode will integrate with external multi-modal models via HTTP API:
    - Gemma4 (E2B/E4B variants)
    - MiniCPM-o 2.6 / 4.5
    
    The model service handles GPU inference separately.
    """
    
    def get_mode_name(self) -> str:
        return "guard"
    
    def get_requirements(self) -> list:
        return [
            "numpy",
            "opencv-python",
            "requests",  # For HTTP API calls to model service
            "pillow",
        ]
    
    def setup(self) -> None:
        """Placeholder - not yet implemented."""
        raise NotImplementedError(
            "\n"
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║  GUARD MODE - COMING SOON                                    ║\n"
            "╠══════════════════════════════════════════════════════════════╣\n"
            "║  Multi-modal security monitoring with:                       ║\n"
            "║    • Vision-Language models (Gemma4, MiniCPM-o)              ║\n"
            "║    • HTTP API integration                                    ║\n"
            "║    • Event recording & alerts                                ║\n"
            "╠══════════════════════════════════════════════════════════════╣\n"
            "║  Usage:                                                      ║\n"
            "║    python main.py --guard --guard-endpoint http://...        ║\n"
            "╚══════════════════════════════════════════════════════════════╝\n"
        )
    
    def run_frame(self, frame) -> bool:
        return False
    
    def cleanup(self) -> None:
        pass
