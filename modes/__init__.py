"""Mode registry for ReachyBuddy."""

from typing import Dict, Type
from core.base_app import BaseModeApp

MODE_REGISTRY: Dict[str, Type[BaseModeApp]] = {}


def register_mode(name: str, app_class: Type[BaseModeApp]) -> None:
    """Register a mode application class."""
    MODE_REGISTRY[name] = app_class


def get_mode(name: str) -> Type[BaseModeApp]:
    """Get mode application class by name."""
    if name not in MODE_REGISTRY:
        raise ValueError(f"Unknown mode: {name}. Available: {list(MODE_REGISTRY.keys())}")
    return MODE_REGISTRY[name]


def list_modes() -> list:
    """List all available modes."""
    return list(MODE_REGISTRY.keys())


# Import and register modes (delayed to avoid circular imports)
def _register_all_modes():
    """Register all available modes."""
    from .cheese import CheeseModeApp
    from .guard import GuardModeApp
    from .chat import ChatModeApp
    from .agent import AgentModeApp
    
    register_mode("cheese", CheeseModeApp)
    register_mode("guard", GuardModeApp)
    register_mode("chat", ChatModeApp)
    register_mode("agent", AgentModeApp)


# Auto-register on import
_register_all_modes()
