"""Event bus for inter-mode communication (future use)."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional


class EventBus:
    """Simple event bus for decoupled communication.
    
    Currently minimal implementation - can be extended for:
    - Cross-mode event subscription
    - Async event handling
    - Event persistence/queuing
    """
    
    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = {}
    
    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
    
    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """Unsubscribe from an event type."""
        if event_type in self._handlers:
            self._handlers[event_type] = [
                h for h in self._handlers[event_type] if h != handler
            ]
    
    def emit(self, event_type: str, data: Optional[dict] = None) -> None:
        """Emit an event to all subscribers."""
        if event_type not in self._handlers:
            return
        for handler in self._handlers[event_type]:
            try:
                handler(data)
            except Exception:
                pass  # Ignore handler errors
    
    def clear(self) -> None:
        """Clear all subscriptions."""
        self._handlers.clear()


# Global event bus instance (for cross-mode events if needed)
_global_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get global event bus instance."""
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus
