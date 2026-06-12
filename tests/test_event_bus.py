"""Test EventBus pub/sub system."""
import pytest
from core.event_bus import EventBus, get_event_bus


class TestEventBus:
    def test_subscribe_and_emit(self):
        bus = EventBus()
        received = []

        def handler(data):
            received.append(data)

        bus.subscribe("test_event", handler)
        bus.emit("test_event", {"msg": "hello"})
        assert received == [{"msg": "hello"}]

    def test_multiple_subscribers(self):
        bus = EventBus()
        results = []

        def h1(d):
            results.append("h1")

        def h2(d):
            results.append("h2")

        bus.subscribe("e", h1)
        bus.subscribe("e", h2)
        bus.emit("e", None)
        assert results == ["h1", "h2"]

    def test_unsubscribe(self):
        bus = EventBus()

        def handler(data):
            pass

        bus.subscribe("e", handler)
        assert "e" in bus._handlers
        bus.unsubscribe("e", handler)
        assert handler not in bus._handlers["e"]

    def test_emit_unknown_event_no_error(self):
        bus = EventBus()
        bus.emit("nonexistent", None)

    def test_clear_removes_all(self):
        bus = EventBus()
        bus.subscribe("a", lambda d: None)
        bus.subscribe("b", lambda d: None)
        bus.clear()
        assert len(bus._handlers) == 0

    def test_singleton(self):
        assert get_event_bus() is get_event_bus()
