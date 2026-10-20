"""
SENTINEL — In-process Event Bus
Lightweight pub/sub replacing Kafka for the initial build.
The same interface will be backed by a real Kafka producer later.
Subscribers receive every intercepted action event in real time.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from typing import Any, Callable

# ── Event store (ring buffer — last 500 events) ───────────────────────────────
_events: deque[dict[str, Any]] = deque(maxlen=500)
_lock = threading.Lock()
_subscribers: list[Callable[[dict], None]] = []


def publish(event: dict[str, Any]) -> None:
    """Publish an event to all subscribers and the in-memory store."""
    event.setdefault("published_at", time.time())
    with _lock:
        _events.append(event)
        for cb in _subscribers:
            try:
                cb(event)
            except Exception:
                pass  # never let a subscriber crash the agent


def subscribe(callback: Callable[[dict], None]) -> None:
    """Register a callback to receive every published event."""
    with _lock:
        _subscribers.append(callback)


def get_events(last_n: int | None = None) -> list[dict]:
    with _lock:
        events = list(_events)
    return events[-last_n:] if last_n else events


def clear() -> None:
    with _lock:
        _events.clear()
        _subscribers.clear()
