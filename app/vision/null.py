"""No-camera attention for headless tests. Never produces events."""
from .types import Reading


class NullAttention:
    attending = None  # unknown: never force a disengage

    def update(self, jpeg: bytes):
        return [], Reading()
