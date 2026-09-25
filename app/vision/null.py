"""No-camera attention for headless tests. Never produces events."""
from .types import Reading


class NullAttention:
    def update(self, jpeg: bytes):
        return [], Reading()
