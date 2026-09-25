"""Single multiplexed WebSocket. Binary frames = [1 byte kind][payload]. Text frames = JSON control."""
from enum import IntEnum


class Frame(IntEnum):
    AUDIO_IN = 0x01   # client -> server: PCM16 LE mono 16 kHz (20 ms chunks)
    VIDEO_IN = 0x02   # client -> server: JPEG
    AUDIO_OUT = 0x03  # server -> client: PCM16 LE mono 24 kHz


def pack(kind: Frame, payload: bytes) -> bytes:
    return bytes([kind]) + payload

# JSON control messages
# client -> server: {"type":"playback_done"} | {"type":"text","text":"..."} (typed input, bypasses STT)
# server -> client: state | transcript | scene | body | sfx | music | tts_start | tts_end | stop_audio | goal
# goal: {step: start|observe|aim|verify|done|failed, ms since goal start, ...step details}
