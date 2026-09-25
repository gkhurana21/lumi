"""Engagement state machine + continuous affect layer.

Engagement (discrete) = what the lamp is doing.
Affect (valence/arousal, continuous) = how it feels while doing it.
Behaviors read both, so the same state looks different when the lamp is excited vs sleepy.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class State(str, Enum):
    IDLE = "idle"                # nobody around, slumped, dim, slow breathing
    NOTICING = "noticing"        # face appeared, perks up, looks toward it
    ENGAGED = "engaged"          # user is looking at the lamp, tracks their face
    LISTENING = "listening"      # user is speaking, head tilt, holds still
    THINKING = "thinking"        # STT + LLM in flight, looks up/away, light pulses
    SPEAKING = "speaking"        # TTS playing, bobs with speech, faces user
    DISENGAGING = "disengaging"  # user looked away or left, lingers before sleeping


class Ev(str, Enum):
    FACE_SEEN = "face_seen"
    FACE_LOST = "face_lost"
    ATTN_ON = "attn_on"
    ATTN_OFF = "attn_off"
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    REPLY_READY = "reply_ready"
    TTS_DONE = "tts_done"
    ABORT = "abort"      # empty transcript, API error
    TIMEOUT = "timeout"  # fired by tick() using TIMEOUTS


S, E = State, Ev
TRANSITIONS: dict[tuple[State, Ev], State] = {
    (S.IDLE, E.FACE_SEEN): S.NOTICING,
    (S.IDLE, E.ATTN_ON): S.ENGAGED,
    (S.IDLE, E.SPEECH_START): S.LISTENING,
    (S.NOTICING, E.ATTN_ON): S.ENGAGED,
    (S.NOTICING, E.SPEECH_START): S.LISTENING,
    (S.NOTICING, E.FACE_LOST): S.IDLE,
    (S.NOTICING, E.TIMEOUT): S.IDLE,
    (S.ENGAGED, E.SPEECH_START): S.LISTENING,
    (S.ENGAGED, E.ATTN_OFF): S.DISENGAGING,
    (S.ENGAGED, E.FACE_LOST): S.DISENGAGING,
    (S.ENGAGED, E.REPLY_READY): S.SPEAKING,     # proactive speech (greeting)
    (S.LISTENING, E.SPEECH_END): S.THINKING,
    (S.LISTENING, E.TIMEOUT): S.THINKING,       # monologue cap, flush what we have
    (S.THINKING, E.REPLY_READY): S.SPEAKING,
    (S.THINKING, E.SPEECH_START): S.LISTENING,  # user kept talking, drop the reply
    (S.THINKING, E.ABORT): S.ENGAGED,
    (S.THINKING, E.TIMEOUT): S.ENGAGED,
    (S.SPEAKING, E.TTS_DONE): S.ENGAGED,
    (S.SPEAKING, E.SPEECH_START): S.LISTENING,  # barge-in
    (S.SPEAKING, E.ABORT): S.ENGAGED,
    (S.SPEAKING, E.TIMEOUT): S.ENGAGED,         # safety net if client never reports playback_done
    (S.DISENGAGING, E.ATTN_ON): S.ENGAGED,
    (S.DISENGAGING, E.SPEECH_START): S.LISTENING,
    (S.DISENGAGING, E.TIMEOUT): S.IDLE,
}

TIMEOUTS: dict[State, float] = {
    S.NOTICING: 3.0,
    S.LISTENING: 12.0,
    S.THINKING: 10.0,
    S.SPEAKING: 30.0,
    S.DISENGAGING: 5.0,
}

# (valence, arousal), both in [-1, 1]
EMOTIONS: dict[str, tuple[float, float]] = {
    "neutral": (0.0, 0.0),
    "happy": (0.7, 0.4),
    "excited": (0.8, 0.9),
    "curious": (0.3, 0.5),
    "surprised": (0.2, 0.9),
    "shy": (0.2, -0.3),
    "confused": (-0.2, 0.3),
    "sad": (-0.6, -0.4),
    "annoyed": (-0.5, 0.5),
    "sleepy": (0.0, -0.8),
}

# Mood each state drifts toward when no LLM emotion is being held
STATE_BASELINE: dict[State, tuple[float, float]] = {
    S.IDLE: (0.0, -0.6),
    S.NOTICING: (0.1, 0.3),
    S.ENGAGED: (0.2, 0.1),
    S.LISTENING: (0.2, 0.3),
    S.THINKING: (0.1, 0.2),
    S.SPEAKING: (0.3, 0.3),
    S.DISENGAGING: (-0.1, -0.2),
}

# Perception events nudge affect instantly so the lamp reacts before any LLM call
EVENT_IMPULSE: dict[Ev, tuple[float, float]] = {
    E.FACE_SEEN: (0.1, 0.35),
    E.ATTN_ON: (0.2, 0.2),
    E.FACE_LOST: (-0.15, -0.2),
    E.SPEECH_START: (0.0, 0.2),
    E.ABORT: (-0.2, 0.1),
}


def _clamp(x: float) -> float:
    return max(-1.0, min(1.0, x))


@dataclass
class Affect:
    valence: float = 0.0
    arousal: float = -0.6
    baseline: tuple[float, float] = (0.0, -0.6)
    rate: float = 2.0  # 1/s, exponential approach speed
    _emo: tuple[float, float] | None = None
    _hold_until: float = 0.0

    def set_emotion(self, name: str, hold: float = 4.0) -> None:
        self._emo = EMOTIONS.get(name, EMOTIONS["neutral"])
        self._hold_until = time.monotonic() + hold

    def impulse(self, dv: float, da: float) -> None:
        self.valence = _clamp(self.valence + dv)
        self.arousal = _clamp(self.arousal + da)

    def step(self, dt: float) -> None:
        held = self._emo is not None and time.monotonic() < self._hold_until
        tv, ta = self._emo if held else self.baseline
        k = 1.0 - math.exp(-self.rate * dt)
        self.valence += (tv - self.valence) * k
        self.arousal += (ta - self.arousal) * k

    def label(self) -> str:
        return min(EMOTIONS, key=lambda n: (EMOTIONS[n][0] - self.valence) ** 2 + (EMOTIONS[n][1] - self.arousal) ** 2)


OnTransition = Callable[[State, State, Ev], None]


class Machine:
    def __init__(self, on_transition: OnTransition | None = None):
        self.state = S.IDLE
        self.entered = time.monotonic()
        self.affect = Affect(baseline=STATE_BASELINE[S.IDLE])
        self._cb = on_transition

    def fire(self, ev: Ev) -> bool:
        if ev in EVENT_IMPULSE:
            self.affect.impulse(*EVENT_IMPULSE[ev])
        nxt = TRANSITIONS.get((self.state, ev))
        if nxt is None:
            return False
        prev, self.state, self.entered = self.state, nxt, time.monotonic()
        self.affect.baseline = STATE_BASELINE[nxt]
        if self._cb:
            self._cb(prev, nxt, ev)
        return True

    def tick(self, dt: float) -> None:
        self.affect.step(dt)
        limit = TIMEOUTS.get(self.state)
        if limit is not None and time.monotonic() - self.entered > limit:
            self.fire(E.TIMEOUT)
