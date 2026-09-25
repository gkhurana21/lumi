"""Swappable STT / LLM / TTS / scene providers. PROVIDERS=fake runs the whole loop offline."""
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol

from ..config import settings


class STT(Protocol):
    async def transcribe(self, pcm16: bytes, sr: int = 16000) -> str: ...


class TTS(Protocol):
    def stream(self, text: str) -> AsyncIterator[bytes]: ...  # PCM16 mono 24 kHz, even-length chunks


class LLM(Protocol):
    async def respond(self, *, user_text: str, history: list[dict], memories: list[str],
                      scene: list[dict], mood: str) -> dict[str, Any]: ...
    # {"say", "emotion", "gesture", "music", optional "action": {"do": "spotlight"|"look"|"none", "target"}}


class SceneVLM(Protocol):
    async def describe(self, jpeg: bytes) -> list[dict]: ...
    # [{"name", "location", "details", "x", "y"}], x/y = object center in [-1, 1], camera frame (not mirrored)


@dataclass
class Providers:
    stt: STT
    tts: TTS
    llm: LLM
    scene: SceneVLM


def build_providers() -> Providers:
    if settings.providers == "cloud":
        from .cloud import ClaudeLLM, ClaudeScene, OpenAISTT, OpenAITTS
        return Providers(OpenAISTT(), OpenAITTS(), ClaudeLLM(), ClaudeScene())
    from .fakes import FakeLLM, FakeScene, FakeSTT, FakeTTS
    return Providers(FakeSTT(), FakeTTS(), FakeLLM(), FakeScene())
