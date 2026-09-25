from __future__ import annotations

import asyncio

import numpy as np


class FakeSTT:
    async def transcribe(self, pcm16: bytes, sr: int = 16000) -> str:
        await asyncio.sleep(0.2)
        return "what do you see on my desk?" if len(pcm16) > sr * 2 * 0.3 else ""


class FakeLLM:
    async def respond(self, *, user_text, history, memories, scene, mood):
        await asyncio.sleep(0.4)
        t = user_text.lower()
        if "music" in t or "song" in t:
            return {"say": "Ooh, dance break!", "emotion": "excited", "gesture": "bounce", "music": "happy"}
        if "stop" in t:
            return {"say": "Okay, quiet time.", "emotion": "neutral", "gesture": "nod", "music": "stop"}
        if memories:
            return {"say": f"I remember: {memories[0]}.", "emotion": "happy", "gesture": "nod", "music": "none"}
        seen = ", ".join(o["name"] for o in scene) or "nothing yet"
        return {"say": f"I can see {seen}.", "emotion": "curious", "gesture": "tilt", "music": "none"}


class FakeScene:
    async def describe(self, jpeg: bytes):
        await asyncio.sleep(0.5)
        return [{"name": "mug", "location": "left of the keyboard", "details": "white ceramic"},
                {"name": "notebook", "location": "front right", "details": "open"}]


class FakeTTS:
    """Warbly robot babble whose length tracks the text. Good enough to test turn-taking."""

    async def stream(self, text: str):
        sr = 24000
        dur = min(4.0, 0.06 * len(text))
        t = np.arange(int(sr * dur)) / sr
        freq = 520 + 140 * np.sin(2 * np.pi * 3 * t)
        gate = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 6 * t))
        wave = 0.2 * np.sin(2 * np.pi * np.cumsum(freq) / sr) * gate
        pcm = (wave * 32767).astype("<i2").tobytes()
        for i in range(0, len(pcm), 9600):  # 200 ms chunks
            yield pcm[i:i + 9600]
            await asyncio.sleep(0.05)
