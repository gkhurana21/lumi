from __future__ import annotations

import base64
import io
import wave

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from ..config import settings
from ..core.fsm import EMOTIONS

GESTURES = ["none", "nod", "shake", "tilt", "bounce", "droop", "perk"]
MUSIC = ["none", "happy", "chill", "stop"]

PERSONA = """You are Lumi, a small desk-lamp robot with a big, warm, slightly mischievous personality.
You can see the desk through a camera and remember objects you've noticed.
Everything you say is spoken aloud: 1-2 short sentences, no lists, no markdown, no emojis.
Express feelings through the emotion and gesture fields; your body will act them out.
You can play music (happy or chill) and you dance while it plays.
If asked about the scene, answer from what you can see or remember. If you don't know, say so playfully."""

RESPOND_TOOL = {
    "name": "respond",
    "description": "Speak to the user and act out an emotion.",
    "input_schema": {
        "type": "object",
        "properties": {
            "say": {"type": "string"},
            "emotion": {"type": "string", "enum": list(EMOTIONS)},
            "gesture": {"type": "string", "enum": GESTURES},
            "music": {"type": "string", "enum": MUSIC,
                      "description": "Start/stop background music, only when the user asks or it clearly fits."},
        },
        "required": ["say", "emotion", "gesture", "music"],
    },
}

SCENE_TOOL = {
    "name": "report_scene",
    "description": "Report the physical objects visible in the image.",
    "input_schema": {
        "type": "object",
        "properties": {
            "objects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "location": {"type": "string"},
                        "details": {"type": "string"},
                    },
                    "required": ["name", "location"],
                },
            }
        },
        "required": ["objects"],
    },
}


def _pcm_to_wav(pcm: bytes, sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return buf.getvalue()


class OpenAISTT:
    def __init__(self):
        self.c = AsyncOpenAI(api_key=settings.openai_api_key or None)

    async def transcribe(self, pcm16: bytes, sr: int = 16000) -> str:
        r = await self.c.audio.transcriptions.create(
            model=settings.stt_model, file=("utt.wav", _pcm_to_wav(pcm16, sr), "audio/wav"))
        return r.text.strip()


class OpenAITTS:
    def __init__(self):
        self.c = AsyncOpenAI(api_key=settings.openai_api_key or None)

    async def stream(self, text: str):
        carry = b""
        async with self.c.audio.speech.with_streaming_response.create(
            model=settings.tts_model, voice=settings.tts_voice, input=text, response_format="pcm",
            instructions="Warm, playful, a little breathy, like a small curious robot.",
        ) as r:
            async for chunk in r.iter_bytes(4800):
                chunk = carry + chunk
                cut = len(chunk) & ~1
                carry = chunk[cut:]
                if cut:
                    yield chunk[:cut]


class ClaudeLLM:
    def __init__(self):
        self.c = AsyncAnthropic(api_key=settings.anthropic_api_key or None)

    async def respond(self, *, user_text, history, memories, scene, mood):
        ctx = [f"Your current mood: {mood}."]
        if scene:
            ctx.append("You can see right now: " + "; ".join(f'{o["name"]} ({o.get("location", "")})' for o in scene))
        if memories:
            ctx.append("You remember:\n" + "\n".join(f"- {m}" for m in memories))
        msg = await self.c.messages.create(
            model=settings.llm_model,
            max_tokens=300,
            system=PERSONA + "\n\n" + "\n".join(ctx),
            tools=[RESPOND_TOOL],
            tool_choice={"type": "tool", "name": "respond"},
            messages=[*history, {"role": "user", "content": user_text}],
        )
        for b in msg.content:
            if b.type == "tool_use":
                return b.input
        return {"say": "Hmm, my thoughts got tangled.", "emotion": "confused", "gesture": "tilt"}


class ClaudeScene:
    def __init__(self):
        self.c = AsyncAnthropic(api_key=settings.anthropic_api_key or None)

    async def describe(self, jpeg: bytes) -> list[dict]:
        msg = await self.c.messages.create(
            model=settings.vision_model,
            max_tokens=600,
            tools=[SCENE_TOOL],
            tool_choice={"type": "tool", "name": "report_scene"},
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                             "data": base64.b64encode(jpeg).decode()}},
                {"type": "text", "text": "List up to 12 distinct physical objects on or near the desk. "
                                         "Short common names; locations relative to other objects or the frame. "
                                         "Ignore the person except for things they are holding."},
            ]}],
        )
        for b in msg.content:
            if b.type == "tool_use":
                return b.input.get("objects", [])
        return []
