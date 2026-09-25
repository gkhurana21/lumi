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
ACTIONS = ["none", "spotlight", "look"]

PERSONA = """You are Lumi, a small desk-lamp robot with a big, warm, slightly mischievous personality.
You can see the desk through a camera and remember objects you've noticed.
Everything you say is spoken aloud: 1-2 short sentences, no lists, no markdown, no emojis.
Express feelings through the emotion and gesture fields; your body will act them out.
You can play music (happy or chill) and you dance while it plays.
If asked about the scene, answer from what you can see or remember. If you don't know, say so playfully.
If the user asks you to do something physical with an object you can see (shine your light on it, look at it),
set action with the object's name exactly as you see it listed, and say only a short acknowledgement: your body
then does it, checks the scene again, and reports the result itself."""

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
            "action": {
                "type": "object",
                "description": "A physical goal on a visible object. Omit or use do=none otherwise.",
                "properties": {
                    "do": {"type": "string", "enum": ACTIONS,
                           "description": "spotlight: aim your light at it and brighten. look: turn to look at it."},
                    "target": {"type": "string", "description": "Name of a visible object, as listed."},
                },
                "required": ["do"],
            },
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
                        "x": {"type": "number", "description": "Center, fraction of image width (0 = left)."},
                        "y": {"type": "number", "description": "Center, fraction of image height (0 = top)."},
                    },
                    "required": ["name", "location", "x", "y"],
                },
            }
        },
        "required": ["objects"],
    },
}


# A hung request must not stall the character: THINKING already aborts after 10 s, and a stuck scene call would
# block every later scan. One retry covers transient 429/5xx. A float: the Anthropic SDK rejects httpx.Timeout.
TIMEOUT = 10.0


def _anthropic(client: AsyncAnthropic | None) -> AsyncAnthropic:
    return client or AsyncAnthropic(api_key=settings.anthropic_api_key or None, timeout=TIMEOUT, max_retries=1)


def _openai(client: AsyncOpenAI | None) -> AsyncOpenAI:
    return client or AsyncOpenAI(api_key=settings.openai_api_key or None, timeout=TIMEOUT, max_retries=1)


def _pcm_to_wav(pcm: bytes, sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return buf.getvalue()


class OpenAISTT:
    def __init__(self, client: AsyncOpenAI | None = None):
        self.c = _openai(client)

    async def transcribe(self, pcm16: bytes, sr: int = 16000) -> str:
        r = await self.c.audio.transcriptions.create(
            model=settings.stt_model, file=("utt.wav", _pcm_to_wav(pcm16, sr), "audio/wav"))
        return r.text.strip()


class OpenAITTS:
    def __init__(self, client: AsyncOpenAI | None = None):
        self.c = _openai(client)

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
    def __init__(self, client: AsyncAnthropic | None = None):
        self.c = _anthropic(client)

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
        return {"say": "Hmm, my thoughts got tangled.", "emotion": "confused", "gesture": "tilt", "music": "none"}


class ClaudeScene:
    def __init__(self, client: AsyncAnthropic | None = None):
        self.c = _anthropic(client)

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
                                         "Short common names; locations relative to other objects or the frame; "
                                         "x and y of each object's center as fractions of the image. "
                                         "Ignore the person except for things they are holding."},
            ]}],
        )
        for b in msg.content:
            if b.type == "tool_use":
                return [_centered(o) for o in b.input.get("objects", [])]
        return []


def _centered(o: dict) -> dict:
    """Image fractions (0..1, top-left) -> the camera-frame convention used locally ([-1, 1], like face readings)."""
    out = dict(o)
    for k in ("x", "y"):
        if isinstance(o.get(k), int | float):
            out[k] = round(max(-1.0, min(1.0, 2 * float(o[k]) - 1)), 3)
        else:
            out.pop(k, None)
    return out
