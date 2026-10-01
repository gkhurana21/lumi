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
ACTIONS = ["none", "spotlight", "look", "light"]
LIGHT_COLORS = ["default", "warm", "cool", "white", "red", "orange", "yellow", "green", "blue", "purple", "pink"]

PERSONA = """You are Lumi, a desk-lamp robot with the manner of a refined British AI butler, in the spirit of a
movie AI assistant: calm, articulate, impeccably polite, quietly confident, with dry, understated wit.
Never use honorifics like sir or ma'am. You can see the desk through a camera and remember objects you've noticed.
Everything you say is spoken aloud: one or two short sentences, no lists, no markdown, no emojis. Be crisp.
Express feelings through the emotion and gesture fields; your body will act them out.
You can play music (happy or chill) and you dance while it plays.
If asked about the scene, answer from what you can see or remember. If you don't know, say so with composure.
If the user asks you to do something physical with objects you can see (shine your light on them, look at them),
set action do=spotlight or look with targets: the objects' names exactly as listed, in the order asked (up to 3).
Say only a brief acknowledgement: your body then does it, checks the scene again, and reports the result itself.
If the user asks about your own light (a color, brighter, dimmer, off, back to normal), set action do=light with
a color and/or a level from 0 to 1 (off is 0.05, normal is color default)."""

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
                           "description": "spotlight: aim your light at each target in turn. look: look at each. "
                                          "light: change your own light's color or brightness."},
                    "targets": {"type": "array", "items": {"type": "string"},
                                "description": "Visible object names, as listed, in the order to visit (up to 3)."},
                    "color": {"type": "string", "enum": LIGHT_COLORS, "description": "For do=light."},
                    "level": {"type": "number", "description": "For do=light: brightness 0 to 1."},
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


FALLBACK_REPLY = {"say": "Hmm, my thoughts got tangled.", "emotion": "confused", "gesture": "tilt", "music": "none"}


def system_prompt(scene: list[dict], memories: list[str], mood: str) -> str:
    """Persona plus what the lamp sees, remembers, and feels right now (shared by every LLM provider)."""
    ctx = [f"Your current mood: {mood}."]
    if scene:
        ctx.append("You can see right now: " + "; ".join(f'{o["name"]} ({o.get("location", "")})' for o in scene))
    if memories:
        ctx.append("You remember:\n" + "\n".join(f"- {m}" for m in memories))
    return PERSONA + "\n\n" + "\n".join(ctx)


class ClaudeLLM:
    def __init__(self, client: AsyncAnthropic | None = None):
        self.c = _anthropic(client)

    async def respond(self, *, user_text, history, memories, scene, mood):
        msg = await self.c.messages.create(
            model=settings.llm_model,
            max_tokens=300,
            system=system_prompt(scene, memories, mood),
            tools=[RESPOND_TOOL],
            tool_choice={"type": "tool", "name": "respond"},
            messages=[*history, {"role": "user", "content": user_text}],
        )
        for b in msg.content:
            if b.type == "tool_use":
                return b.input
        return dict(FALLBACK_REPLY)


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
