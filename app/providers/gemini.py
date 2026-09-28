"""Google Gemini providers: one API key covers speech-to-text, dialogue, scene vision, and voice.

Same contracts as the other providers (see __init__.py). Tuned for the free tier: fast "lite" models by default,
every model name configurable in .env, and object positions taken from Gemini's native box_2d detections.
"""
from __future__ import annotations

import copy
import io
import logging
import wave

import numpy as np
from google import genai
from google.genai import types

from ..config import settings
from .cloud import FALLBACK_REPLY, RESPOND_TOOL, _pcm_to_wav, system_prompt

log = logging.getLogger("gemini")

STT_PROMPT = ("Transcribe the speech in this audio exactly as spoken, in its language. Reply with the words only. "
              "If there is no speech, reply with nothing.")
SCENE_PROMPT = ("Detect up to 12 distinct physical objects on or near the desk. For each give a short common name, "
                "where it is relative to other objects or the frame, and box_2d as [ymin, xmin, ymax, xmax] "
                "normalized to 0-1000. Ignore the person except for things they are holding.")
SCENE_SCHEMA = {
    "type": "object",
    "properties": {"objects": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "location": {"type": "string"},
            "details": {"type": "string"},
            "box_2d": {"type": "array", "items": {"type": "integer"},
                       "description": "[ymin, xmin, ymax, xmax], 0-1000"},
        },
        "required": ["name", "location", "box_2d"],
    }}},
    "required": ["objects"],
}
NO_AFC = types.AutomaticFunctionCallingConfig(disable=True)  # we only declare schemas; nothing runs automatically
# One-call turns: the utterance audio goes straight to the dialogue model, which also returns what it heard.
RESPOND_HEARD = copy.deepcopy(RESPOND_TOOL["input_schema"])
RESPOND_HEARD["properties"]["heard"] = {"type": "string",
                                        "description": "The user's words in the audio, verbatim. Empty if no speech."}
RESPOND_HEARD["required"] = ["heard", *RESPOND_HEARD["required"]]
HEARD_NOTE = ("The user's latest turn is audio. Put their exact words in heard. If there is no speech in it, "
              "set heard and say to empty strings.")


def make_client() -> genai.Client:
    # 10 s timeout like the other clouds (THINKING aborts at 10 s anyway); one retry covers a transient 429/5xx.
    opts = types.HttpOptions(timeout=10_000, retry_options=types.HttpRetryOptions(attempts=2))
    return genai.Client(api_key=settings.gemini_api_key or None, http_options=opts)


def _forced(name: str, schema: dict, description: str, **extra) -> types.GenerateContentConfig:
    """Force exactly one call of the named function, so the reply is always parseable structured data."""
    fn = types.FunctionDeclaration(name=name, description=description, parameters_json_schema=schema)
    return types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=[fn])],
        tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(
            mode="ANY", allowed_function_names=[name])),
        automatic_function_calling=NO_AFC,
        **extra,
    )


def _args(response, name: str) -> dict | None:
    for call in response.function_calls or []:
        if call.name == name:
            return dict(call.args or {})
    return None


def _from_box(o: dict) -> dict:
    """box_2d (0-1000, [ymin, xmin, ymax, xmax]) -> center x/y in the local [-1, 1] camera-frame convention."""
    out = {k: v for k, v in o.items() if k != "box_2d"}
    box = o.get("box_2d")
    if isinstance(box, list) and len(box) == 4 and all(isinstance(v, int | float) for v in box):
        ymin, xmin, ymax, xmax = (max(0.0, min(1000.0, float(v))) for v in box)
        out["x"] = round((xmin + xmax) / 1000 - 1, 3)
        out["y"] = round((ymin + ymax) / 1000 - 1, 3)
    return out


class GeminiSTT:
    def __init__(self, client: genai.Client):
        self.c = client

    async def transcribe(self, pcm16: bytes, sr: int = 16000) -> str:
        r = await self.c.aio.models.generate_content(
            model=settings.gemini_fast_model,
            contents=[types.Part.from_bytes(data=_pcm_to_wav(pcm16, sr), mime_type="audio/wav"), STT_PROMPT],
            config=types.GenerateContentConfig(automatic_function_calling=NO_AFC))
        return (r.text or "").strip()


class GeminiLLM:
    def __init__(self, client: genai.Client):
        self.c = client

    async def respond(self, *, user_text, history, memories, scene, mood):
        contents = self._history(history)
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_text)]))
        config = _forced("respond", RESPOND_TOOL["input_schema"], RESPOND_TOOL["description"],
                         system_instruction=system_prompt(scene, memories, mood), max_output_tokens=300)
        r = await self.c.aio.models.generate_content(model=settings.gemini_llm_model, contents=contents, config=config)
        return _args(r, "respond") or dict(FALLBACK_REPLY)

    async def respond_to_audio(self, *, audio: bytes, sr: int, history, memories, scene, mood) -> dict:
        """Transcribe and reply in one call (saves a round trip, about 1 s). The reply carries `heard`."""
        contents = self._history(history)
        contents.append(types.Content(role="user", parts=[
            types.Part.from_bytes(data=_pcm_to_wav(audio, sr), mime_type="audio/wav")]))
        config = _forced("respond", RESPOND_HEARD, RESPOND_TOOL["description"], max_output_tokens=400,
                         system_instruction=system_prompt(scene, memories, mood) + "\n" + HEARD_NOTE)
        r = await self.c.aio.models.generate_content(model=settings.gemini_llm_model, contents=contents, config=config)
        return _args(r, "respond") or {**FALLBACK_REPLY, "heard": ""}

    @staticmethod
    def _history(history) -> list[types.Content]:
        return [types.Content(role="user" if m["role"] == "user" else "model",
                              parts=[types.Part.from_text(text=m["content"])]) for m in history]


class GeminiScene:
    def __init__(self, client: genai.Client):
        self.c = client

    async def describe(self, jpeg: bytes) -> list[dict]:
        r = await self.c.aio.models.generate_content(
            model=settings.gemini_fast_model,
            contents=[types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), SCENE_PROMPT],
            config=_forced("report_scene", SCENE_SCHEMA, "Report the physical objects visible in the image."))
        objects = (_args(r, "report_scene") or {}).get("objects", [])
        return [_from_box(o) for o in objects if isinstance(o, dict) and o.get("name")]


class GeminiTTS:
    """Streamed: first audio after about 0.8 s instead of 3.9 s for the whole line (measured). No style prompt: this
    model reads style instructions aloud or slows to a crawl (a 3 s line became 10 s), so the voice carries the
    character."""

    def __init__(self, client: genai.Client):
        self.c = client

    async def stream(self, text: str):
        voice = types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=settings.gemini_voice))
        config = types.GenerateContentConfig(response_modalities=["AUDIO"], automatic_function_calling=NO_AFC,
                                             speech_config=types.SpeechConfig(voice_config=voice))
        got, carry = False, b""
        async for chunk in await self.c.aio.models.generate_content_stream(
                model=settings.gemini_tts_model, contents=text, config=config):
            for cand in chunk.candidates or []:
                for part in (cand.content.parts if cand.content else None) or []:
                    if part.inline_data and part.inline_data.data:
                        raw = carry + part.inline_data.data  # keep samples whole across chunk boundaries
                        cut = len(raw) & ~1
                        carry = raw[cut:]
                        if cut:
                            got = True
                            yield _to_24k(raw[:cut], part.inline_data.mime_type or "")
        if not got:
            raise RuntimeError("Gemini TTS returned no audio")


def _to_24k(data: bytes, mime: str) -> bytes:
    """The client plays 24 kHz PCM16 mono. Streamed chunks are raw 'audio/l16; rate=24000; channels=1'; a
    non-streamed reply is 'audio/wav' (with a header that would play as a click)."""
    rate = 24000
    if "wav" in mime.lower():
        with wave.open(io.BytesIO(data)) as w:
            rate, data = w.getframerate(), w.readframes(w.getnframes())
    else:
        rate = next((int(p.split("=")[1]) for p in mime.replace(" ", "").split(";") if p.startswith("rate=")), rate)
    pcm = data[:len(data) & ~1]
    if rate == 24000:
        return pcm
    x = np.frombuffer(pcm, "<i2").astype(np.float32)
    t = np.arange(0, len(x) * 24000 / rate) * rate / 24000
    return np.interp(t, np.arange(len(x)), x).astype("<i2").tobytes()
