"""Cloud providers against a mocked HTTP transport (no keys, no network), plus VAD on a recorded utterance."""
import asyncio
import base64
import io
import json
import wave

import httpx
import httpx2
import numpy as np
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from app.audio.vad import VAD
from app.providers.cloud import ClaudeLLM, ClaudeScene, OpenAISTT, OpenAITTS

WAV = "tests/data/utterance.wav"  # macOS `say` "Hey Lumi, where did I leave my mug?", 16 kHz mono, padded with noise


def mock(handler, cls):
    lib = httpx2 if cls is AsyncAnthropic else httpx  # anthropic>=1.x runs on the httpx2 fork, openai on httpx
    http = lib.AsyncClient(transport=lib.MockTransport(handler))
    return cls(api_key="test", http_client=http, max_retries=0)


def claude_reply(tool_input=None, name="respond"):
    content = [{"type": "tool_use", "id": "tu_1", "name": name, "input": tool_input}] if tool_input is not None \
        else [{"type": "text", "text": "no tool"}]
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "m", "content": content,
            "stop_reason": "tool_use", "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}}


def recorded_utterance(noise_rms: float = 150.0) -> tuple[bytes, float]:
    """The recording embedded in 1 s + 2.5 s of pink noise. Returns (pcm, time the speech ends)."""
    with wave.open(WAV) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(float)
    speech = a[8000:39000]
    rng = np.random.default_rng(0)

    def pink(n):
        spec = np.fft.rfft(rng.normal(0, 1, n))
        f = np.arange(len(spec))
        f[0] = 1
        x = np.fft.irfft(spec / np.sqrt(f), n)
        return x / x.std() * noise_rms

    pcm = np.concatenate([pink(16000), speech, pink(40000)]).clip(-32768, 32767).astype(np.int16).tobytes()
    return pcm, 1.0 + len(speech) / 16000


def test_vad_closes_the_utterance_in_steady_noise():
    pcm, speech_end = recorded_utterance()
    v, events = VAD(), []
    for i in range(0, len(pcm), 640):  # the client sends 20 ms chunks
        events += [(e, u, i / 32000) for e, u in v.feed(pcm[i:i + 640])]
    assert [e for e, _, _ in events] == ["start", "end"]  # webrtcvad alone never ends here (see DECISIONS)
    _, utt, t_end = events[1]
    assert speech_end < t_end < speech_end + 1.0 and len(utt) / 32000 > 1.8


def test_stt_uploads_the_vad_utterance_as_16k_wav():
    pcm, _ = recorded_utterance()
    v, utt = VAD(), None
    for i in range(0, len(pcm), 640):
        for e, u in v.feed(pcm[i:i + 640]):
            utt = u if e == "end" else utt
    seen = {}

    def handler(req: httpx.Request):
        seen["path"], body = req.url.path, req.content
        riff = body.index(b"RIFF")
        with wave.open(io.BytesIO(body[riff:riff + 44 + len(utt)])) as w:
            seen["wav"] = (w.getframerate(), w.getnchannels(), w.getnframes())
        return httpx.Response(200, json={"text": " Hey Lumi, where did I leave my mug? "})

    text = asyncio.run(OpenAISTT(mock(handler, AsyncOpenAI)).transcribe(utt))
    assert text == "Hey Lumi, where did I leave my mug?"
    assert seen["path"].endswith("/audio/transcriptions") and seen["wav"] == (16000, 1, len(utt) // 2)


def test_tts_yields_even_length_pcm_chunks():
    audio = bytes(range(256)) * 39 + b"\x01"  # 9985 bytes: odd total length

    async def collect():
        tts = OpenAITTS(mock(lambda req: httpx.Response(200, content=audio), AsyncOpenAI))
        return [c async for c in tts.stream("hello")]

    chunks = asyncio.run(collect())
    assert all(len(c) % 2 == 0 for c in chunks) and b"".join(chunks) == audio[:-1]


def test_llm_forces_the_tool_and_grounds_on_scene_and_memory():
    seen = {}
    reply = {"say": "Your mug is by the keyboard.", "emotion": "happy", "gesture": "nod", "music": "none"}

    def handler(req):
        seen.update(json.loads(req.content))
        return httpx2.Response(200, json=claude_reply(reply))

    llm = ClaudeLLM(mock(handler, AsyncAnthropic))
    out = asyncio.run(llm.respond(user_text="where is my mug?", history=[], memories=["mug: left of the keyboard"],
                                  scene=[{"name": "notebook", "location": "front right"}], mood="curious"))
    assert out == reply
    assert seen["tool_choice"] == {"type": "tool", "name": "respond"}
    assert "notebook (front right)" in seen["system"] and "mug: left of the keyboard" in seen["system"]
    assert seen["messages"][-1] == {"role": "user", "content": "where is my mug?"}


def test_llm_falls_back_when_no_tool_call():
    llm = ClaudeLLM(mock(lambda req: httpx2.Response(200, json=claude_reply(None)), AsyncAnthropic))
    out = asyncio.run(llm.respond(user_text="hi", history=[], memories=[], scene=[], mood="neutral"))
    assert set(out) == {"say", "emotion", "gesture", "music"} and out["emotion"] == "confused"


def test_scene_sends_the_frame_and_parses_objects():
    jpeg, seen = b"\xff\xd8fake-jpeg\xff\xd9", {}
    objects = [{"name": "mug", "location": "left of the keyboard", "details": "white"}]

    def handler(req):
        seen.update(json.loads(req.content))
        return httpx2.Response(200, json=claude_reply({"objects": objects}, name="report_scene"))

    out = asyncio.run(ClaudeScene(mock(handler, AsyncAnthropic)).describe(jpeg))
    image = seen["messages"][0]["content"][0]
    assert out == objects and base64.b64decode(image["source"]["data"]) == jpeg
    assert seen["tool_choice"] == {"type": "tool", "name": "report_scene"}


def test_cloud_providers_build_with_default_clients(monkeypatch):
    from app.config import settings
    from app.providers import build_providers

    monkeypatch.setattr(settings, "providers", "cloud")
    monkeypatch.setattr(settings, "anthropic_api_key", "test")
    monkeypatch.setattr(settings, "openai_api_key", "test")
    p = build_providers()  # constructs the real SDK clients; no network until a call
    assert type(p.llm).__name__ == "ClaudeLLM" and p.llm.c.timeout == 10.0 and p.stt.c.max_retries == 1
