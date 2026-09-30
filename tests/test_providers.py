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


def _tones(seconds: float = 6.0) -> dict[str, np.ndarray]:
    t = np.arange(int(16000 * seconds)) / 16000
    step = 60 / 112 / 2  # the client's music at 112 BPM
    melody = sum(np.sin(2 * np.pi * 262 * 2 ** (k / 12) * t) * ((t // step) % 8 == i)
                 for i, k in enumerate([0, 4, 7, 12, 7, 4, 9, 7]))
    bass = 2 * np.abs(2 * ((t * 131) % 1) - 1) - 1
    warble = 520 + 140 * np.sin(2 * np.pi * 3 * t)
    return {
        "beeping appliance": 8000 * np.sin(2 * np.pi * 1000 * t) * ((t % 0.5) < 0.25),
        "lamp music": 6000 * (0.7 * melody * ((t % step) < step * 0.9) + 0.5 * bass),
        "fake TTS babble": 8000 * np.sin(2 * np.pi * np.cumsum(warble) / 16000),
    }


def _events(x: np.ndarray) -> list[tuple[str, float]]:
    pcm, v, out = x.clip(-32768, 32767).astype(np.int16).tobytes(), VAD(), []
    for i in range(0, len(pcm), 640):
        out += [(e, i / 32000) for e, _ in v.feed(pcm[i:i + 640])]
    return out


def test_tones_and_music_do_not_open_a_turn():
    for name, x in _tones().items():
        assert _events(x) == [], name  # before the harmonic gate each opened a turn that never closed


def test_speech_over_music_still_opens_a_turn_when_speech_starts():
    with wave.open(WAV) as w:
        speech = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(float)[8000:39000]
    mix = _tones()["lamp music"].copy()
    mix[32000:32000 + len(speech)] += 0.8 * speech  # speech from 2.0 s (voice onset ~2.25 s)
    ev = _events(mix)
    assert ev and ev[0][0] == "start" and 2.0 <= ev[0][1] <= 2.6


# ---------- Gemini (google-genai on a mocked httpx transport) ----------

def gemini(handler):
    from google import genai
    from google.genai import types

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return genai.Client(api_key="test", http_options=types.HttpOptions(httpx_async_client=http))


def unb64(s: str) -> bytes:  # the SDK sends inline bytes as unpadded URL-safe base64
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def gemini_reply(*parts):
    return {"candidates": [{"content": {"role": "model", "parts": list(parts)}, "finishReason": "STOP"}]}


def test_gemini_stt_sends_16k_wav_and_returns_text():
    from app.providers.gemini import GeminiSTT
    pcm, seen = recorded_utterance()[0], {}

    def handler(req):
        body = json.loads(req.content)
        seen["path"] = req.url.path
        seen["audio"] = next(p["inlineData"] for p in body["contents"][0]["parts"] if "inlineData" in p)
        return httpx.Response(200, json=gemini_reply({"text": " Hey Lumi, where did I leave my mug? "}))

    text = asyncio.run(GeminiSTT(gemini(handler)).transcribe(pcm))
    with wave.open(io.BytesIO(unb64(seen["audio"]["data"]))) as w:
        assert (w.getframerate(), w.getnchannels(), w.getnframes()) == (16000, 1, len(pcm) // 2)
    assert text == "Hey Lumi, where did I leave my mug?" and seen["audio"]["mime_type"] == "audio/wav"
    assert seen["path"].endswith(":generateContent")


def test_gemini_llm_forces_the_tool_and_parses_a_goal():
    from app.providers.gemini import GeminiLLM
    seen = {}
    reply = {"say": "On it!", "emotion": "excited", "gesture": "perk", "music": "none",
             "action": {"do": "spotlight", "target": "mug"}}

    def handler(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, json=gemini_reply({"functionCall": {"name": "respond", "args": reply}}))

    out = asyncio.run(GeminiLLM(gemini(handler)).respond(
        user_text="shine your light on my mug", history=[{"role": "user", "content": "hi"},
                                                          {"role": "assistant", "content": "hello!"}],
        memories=["mug was left of the keyboard (just now)"], scene=[{"name": "mug", "location": "left"}],
        mood="curious"))
    assert out == reply
    fc = seen["toolConfig"]["functionCallingConfig"]
    assert fc["mode"] == "ANY" and fc["allowedFunctionNames"] == ["respond"]
    system = seen["systemInstruction"]["parts"][0]["text"]
    assert "mug (left)" in system and "left of the keyboard" in system
    assert [c["role"] for c in seen["contents"]] == ["user", "model", "user"]


def test_gemini_scene_turns_boxes_into_aim_points():
    from app.providers.gemini import GeminiScene
    jpeg, seen = b"\xff\xd8fake\xff\xd9", {}
    objects = [{"name": "mug", "location": "left of the keyboard", "box_2d": [600, 100, 900, 300]},
               {"name": "notebook", "location": "right", "box_2d": [500, 700, 800, 1000]},
               {"name": "pen", "location": "center"}]  # no box: kept for memory, not aimable

    def handler(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, json=gemini_reply(
            {"functionCall": {"name": "report_scene", "args": {"objects": objects}}}))

    out = asyncio.run(GeminiScene(gemini(handler)).describe(jpeg))
    assert out[0] == {"name": "mug", "location": "left of the keyboard", "x": -0.6, "y": 0.5}
    assert out[1]["x"] == 0.7 and out[1]["y"] == 0.3 and "x" not in out[2]
    image = next(p["inlineData"] for p in seen["contents"][0]["parts"] if "inlineData" in p)
    assert unb64(image["data"]) == jpeg and image["mime_type"] == "image/jpeg"


def test_gemini_tts_streams_24k_chunks_without_a_style_prompt():
    from app.providers.gemini import GeminiTTS
    pcm = (np.sin(np.arange(24000) / 10) * 8000).astype("<i2").tobytes()  # 1 s at 24 kHz
    pieces = [pcm[:9601], pcm[9601:30000], pcm[30000:]]  # an odd split, as a stream can deliver
    seen = {}

    def handler(req):
        seen["path"], seen["body"] = req.url.path, json.loads(req.content)
        events = "".join("data: " + json.dumps(gemini_reply({"inlineData": {
            "mimeType": "audio/l16; rate=24000; channels=1", "data": base64.b64encode(p).decode()}})) + "\r\n\r\n"
            for p in pieces)
        return httpx.Response(200, content=events.encode(), headers={"content-type": "text/event-stream"})

    async def collect():
        return [c async for c in GeminiTTS(gemini(handler)).stream("Hello!")]

    chunks = asyncio.run(collect())
    assert b"".join(chunks) == pcm and all(len(c) % 2 == 0 for c in chunks) and len(chunks) == 3
    assert seen["path"].endswith(":streamGenerateContent")
    assert seen["body"]["contents"][0]["parts"][0]["text"] == "Hello!"  # the text only: styles get read aloud
    assert seen["body"]["generationConfig"]["responseModalities"] == ["AUDIO"]


def test_gemini_audio_formats_become_24k_pcm():
    from app.providers.gemini import _to_24k
    pcm16 = (np.sin(np.arange(16000) / 10) * 8000).astype("<i2").tobytes()  # 1 s at 16 kHz
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm16)
    assert _to_24k(buf.getvalue(), "audio/wav") == pcm16  # header stripped (it would play as a click)
    assert abs(len(_to_24k(pcm16, "audio/L16;codec=pcm;rate=16000")) // 2 - 24000) <= 2  # resampled, same length
    assert _to_24k(pcm16 + b"\x01", "audio/l16; rate=24000; channels=1") == pcm16  # odd byte dropped


def test_gemini_providers_build_from_settings(monkeypatch):
    from app.config import settings
    from app.providers import build_providers

    monkeypatch.setattr(settings, "providers", "gemini")
    monkeypatch.setattr(settings, "gemini_api_key", "test")
    p = build_providers()
    assert type(p.stt).__name__ == "GeminiSTT" and p.stt.c is p.llm.c is p.scene.c is p.tts.c


def test_gemini_one_call_turn_sends_audio_and_returns_what_it_heard():
    from app.providers.gemini import GeminiLLM
    pcm, seen = recorded_utterance()[0], {}
    reply = {"heard": "Hey Lumi, where did I leave my mug?", "say": "Left of your keyboard!", "emotion": "happy",
             "gesture": "nod", "music": "none"}

    def handler(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, json=gemini_reply({"functionCall": {"name": "respond", "args": reply}}))

    out = asyncio.run(GeminiLLM(gemini(handler)).respond_to_audio(
        audio=pcm, sr=16000, history=[], memories=["mug was left of the keyboard (2 min ago)"], scene=[], mood="calm"))
    assert out == reply
    audio = seen["contents"][-1]["parts"][0]["inlineData"]
    assert audio["mime_type"] == "audio/wav" and len(unb64(audio["data"])) == 44 + len(pcm)
    schema = seen["tools"][0]["functionDeclarations"][0]["parameters_json_schema"]
    assert "heard" in schema["required"] and "left of the keyboard" in seen["systemInstruction"]["parts"][0]["text"]


def _find(obj, *keys):  # the SDK mixes camelCase and snake_case when it serializes configs
    for k in keys:
        if isinstance(obj, dict) and k in obj:
            return obj[k]
    return None


def test_gemini_requests_carry_voice_language_and_fast_thinking():
    from app.providers.gemini import GeminiLLM, GeminiTTS
    seen = []

    def handler(req):
        body = json.loads(req.content)
        seen.append(body)
        if req.url.path.endswith(":streamGenerateContent"):
            pcm = base64.b64encode(b"\x00\x01" * 2400).decode()
            event = gemini_reply({"inlineData": {"mimeType": "audio/l16; rate=24000", "data": pcm}})
            return httpx.Response(200, content=f"data: {json.dumps(event)}\r\n\r\n".encode(),
                                  headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=gemini_reply({"functionCall": {"name": "respond", "args": {
            "heard": "hi", "say": "Good evening.", "emotion": "neutral", "gesture": "none", "music": "none"}}}))

    client = gemini(handler)

    async def go():
        await GeminiLLM(client).respond_to_audio(audio=b"\x00\x00" * 1600, sr=16000, history=[], memories=[], scene=[],
                                                 mood="calm")
        return [c async for c in GeminiTTS(client).stream("Good evening.")]

    asyncio.run(go())
    llm_cfg, tts_cfg = seen[0]["generationConfig"], seen[1]["generationConfig"]
    thinking = _find(llm_cfg, "thinkingConfig", "thinking_config")
    assert thinking and _find(thinking, "thinkingLevel", "thinking_level") == "MINIMAL"
    speech = _find(tts_cfg, "speechConfig", "speech_config")
    assert _find(speech, "languageCode", "language_code") == "en-GB"
    voice = json.dumps(speech)
    assert "Charon" in voice
