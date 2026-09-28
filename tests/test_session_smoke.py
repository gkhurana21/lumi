"""End-to-end over the real WebSocket with fake providers: no keys, camera, mic, or sim."""
import json
import time
import wave

import pytest
from fastapi.testclient import TestClient

from app.config import settings


@pytest.fixture
def client():
    settings.providers, settings.memory, settings.attention, settings.sim = "fake", "inmem", "none", False
    settings.trace_dir = ""
    from app.main import app
    with TestClient(app) as c:
        yield c


def recv_until(ws, pred, limit=400):
    seen = []
    for _ in range(limit):
        m = ws.receive()
        if m.get("text"):
            d = json.loads(m["text"])
            seen.append(d)
            if pred(d):
                return d, seen
    raise AssertionError(f"condition never met; saw types {[d['type'] for d in seen][-30:]}")


def say(ws, text):
    ws.send_text(json.dumps({"type": "text", "text": text}))
    _, seen = recv_until(ws, lambda d: d["type"] == "tts_end")
    ws.send_text(json.dumps({"type": "playback_done"}))
    return seen


def test_music_request_plays_after_speech(client):
    with client.websocket_connect("/ws") as ws:
        seen = say(ws, "play me a song")
        assert any(d["type"] == "metrics" for d in seen)
        d, _ = recv_until(ws, lambda d: d["type"] == "music")
        assert d["action"] == "play" and d["bpm"] == 112


def test_scene_memory_answers_later(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_bytes(b"\x02not-a-real-jpeg")  # FakeScene ignores content, reports a mug
        recv_until(ws, lambda d: d["type"] == "scene")
        seen = say(ws, "where is my mug?")
        robot = [d["text"] for d in seen if d["type"] == "transcript" and d["role"] == "robot"]
        assert robot and "mug" in robot[-1] and "keyboard" in robot[-1]


class _ScriptedAttention:
    """Real decide() (hysteresis + debounce); readings scripted by the frame payload instead of MediaPipe."""
    READINGS = {b"look": dict(face=True, yaw=0.0, w=0.2), b"away": dict(face=True, yaw=0.6, w=0.2)}

    @classmethod
    def make(cls):
        from app.vision.attention import AttentionTracker
        from app.vision.types import Reading

        class Scripted(AttentionTracker):
            def read(self, jpeg):
                return Reading(**cls.READINGS.get(jpeg, {}))
        return Scripted


def frames(ws, what: bytes, n: int) -> None:
    for _ in range(n):
        ws.send_bytes(b"\x02" + what)
        time.sleep(0.04)  # the server drops frames that arrive while one is being processed


def is_state(name):
    return lambda d: d["type"] == "state" and d["state"] == name


def test_engagement_greets_then_disengages_even_if_look_away_lands_mid_speech(client, monkeypatch):
    from app.core.fsm import TIMEOUTS, State

    monkeypatch.setattr(settings, "attention", "mediapipe")
    monkeypatch.setattr("app.vision.attention.AttentionTracker", _ScriptedAttention.make())
    monkeypatch.setitem(TIMEOUTS, State.DISENGAGING, 0.5)
    with client.websocket_connect("/ws") as ws:
        frames(ws, b"look", 4)
        _, seen = recv_until(ws, lambda d: d["type"] == "tts_end")  # greeting spoken
        states = [d["state"] for d in seen if d["type"] == "state"]
        assert states[:3] == ["noticing", "engaged", "speaking"]
        assert [d["name"] for d in seen if d["type"] == "sfx"][:2] == ["notice", "greet"]

        frames(ws, b"away", 8)  # attention drops while the lamp is still speaking: ATTN_OFF is ignored there
        ws.send_text(json.dumps({"type": "playback_done"}))
        recv_until(ws, is_state("engaged"), limit=60)
        frames(ws, b"away", 1)  # level check: back in ENGAGED with attention off -> disengage
        _, seen2 = recv_until(ws, is_state("disengaging"), limit=60)
        _, seen3 = recv_until(ws, is_state("idle"), limit=60)
        sfx = [d["name"] for d in seen + seen2 + seen3 if d["type"] == "sfx"]
        assert sfx.count("greet") == 1 and sfx[-1] == "sleep"


def test_memory_answers_about_an_object_that_left_the_scene(client, monkeypatch):
    from app.providers.fakes import FakeScene

    scans = iter([[{"name": "mug", "location": "left of the keyboard"}, {"name": "notebook", "location": "front"}],
                  [{"name": "notebook", "location": "front"}]])

    async def describe(self, jpeg):
        return next(scans, [{"name": "notebook", "location": "front"}])

    monkeypatch.setattr(FakeScene, "describe", describe)
    monkeypatch.setattr(settings, "scene_interval_s", 0.0)
    with client.websocket_connect("/ws") as ws:
        ws.send_bytes(b"\x02frame-with-mug")
        first, _ = recv_until(ws, lambda d: d["type"] == "scene")
        ws.send_bytes(b"\x02frame-without-mug")
        latest, _ = recv_until(ws, lambda d: d["type"] == "scene")
        assert "mug" in [o["name"] for o in first["objects"]]
        assert "mug" not in [o["name"] for o in latest["objects"]]  # gone from scene_now
        seen = say(ws, "where did I leave my mug?")
        robot = [d["text"] for d in seen if d["type"] == "transcript" and d["role"] == "robot"]
        assert robot and "mug" in robot[-1] and "left of the keyboard" in robot[-1]


def scripted_scenes(monkeypatch, *scans):
    """Each scene scan returns the next list; the last one repeats."""
    from app.providers.fakes import FakeScene

    it, last = iter(scans), {"v": scans[-1]}

    async def describe(self, jpeg):
        last["v"] = next(it, last["v"])
        return last["v"]

    monkeypatch.setattr(FakeScene, "describe", describe)


def run_goal(ws, text):
    """Ask for a goal, let the acknowledgement play, and collect everything until the outcome line is spoken."""
    ws.send_bytes(b"\x02frame")  # gives the session a frame to re-observe
    recv_until(ws, lambda d: d["type"] == "scene")
    say(ws, text)  # acknowledgement, then playback_done -> ENGAGED -> GOAL
    _, seen = recv_until(ws, lambda d: d["type"] == "tts_end")  # outcome line
    ws.send_text(json.dumps({"type": "playback_done"}))
    _, after = recv_until(ws, is_state("engaged"), limit=60)
    return seen, after


MUG = {"name": "mug", "location": "left of the keyboard", "x": -0.5, "y": 0.6}


def test_goal_spotlights_the_mug_after_rechecking_the_scene(client, monkeypatch):
    scripted_scenes(monkeypatch, [MUG, {"name": "notebook", "location": "front", "x": 0.5, "y": 0.7}])
    with client.websocket_connect("/ws") as ws:
        seen, after = run_goal(ws, "shine your light on my mug")
    steps = [d["step"] for d in seen if d["type"] == "goal"]
    assert steps == ["start", "aim", "observe", "verify", "done"]
    assert [d["state"] for d in seen if d["type"] == "state"] == ["engaged", "acting", "speaking"]
    assert "spot" in [d["name"] for d in seen if d["type"] == "sfx"]
    lit = [d for d in seen if d["type"] == "body" and d["light"]["brightness"] == 1.0]
    assert lit and lit[-1]["joints"]["base_yaw_joint"] > 0.05  # mug is image-left: lamp turns to its left
    assert lit[-1].get("focus") == "mug"  # the sim labels the lit spot with the target's name
    outcome = [d["text"] for d in seen if d["type"] == "transcript" and d["role"] == "robot"][-1]
    assert "mug" in outcome and "spotlight" in outcome
    bodies = [d for d in after if d["type"] == "body"]
    assert not bodies or (bodies[-1]["light"]["brightness"] < 1.0 and "focus" not in bodies[-1])  # released


def test_goal_reaims_when_the_target_moved(client, monkeypatch):
    moved = {**MUG, "x": 0.3}
    scripted_scenes(monkeypatch, [MUG], [moved])
    with client.websocket_connect("/ws") as ws:
        seen, _ = run_goal(ws, "shine your light on my mug")
    aims = [d["x"] for d in seen if d["type"] == "goal" and d["step"] == "aim"]
    assert aims == [-0.5, 0.3]
    last = [d for d in seen if d["type"] == "goal"][-1]
    assert last["step"] == "done" and last["attempts"] == 2
    assert "Caught it moving" in [d["text"] for d in seen if d["type"] == "transcript"][-1]


def test_goal_reports_an_invisible_target(client, monkeypatch):
    scripted_scenes(monkeypatch, [{"name": "notebook", "location": "front", "x": 0.5, "y": 0.7}])
    with client.websocket_connect("/ws") as ws:
        seen, _ = run_goal(ws, "shine your light on my phone")
    goal = [d for d in seen if d["type"] == "goal"]
    assert [d["step"] for d in goal] == ["start", "observe", "failed"] and goal[-1]["why"] == "not visible"
    assert "phone" in [d["text"] for d in seen if d["type"] == "transcript"][-1]


def test_talking_over_a_goal_cancels_it(client, monkeypatch):
    scripted_scenes(monkeypatch, [MUG])
    with client.websocket_connect("/ws") as ws:
        ws.send_bytes(b"\x02frame")
        recv_until(ws, lambda d: d["type"] == "scene")
        say(ws, "shine your light on my mug")
        recv_until(ws, is_state("acting"), limit=60)
        ws.send_text(json.dumps({"type": "text", "text": "never mind"}))
        _, seen = recv_until(ws, is_state("listening"), limit=60)
        _, seen2 = recv_until(ws, lambda d: d["type"] == "tts_end")
    goal_steps = [d["step"] for d in seen + seen2 if d["type"] == "goal"]
    assert "done" not in goal_steps and "failed" not in goal_steps


NOTEBOOK = {"name": "notebook", "location": "front right", "x": 0.55, "y": 0.7}


@pytest.fixture
def live(client, monkeypatch):
    """Scripted camera (look/away frames) and a desk the test can change, over the real WebSocket."""
    from app.core.fsm import TIMEOUTS, State
    from app.providers.fakes import FakeScene

    desk = {"objs": [MUG, NOTEBOOK]}

    async def describe(self, jpeg):
        return list(desk["objs"])

    monkeypatch.setattr(FakeScene, "describe", describe)
    monkeypatch.setattr(settings, "attention", "mediapipe")
    monkeypatch.setattr("app.vision.attention.AttentionTracker", _ScriptedAttention.make())
    monkeypatch.setattr(settings, "scene_interval_s", 0.0)
    monkeypatch.setitem(TIMEOUTS, State.DISENGAGING, 0.5)
    return desk


def test_demo_script_end_to_end(client, live):
    """README demo script as one continuous session: every spec moment, with motion, light, voice, SFX, music."""
    log: list[dict] = []

    def until(pred, limit=200):
        _, seen = recv_until(ws, pred, limit)
        log.extend(seen)

    def spoken(text):
        log.extend(say(ws, text))

    with client.websocket_connect("/ws") as ws:
        until(lambda d: d["type"] == "body")  # 1. idle
        frames(ws, b"look", 4)  # 2. engagement: notice, greet
        until(lambda d: d["type"] == "tts_end")
        ws.send_text(json.dumps({"type": "playback_done"}))
        spoken("what do you see on my desk?")  # 3. spoken interaction, grounded in the scene
        spoken("shine your light on my mug")  # 4. goal-directed action
        until(lambda d: d["type"] == "tts_end")
        ws.send_text(json.dumps({"type": "playback_done"}))
        until(is_state("engaged"), limit=60)
        live["objs"] = [NOTEBOOK]  # 5. the mug leaves the desk; memory keeps it
        frames(ws, b"look", 2)
        until(lambda d: d["type"] == "scene" and "mug" not in [o["name"] for o in d["objects"]])
        spoken("where did I leave my mug?")
        spoken("play me a song")  # 6. music and dancing
        until(lambda d: d["type"] == "music")
        frames(ws, b"look", 10)  # dance for a couple of seconds
        until(lambda d: d["type"] == "body", limit=20)
        frames(ws, b"away", 8)  # 7. disengagement
        until(is_state("idle"), limit=100)

    states = {d["state"] for d in log if d["type"] == "state"}
    assert states >= {"noticing", "engaged", "listening", "thinking", "speaking", "acting", "disengaging", "idle"}
    sfx = {d["name"] for d in log if d["type"] == "sfx"}
    assert sfx >= {"notice", "greet", "listen", "spot", "sleep"}
    assert [d["action"] for d in log if d["type"] == "music"] == ["play", "stop"]
    robot = " | ".join(d["text"] for d in log if d["type"] == "transcript" and d["role"] == "robot")
    assert "I can see mug" in robot and "spotlight" in robot and "I remember: mug was left of the keyboard" in robot
    bodies = [d for d in log if d["type"] == "body"]
    bright = [d["light"]["brightness"] for d in bodies]
    assert min(bright) <= 0.15 and max(bright) == 1.0  # dim while idle, spotlight for the goal
    assert len({tuple(d["light"]["rgb"]) for d in bodies}) > 5  # color changes (mood, spotlight, dance)
    assert any(d["face"] and d["face"]["attending"] for d in bodies)  # the client can draw what the lamp sees
    assert any(d["face"] is None for d in bodies)  # and that nobody is there before the user looks
    yaw = [d["joints"]["base_yaw_joint"] for d in bodies]
    assert max(yaw) - min(yaw) > 0.1  # the body visibly moves


def test_a_glance_that_never_engages_settles_without_the_sleep_sound(client, live):
    with client.websocket_connect("/ws") as ws:
        frames(ws, b"look", 2)  # face seen, attention not yet confirmed
        recv_until(ws, is_state("noticing"), limit=60)
        frames(ws, b"gone", 9)
        _, seen = recv_until(ws, is_state("idle"), limit=60)
    assert "sleep" not in [d["name"] for d in seen if d["type"] == "sfx"]


def test_looking_back_while_disengaging_is_welcomed(client, live, monkeypatch):
    monkeypatch.setattr(settings, "greet_cooldown_s", 1e12)  # no greeting speech in this test
    with client.websocket_connect("/ws") as ws:
        frames(ws, b"look", 4)
        recv_until(ws, is_state("engaged"), limit=60)
        frames(ws, b"away", 7)
        recv_until(ws, is_state("disengaging"), limit=60)
        frames(ws, b"look", 4)
        _, seen = recv_until(ws, is_state("engaged"), limit=60)
    assert "notice" in [d["name"] for d in seen if d["type"] == "sfx"]


def test_greeting_voice_is_synthesized_before_it_is_needed(client, live, monkeypatch):
    from app.providers.fakes import FakeTTS
    from app.session import GREETINGS

    calls, orig = [], FakeTTS.stream

    def counting(self, text):
        calls.append(text)
        return orig(self, text)

    monkeypatch.setattr(FakeTTS, "stream", counting)
    with client.websocket_connect("/ws") as ws:
        recv_until(ws, lambda d: d["type"] == "body")
        for _ in range(60):  # prefetch runs at session start, one line after another (paced like a real stream)
            if len(calls) == len(GREETINGS):
                break
            time.sleep(0.05)
        time.sleep(1.0)  # let the last line finish
        assert sorted(calls) == sorted(GREETINGS)
        frames(ws, b"look", 4)
        _, seen = recv_until(ws, lambda d: d["type"] == "tts_end")
    greeting = [d["text"] for d in seen if d["type"] == "transcript" and d["role"] == "robot"][0]
    assert greeting in GREETINGS and sorted(calls) == sorted(GREETINGS)  # served from the cache, no new TTS call


def test_looking_at_the_target_while_asking_still_runs_the_goal(client, live, monkeypatch):
    monkeypatch.setattr(settings, "greet_cooldown_s", 1e12)
    with client.websocket_connect("/ws") as ws:
        frames(ws, b"look", 4)
        recv_until(ws, is_state("engaged"), limit=60)
        ws.send_text(json.dumps({"type": "text", "text": "shine your light on my mug"}))
        recv_until(ws, lambda d: d["type"] == "tts_end")  # acknowledgement spoken
        frames(ws, b"away", 7)  # the user looks at the mug: attention drops while the lamp is still speaking
        ws.send_text(json.dumps({"type": "playback_done"}))
        frames(ws, b"away", 2)
        _, seen = recv_until(ws, lambda d: d["type"] == "goal" and d["step"] in ("done", "failed"), limit=100)
    assert "acting" in [d["state"] for d in seen if d["type"] == "state"] and seen[-1]["step"] == "done"


def test_a_failing_scene_scan_mid_goal_is_audible(client, live, monkeypatch):
    from app.providers.fakes import FakeScene

    calls = {"n": 0}

    async def flaky(self, jpeg):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("vision API down")
        return [MUG]

    monkeypatch.setattr(FakeScene, "describe", flaky)
    with client.websocket_connect("/ws") as ws:
        ws.send_bytes(b"\x02frame")
        recv_until(ws, lambda d: d["type"] == "scene")
        say(ws, "shine your light on my mug")
        _, seen = recv_until(ws, lambda d: d["type"] == "state" and d["prev"] == "acting", limit=100)
    assert seen[-1]["state"] == "engaged" and "error" in [d["name"] for d in seen if d["type"] == "sfx"]


def speak_audio(ws, pcm: bytes) -> None:
    for i in range(0, len(pcm), 640):  # 20 ms chunks, like the browser
        ws.send_bytes(b"\x01" + pcm[i:i + 640])


def test_spoken_turn_and_barge_in_from_recorded_audio(client):
    from tests.test_providers import recorded_utterance

    pcm, _ = recorded_utterance()
    with client.websocket_connect("/ws") as ws:
        speak_audio(ws, pcm)  # real VAD segmentation over the socket
        _, seen = recv_until(ws, lambda d: d["type"] == "tts_start")
        states = [d["state"] for d in seen if d["type"] == "state"]
        assert states == ["listening", "thinking", "speaking"]
        speak_audio(ws, pcm)  # talk over the lamp before its playback is done
        _, seen = recv_until(ws, is_state("listening"), limit=100)
    assert "stop_audio" in [d["type"] for d in seen]  # server told the client to cut the voice


def test_a_beeping_sound_does_not_start_a_conversation(client):
    import numpy as np

    t = np.arange(16000 * 3) / 16000
    beep = (8000 * np.sin(2 * np.pi * 1000 * t) * ((t % 0.5) < 0.25)).astype(np.int16).tobytes()
    with client.websocket_connect("/ws") as ws:
        speak_audio(ws, beep)
        seen = []
        while sum(d["type"] == "body" for d in seen) < 8:  # ~1.6 s of the body stream after the beeps
            seen.append(recv_until(ws, lambda d: True)[0])
    assert not [d for d in seen if d["type"] in ("state", "sfx")]  # no listen blip, lamp still idle


def test_a_short_noise_does_not_cancel_a_goal(client, monkeypatch):
    import numpy as np

    from tests.test_providers import WAV
    scripted_scenes(monkeypatch, [MUG])
    with wave.open(WAV) as w:
        speech = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    silence = np.zeros(16000, np.int16)
    burst = np.concatenate([silence[:1600], speech[12000:15200], silence])  # 200 ms of voice
    with client.websocket_connect("/ws") as ws:
        ws.send_bytes(b"\x02frame")
        recv_until(ws, lambda d: d["type"] == "scene")
        say(ws, "shine your light on my mug")
        recv_until(ws, is_state("acting"), limit=60)
        speak_audio(ws, burst.tobytes())  # 200 ms: passes the normal 120 ms start, not the strict 360 ms one
        _, seen = recv_until(ws, lambda d: d["type"] == "goal" and d["step"] in ("done", "failed"), limit=100)
    assert "listening" not in [d.get("state") for d in seen] and seen[-1]["step"] == "done"


def test_voice_falls_back_to_browser_speech_when_tts_fails(client, monkeypatch):
    from app.providers.fakes import FakeTTS

    calls = []

    async def broken(self, text):
        calls.append(text)
        raise RuntimeError("429 quota exceeded")
        yield b""  # pragma: no cover (makes this an async generator)

    monkeypatch.setattr(FakeTTS, "stream", broken)
    with client.websocket_connect("/ws") as ws:
        for _ in range(2):
            ws.send_text(json.dumps({"type": "text", "text": "what do you see?"}))
            d, seen = recv_until(ws, lambda d: d["type"] == "speak")
            assert d["text"] and "tts_end" not in [x["type"] for x in seen]  # the browser speaks this line
            ws.send_text(json.dumps({"type": "playback_done"}))  # sent by the client when its speech ends
            recv_until(ws, is_state("engaged"))
    greetings = 3  # prefetched at session start, also failing
    assert len(calls) == greetings + 1  # the second line skipped the broken TTS (backoff) and went straight to speech
