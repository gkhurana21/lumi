"""End-to-end over the real WebSocket with fake providers: no keys, camera, mic, or sim."""
import json
import time

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
