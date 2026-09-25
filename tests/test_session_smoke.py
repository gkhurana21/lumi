"""End-to-end over the real WebSocket with fake providers: no keys, camera, mic, or sim."""
import json

import pytest
from fastapi.testclient import TestClient

from app.config import settings


@pytest.fixture
def client():
    settings.providers, settings.memory, settings.attention, settings.sim = "fake", "inmem", "none", False
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
