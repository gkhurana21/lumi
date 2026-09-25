"""One WebSocket connection = one live interaction. Owns the FSM and wires perception -> mind -> body."""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time

from fastapi import WebSocket

from .audio.vad import VAD
from .body.behaviors import Body
from .config import settings
from .core.fsm import Ev, Machine, State
from .protocol import Frame, pack
from .providers import Providers

log = logging.getLogger("session")

GREETINGS = ["Oh! Hi there.", "Hey, you! I was hoping you'd look over.", "Hello hello!"]
MUSIC_BPM = {"happy": 112, "chill": 84}


class Session:
    def __init__(self, ws: WebSocket, providers: Providers, memory, body: Body, attention=None):
        self.ws, self.p, self.memory, self.body = ws, providers, memory, body
        if attention is None:
            if settings.attention == "mediapipe":
                from .vision.attention import AttentionTracker  # lazy: mediapipe is heavy
                attention = AttentionTracker()
            else:
                from .vision.null import NullAttention
                attention = NullAttention()
        self.attn = attention
        self.fsm = Machine(on_transition=self._on_transition)
        self.vad = VAD()
        self.out: asyncio.Queue = asyncio.Queue(maxsize=512)
        self.history: list[dict] = []
        self.scene_now: list[dict] = []
        self._pending: bytes | str | None = None
        self._reply: asyncio.Task | None = None
        self._pending_music: str | None = None
        self._gaze = (0.0, 0.0)
        self._vision_busy = self._scene_busy = False
        self._last_scene = 0.0
        self._last_greet = -1e9
        self._bg: set[asyncio.Task] = set()
        self._t_end: float | None = None  # perf_counter at end of user speech, for latency metrics

    # ---------- plumbing ----------
    def _spawn(self, coro) -> asyncio.Task:
        t = asyncio.create_task(coro)
        self._bg.add(t)
        t.add_done_callback(self._done)
        return t

    def _done(self, t: asyncio.Task) -> None:
        self._bg.discard(t)
        if not t.cancelled() and t.exception():
            log.error("task failed", exc_info=t.exception())

    def _emit(self, obj: dict) -> None:
        try:
            self.out.put_nowait(json.dumps(obj))
        except asyncio.QueueFull:
            pass

    def _purge_audio(self) -> None:
        keep = []
        while not self.out.empty():
            item = self.out.get_nowait()
            if not isinstance(item, (bytes, bytearray)):
                keep.append(item)
        for item in keep:
            self.out.put_nowait(item)

    async def _sender(self) -> None:
        while True:
            item = await self.out.get()
            if isinstance(item, (bytes, bytearray)):
                await self.ws.send_bytes(item)
            else:
                await self.ws.send_text(item)

    async def run(self) -> None:
        self._spawn(self._sender())
        self._spawn(self._ticker())
        try:
            while True:
                msg = await self.ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                data = msg.get("bytes")
                if data:
                    kind, payload = data[0], data[1:]
                    if kind == Frame.AUDIO_IN:
                        self._on_audio(payload)
                    elif kind == Frame.VIDEO_IN and not self._vision_busy:  # latest-frame-wins
                        self._vision_busy = True
                        self._spawn(self._on_video(payload))
                elif msg.get("text"):
                    self._on_control(json.loads(msg["text"]))
        finally:
            for t in list(self._bg):
                t.cancel()

    # ---------- inputs ----------
    def _on_audio(self, pcm: bytes) -> None:
        for ev, utt in self.vad.feed(pcm):
            if ev == "start":
                self.fsm.fire(Ev.SPEECH_START)
            else:
                self._pending = utt
                self._t_end = time.perf_counter()
                self.fsm.fire(Ev.SPEECH_END)

    def _on_control(self, m: dict) -> None:
        t = m.get("type")
        if t == "playback_done":
            self.fsm.fire(Ev.TTS_DONE)
        elif t == "text" and m.get("text"):  # typed input for demos without a mic
            self.fsm.fire(Ev.SPEECH_START)
            self._pending = m["text"]
            self._t_end = time.perf_counter()
            self.fsm.fire(Ev.SPEECH_END)

    async def _on_video(self, jpeg: bytes) -> None:
        try:
            events, reading = await asyncio.to_thread(self.attn.update, jpeg)
            if reading.face:
                self._gaze = (reading.x, reading.y)
            for ev in events:
                self.fsm.fire(ev)
            now = time.monotonic()
            if (not self._scene_busy and now - self._last_scene > settings.scene_interval_s
                    and self.fsm.state != State.SPEAKING):
                self._last_scene = now
                self._scene_busy = True
                self._spawn(self._scan_scene(jpeg))
        finally:
            self._vision_busy = False

    async def _scan_scene(self, jpeg: bytes) -> None:
        try:
            objs = await self.p.scene.describe(jpeg)
            if objs:
                self.scene_now = objs
                await asyncio.to_thread(self.memory.upsert_objects, objs, time.time())
                self._emit({"type": "scene", "objects": objs})
        finally:
            self._scene_busy = False

    # ---------- state machine side effects ----------
    def _on_transition(self, prev: State, nxt: State, ev: Ev) -> None:
        log.info("%s -> %s (%s)", prev.value, nxt.value, ev.value)
        if prev in (State.THINKING, State.SPEAKING) and nxt != State.SPEAKING:
            self._cancel_reply()
            if prev == State.SPEAKING and ev != Ev.TTS_DONE:
                self._purge_audio()
                self._emit({"type": "stop_audio"})
        self.vad.strict = nxt == State.SPEAKING

        if nxt == State.NOTICING:
            self._emit({"type": "sfx", "name": "notice"})
            self.body.gesture("perk")
        elif nxt == State.ENGAGED and prev in (State.IDLE, State.NOTICING):
            if time.monotonic() - self._last_greet > settings.greet_cooldown_s:
                self._last_greet = time.monotonic()
                self._spawn(self._greet())
        elif nxt == State.LISTENING:
            self._emit({"type": "sfx", "name": "listen"})
        elif nxt == State.THINKING:
            utt, self._pending = self._pending, None
            if utt is None:
                utt = self.vad.flush()
            self._reply = self._spawn(self._think(utt))
        elif nxt == State.ENGAGED and prev == State.SPEAKING and self._pending_music:
            self._set_music(self._pending_music)
            self._pending_music = None
        elif nxt == State.IDLE:
            self._emit({"type": "sfx", "name": "sleep"})
            self._set_music("stop")

        self._emit({"type": "state", "state": nxt.value, "prev": prev.value, "event": ev.value})

    def _cancel_reply(self) -> None:
        if self._reply and not self._reply.done() and self._reply is not asyncio.current_task():
            self._reply.cancel()

    def _set_music(self, mood: str) -> None:
        if mood in MUSIC_BPM:
            self.body.dance_bpm = MUSIC_BPM[mood]
            self._emit({"type": "music", "action": "play", "mood": mood, "bpm": MUSIC_BPM[mood]})
        elif mood == "stop" and self.body.dance_bpm:
            self.body.dance_bpm = None
            self._emit({"type": "music", "action": "stop"})

    # ---------- mind ----------
    async def _greet(self) -> None:
        self._emit({"type": "sfx", "name": "greet"})
        self.fsm.affect.set_emotion("happy")
        self.body.gesture("bounce")
        await asyncio.sleep(0.4)  # let the chirp + motion land before the voice
        if self.fsm.state == State.ENGAGED and self.fsm.fire(Ev.REPLY_READY):
            self._reply = asyncio.current_task()
            await self._speak(random.choice(GREETINGS), "happy")

    async def _think(self, utt: bytes | str) -> None:
        t0 = self._t_end or time.perf_counter()
        marks: dict[str, float] = {}
        try:
            text = utt if isinstance(utt, str) else (await self.p.stt.transcribe(utt) if utt else "")
            marks["stt_ms"] = (time.perf_counter() - t0) * 1000
            if not text.strip():
                self.fsm.fire(Ev.ABORT)
                return
            self._emit({"type": "transcript", "role": "user", "text": text})
            memories = await asyncio.to_thread(self.memory.search, text)
            reply = await self.p.llm.respond(user_text=text, history=self.history[-10:], memories=memories,
                                             scene=self.scene_now, mood=self.fsm.affect.label())
            marks["llm_ms"] = (time.perf_counter() - t0) * 1000 - marks["stt_ms"]
            self.history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply["say"]}]
            emotion = reply.get("emotion", "neutral")
            self.fsm.affect.set_emotion(emotion)
            if reply.get("gesture", "none") != "none":
                self.body.gesture(reply["gesture"])
            music = reply.get("music", "none")
            if music == "stop":
                self._set_music("stop")
            elif music in MUSIC_BPM:
                self._pending_music = music  # starts after the line is spoken
            if self.fsm.fire(Ev.REPLY_READY):
                await self._speak(reply["say"], emotion, t0=t0, marks=marks)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("reply pipeline failed")
            self.fsm.affect.set_emotion("confused")
            self._emit({"type": "sfx", "name": "error"})
            self.fsm.fire(Ev.ABORT)

    async def _speak(self, text: str, emotion: str, t0: float | None = None, marks: dict | None = None) -> None:
        self._emit({"type": "transcript", "role": "robot", "text": text, "emotion": emotion})
        self._emit({"type": "tts_start"})
        first = True
        async for chunk in self.p.tts.stream(text):
            if first and t0 is not None and marks is not None:
                marks["first_audio_ms"] = (time.perf_counter() - t0) * 1000
                marks = {k: round(v) for k, v in marks.items()}
                log.info("latency %s", marks)
                self._emit({"type": "metrics", **marks})
            first = False
            await self.out.put(pack(Frame.AUDIO_OUT, chunk))
        self._emit({"type": "tts_end"})  # client replies playback_done when the speaker is actually quiet

    # ---------- body clock ----------
    async def _ticker(self) -> None:
        dt, n = 0.05, 0
        while True:
            await asyncio.sleep(dt)
            self.fsm.tick(dt)
            frame = self.body.update(self.fsm.state, self.fsm.affect, self._gaze, dt)
            n += 1
            if n % 4 == 0:  # 5 Hz to the UI, 20 Hz to the sim
                a = self.fsm.affect
                self._emit({"type": "body", **frame, "mood": a.label(),
                            "va": [round(a.valence, 2), round(a.arousal, 2)]})
