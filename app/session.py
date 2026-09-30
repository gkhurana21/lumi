"""One WebSocket connection = one live interaction. Owns the FSM and wires perception -> mind -> body."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import random
import re
import time

from fastapi import WebSocket

from .audio.vad import VAD
from .body.behaviors import Body
from .config import settings
from .core.fsm import Ev, Machine, State
from .mind.memory import tokens
from .protocol import Frame, pack
from .providers import Providers

log = logging.getLogger("session")

GREETINGS = ["Good to see you. How can I help?", "Welcome back. At your service.",
             "Ah, there you are. What can I do for you?"]
MUSIC_BPM = {"happy": 112, "chill": 84}
SPOT_RGB = [1.0, 0.97, 0.9]  # near-white spotlight for goals
MOVE_TOL = 0.25  # image units ([-1, 1] per axis): larger shift between aim and re-observation = the object moved
AIM_TIMEOUT_S = 2.5
TTS_CACHE = "data/tts_cache"
# Seconds a sound effect (plus room echo) keeps the mic from opening a turn: the lamp must not answer its own chirp.
SFX_GUARD_S = {"notice": 0.45, "greet": 0.55, "listen": 0.35, "sleep": 0.9, "error": 0.55, "spot": 0.5}
FILLERS = {"um", "uh", "uhm", "hmm", "hm", "mm", "mhm", "ah", "oh", "er", "erm"}


def real_words(text: str) -> bool:
    """True if a transcript holds actual words: not empty, not only [noises] or *actions*, not only fillers."""
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*", " ", text)
    return any(w.lower() not in FILLERS for w in re.findall(r"[^\W\d_]{2,}", text))
TTS_BACKOFF_S = 60  # after a TTS failure (quota, network), the browser speaks for this long before TTS is retried


def find_object(objects: list[dict], name: str) -> dict | None:
    """Best token match for a goal target among scene objects ("mug" matches "white mug")."""
    want = tokens(name)
    best = max(objects, key=lambda o: len(want & tokens(o.get("name", ""))), default=None)
    return best if best and want & tokens(best.get("name", "")) else None


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
        self._face: dict | None = None  # latest face reading, shown by the client as the lamp's view
        self._vision_busy = self._scene_busy = False
        self._last_scene = 0.0
        self._last_greet = -1e9
        self._bg: set[asyncio.Task] = set()
        self._t_end: float | None = None  # perf_counter at end of user speech, for latency metrics
        self._last_jpeg: bytes | None = None
        self._goal: dict | None = None  # goal from the last reply, run after its acknowledgement is spoken
        self._goal_task: asyncio.Task | None = None
        self._look: tuple[float, float] | None = None  # goal target overriding face gaze (camera frame, [-1, 1])
        self._tts_cache: dict[str, list[bytes]] = {}  # fixed lines synthesized once, so they land on their beat
        self._tts_down_until = 0.0
        self._quiet_until = 0.0  # mic may not open a turn before this (the lamp's own sound effect is playing)
        self._typed = False  # the turn being thought about was typed: the mic cannot preempt it
        self._trace_f = None
        if settings.trace_dir:
            os.makedirs(settings.trace_dir, exist_ok=True)
            path = os.path.join(settings.trace_dir, time.strftime("%Y%m%d-%H%M%S") + ".jsonl")
            self._trace_f = open(path, "a", buffering=1)  # line-buffered; ~150 B at 5 Hz, negligible on the loop

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

    def _trace(self, kind: str, **rec) -> None:
        """Local measurement log (numbers only, never images or audio). Read by scripts/analyze_trace.py."""
        if self._trace_f:
            self._trace_f.write(json.dumps({"t": round(time.monotonic(), 3), "kind": kind, **rec}) + "\n")

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
        self._emit({"type": "hello", "providers": settings.providers, "urdf": "/robot/" + os.path.basename(
            settings.urdf_path), "one_call": settings.one_call_turns and hasattr(self.p.llm, "respond_to_audio")})
        self._spawn(self._sender())
        self._spawn(self._ticker())
        if hasattr(self.attn, "warm"):
            self._spawn(asyncio.to_thread(self.attn.warm))
        self._spawn(self._prefetch(GREETINGS))
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
            if self._trace_f:
                self._trace_f.close()

    # ---------- inputs ----------
    def _on_audio(self, pcm: bytes) -> None:
        for ev, utt in self.vad.feed(pcm):
            if ev == "start":
                why = ("own sound effect" if time.monotonic() < self._quiet_until else
                       "typed turn" if self.fsm.state == State.THINKING and self._typed else None)
                if why:
                    self.vad.flush()
                    self._trace("vad", ev="ignored", why=why)
                    continue
                self._trace("vad", ev="start", during=self.fsm.state.value, strict=self.vad.strict)
                self.fsm.fire(Ev.SPEECH_START)
            else:
                self._trace("vad", ev="end", sec=round(len(utt) / 32000, 2))
                self._pending = utt
                self._t_end = time.perf_counter()
                self.fsm.fire(Ev.SPEECH_END)

    def _on_control(self, m: dict) -> None:
        t = m.get("type")
        if t == "playback_done":
            self.fsm.fire(Ev.TTS_DONE)
            self._start_goal()
        elif t == "text" and m.get("text"):  # typed input for demos without a mic
            self.vad.flush()  # whatever the mic was half-hearing is superseded by the typed words
            self.fsm.fire(Ev.SPEECH_START)
            self._pending = m["text"]
            self._t_end = time.perf_counter()
            self.fsm.fire(Ev.SPEECH_END)

    async def _on_video(self, jpeg: bytes) -> None:
        try:
            self._last_jpeg = jpeg
            t0 = time.perf_counter()
            events, reading = await asyncio.to_thread(self.attn.update, jpeg)
            self._trace("frame", face=reading.face, attending=reading.attending, yaw=round(reading.yaw, 3),
                        x=round(reading.x, 3), y=round(reading.y, 3), w=round(reading.w, 3),
                        level=self.attn.attending, ms=round((time.perf_counter() - t0) * 1000, 1))
            if reading.face:
                self._gaze = (reading.x, reading.y)
            self._face = ({"x": round(reading.x, 3), "y": round(reading.y, 3), "w": round(reading.w, 3),
                           "h": round(reading.h, 3), "attending": reading.attending} if reading.face else None)
            for ev in events:
                self.fsm.fire(ev)
            # Edge events fired while SPEAKING/THINKING are dropped by the FSM, so re-check the level:
            # never sit in ENGAGED while debounced attention is off (None = no camera, unknown).
            if self.fsm.state == State.ENGAGED and self.attn.attending is False:
                self.fsm.fire(Ev.ATTN_OFF)
            now = time.monotonic()
            if (not self._scene_busy and now - self._last_scene > settings.scene_interval_s
                    and self.fsm.state != State.SPEAKING):
                self._last_scene = now
                self._scene_busy = True
                self._spawn(self._scan_scene(jpeg))
        finally:
            self._vision_busy = False

    async def _scan_scene(self, jpeg: bytes) -> list[dict]:
        try:
            objs = await self.p.scene.describe(jpeg)
            if objs:
                self.scene_now = objs
                await asyncio.to_thread(self.memory.upsert_objects, objs, time.time())
                self._emit({"type": "scene", "objects": objs})
            return objs
        finally:
            self._scene_busy = False

    async def _observe(self) -> list[dict]:
        """Fresh scan of the latest frame for a goal, bypassing the periodic interval (waits out one in flight)."""
        for _ in range(100):
            if not self._scene_busy:
                break
            await asyncio.sleep(0.05)
        if self._last_jpeg is None:  # no camera frames (typed-input demo): best effort
            return self.scene_now
        self._scene_busy, self._last_scene = True, time.monotonic()
        return await self._scan_scene(self._last_jpeg)

    # ---------- state machine side effects ----------
    def _on_transition(self, prev: State, nxt: State, ev: Ev) -> None:
        log.info("%s -> %s (%s)", prev.value, nxt.value, ev.value)
        self._trace("transition", prev=prev.value, next=nxt.value, event=ev.value)
        if prev == State.ACTING and nxt != State.SPEAKING:
            self._cancel(self._goal_task)  # interrupted or timed out mid-goal
        if nxt not in (State.ACTING, State.SPEAKING):  # the goal's aim and spotlight last through its outcome line
            self._look, self.body.light_override, self.body.focus = None, None, None
        if nxt in (State.LISTENING, State.DISENGAGING, State.IDLE):
            self._goal = None
        if prev in (State.THINKING, State.SPEAKING) and nxt != State.SPEAKING:
            self._cancel_reply()
            if prev == State.SPEAKING and ev != Ev.TTS_DONE:
                self._purge_audio()
                self._emit({"type": "stop_audio"})
        # A cough should not cut off a reply, a goal, or a reply being thought about (360 ms of voice, not 120 ms).
        self.vad.strict = nxt in (State.SPEAKING, State.ACTING, State.THINKING)

        if nxt == State.NOTICING:
            self._sfx("notice")
            self.body.gesture("perk")
        elif nxt == State.ENGAGED and prev == State.DISENGAGING:  # you looked back before it fell asleep
            self._sfx("notice")
            self.body.gesture("perk")
        elif nxt == State.ENGAGED and prev in (State.IDLE, State.NOTICING):
            if time.monotonic() - self._last_greet > settings.greet_cooldown_s:
                self._last_greet = time.monotonic()
                self._spawn(self._greet())
        elif nxt == State.LISTENING:
            self._sfx("listen")
        elif nxt == State.THINKING:
            utt, self._pending = self._pending, None
            if utt is None:
                utt = self.vad.flush()
            self._typed = isinstance(utt, str)
            self._reply = self._spawn(self._think(utt))
        elif nxt == State.ENGAGED and prev == State.SPEAKING and self._pending_music:
            self._set_music(self._pending_music)
            self._pending_music = None
        elif nxt == State.IDLE:
            if prev == State.DISENGAGING:  # a glance that never engaged just settles back, no sleep sound
                self._sfx("sleep")
            self._set_music("stop")
        if prev == State.ACTING and ev == Ev.TIMEOUT:  # the goal ran out of time: say so with sound and motion
            self._sfx("error")
            self.body.gesture("shake")

        self._emit({"type": "state", "state": nxt.value, "prev": prev.value, "event": ev.value})

    def _sfx(self, name: str) -> None:
        self._emit({"type": "sfx", "name": name})
        self._quiet_until = max(self._quiet_until, time.monotonic() + SFX_GUARD_S.get(name, 0.5))

    def _cancel_reply(self) -> None:
        self._cancel(self._reply)

    @staticmethod
    def _cancel(task: asyncio.Task | None) -> None:
        if task and not task.done() and task is not asyncio.current_task():
            task.cancel()

    def _set_music(self, mood: str) -> None:
        if mood in MUSIC_BPM:
            self.body.dance_bpm = MUSIC_BPM[mood]
            self._emit({"type": "music", "action": "play", "mood": mood, "bpm": MUSIC_BPM[mood]})
        elif mood == "stop" and self.body.dance_bpm:
            self.body.dance_bpm = None
            self._emit({"type": "music", "action": "stop"})

    # ---------- mind ----------
    async def _greet(self) -> None:
        self._sfx("greet")
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
            if isinstance(utt, bytes) and utt and settings.one_call_turns and hasattr(self.p.llm, "respond_to_audio"):
                # One call hears and answers. The question is not known as text beforehand, so the most recent
                # objects stand in for a memory search.
                memories = await asyncio.to_thread(self.memory.recent)
                reply = await self.p.llm.respond_to_audio(audio=utt, sr=16000, history=self.history[-10:],
                                                          memories=memories, scene=self.scene_now,
                                                          mood=self.fsm.affect.label())
                text = str(reply.get("heard", ""))
                marks["stt_ms"], marks["llm_ms"] = 0.0, (time.perf_counter() - t0) * 1000
                if not real_words(text) or not str(reply.get("say", "")).strip():  # a cough, a noise: stay quiet
                    self.fsm.fire(Ev.ABORT)
                    return
                self._emit({"type": "transcript", "role": "user", "text": text})
            else:
                text = utt if isinstance(utt, str) else (await self.p.stt.transcribe(utt) if utt else "")
                marks["stt_ms"] = (time.perf_counter() - t0) * 1000
                if not real_words(text):
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
            action = reply.get("action") or {}
            if action.get("do") in ("spotlight", "look") and str(action.get("target", "")).strip():
                self._goal = {"do": action["do"], "target": str(action["target"]).strip()}
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
            self._sfx("error")
            self.fsm.fire(Ev.ABORT)

    async def _prefetch(self, lines: list[str]) -> None:
        """Synthesize fixed lines up front: the greeting voice then starts exactly 0.4 s after its chime and bounce
        instead of 0.4 s plus a variable TTS round trip. With a real TTS they are also cached on disk, so each line
        is synthesized once, not on every page load (free-tier TTS quotas are small)."""
        for text in lines:
            path = self._tts_disk_path(text)
            try:
                if path and os.path.exists(path):
                    pcm = open(path, "rb").read()
                    self._tts_cache[text] = [pcm[i:i + 4800] for i in range(0, len(pcm), 4800)]
                    continue
                self._tts_cache[text] = [c async for c in self.p.tts.stream(text)]
                if path:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "wb") as f:
                        f.write(b"".join(self._tts_cache[text]))
            except Exception:
                log.warning("could not prefetch %r; it will be synthesized or spoken by the browser", text)

    @staticmethod
    def _tts_disk_path(text: str) -> str | None:
        if settings.providers == "fake":
            return None
        voice = (settings.gemini_tts_model + settings.gemini_voice if settings.providers == "gemini"
                 else settings.tts_model + settings.tts_voice)
        key = hashlib.sha1(f"{settings.providers}|{voice}|{text}".encode()).hexdigest()[:16]
        return os.path.join(TTS_CACHE, f"{key}.pcm")

    async def _audio(self, text: str):
        if text in self._tts_cache:
            for chunk in self._tts_cache[text]:
                yield chunk
        elif time.monotonic() < self._tts_down_until:
            raise RuntimeError("TTS recently failed; skipping it for a while")
        else:
            async for chunk in self.p.tts.stream(text):
                yield chunk

    async def _speak(self, text: str, emotion: str, t0: float | None = None, marks: dict | None = None) -> None:
        self._emit({"type": "transcript", "role": "robot", "text": text, "emotion": emotion})
        self._emit({"type": "tts_start"})
        sent = False
        try:
            async for chunk in self._audio(text):
                if not sent:
                    self._first_audio(t0, marks)
                sent = True
                await self.out.put(pack(Frame.AUDIO_OUT, chunk))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if not sent:  # never go silent: the browser speaks the line and reports playback_done when it ends
                if time.monotonic() >= self._tts_down_until:
                    log.warning("TTS failed (%s); browser speech for the next %d s", e, TTS_BACKOFF_S)
                self._tts_down_until = max(self._tts_down_until, time.monotonic() + TTS_BACKOFF_S)
                self._first_audio(t0, marks)
                self._emit({"type": "speak", "text": text})
                return
            log.warning("TTS stream broke mid-line: %s", e)
        self._emit({"type": "tts_end"})  # client replies playback_done when the speaker is actually quiet

    def _first_audio(self, t0: float | None, marks: dict | None) -> None:
        if t0 is None or marks is None:
            return
        marks["first_audio_ms"] = (time.perf_counter() - t0) * 1000
        rounded = {k: round(v) for k, v in marks.items()}
        log.info("latency %s", rounded)
        self._trace("metrics", **rounded)
        self._emit({"type": "metrics", **rounded})

    # ---------- goals: local planner + executor ----------
    # The model chose only {do, target}. Finding the target, aiming, lighting, re-observing, verifying,
    # retrying, and the outcome all happen here, bounded by the ACTING timeout.
    def _goal_step(self, step: str, t0: float, **info) -> None:
        rec = {"step": step, "ms": round((time.perf_counter() - t0) * 1000), **info}
        self._trace("goal", **rec)
        self._emit({"type": "goal", **rec})

    async def _aim(self, obj: dict) -> None:
        self._look, self.body.focus = (float(obj["x"]), float(obj["y"])), obj["name"]
        await asyncio.sleep(0.15)  # let the 20 Hz body pick up the new target before polling
        deadline = time.monotonic() + AIM_TIMEOUT_S
        while not self.body.settled and time.monotonic() < deadline:
            await asyncio.sleep(0.05)

    async def _goal_outcome(self, text: str, emotion: str, gesture: str) -> None:
        self.fsm.affect.set_emotion(emotion)
        self.body.gesture(gesture)
        if self.fsm.fire(Ev.REPLY_READY):
            self._reply = asyncio.current_task()
            await self._speak(text, emotion)

    def _start_goal(self) -> None:
        """Right after the acknowledgement's TTS_DONE, with no await in between: a frame processed in a gap could
        otherwise disengage the lamp (the user is looking at the target, not the lamp) and drop the goal."""
        if self.fsm.state == State.ENGAGED and self._goal:
            goal, self._goal = self._goal, None
            if self.fsm.fire(Ev.GOAL):
                self._goal_task = self._spawn(self._run_goal(goal))

    async def _run_goal(self, goal: dict) -> None:
        do, want, t0 = goal["do"], goal["target"], time.perf_counter()
        self._goal_step("start", t0, do=do, target=want)
        try:
            obj = find_object(self.scene_now, want)
            if obj is None or "x" not in obj:
                self._goal_step("observe", t0, why="not in the current scene")
                obj = find_object(await self._observe(), want)
            if obj is None or "x" not in obj:
                self._goal_step("failed", t0, why="not visible")
                return await self._goal_outcome(f"I'm afraid I can't see your {want} at the moment.",
                                                "confused", "shake")
            for attempt in range(2):
                self._goal_step("aim", t0, target=obj["name"], x=obj["x"], y=obj["y"])
                await self._aim(obj)
                if do == "spotlight" and self.body.light_override is None:
                    self.body.light_override = (SPOT_RGB, 1.0)
                    self._sfx("spot")
                self._goal_step("observe", t0, why="verify")
                seen = find_object(await self._observe(), want)
                if seen is None or "x" not in seen:
                    self._goal_step("failed", t0, why="lost after aiming")
                    return await self._goal_outcome(f"Your {want} appears to have vanished.", "surprised", "shake")
                moved = math.hypot(seen["x"] - obj["x"], seen["y"] - obj["y"])
                self._goal_step("verify", t0, moved=round(moved, 3))
                if moved <= MOVE_TOL:
                    self._goal_step("done", t0, attempts=attempt + 1)
                    text = (f"Done. Your {seen['name']} is in the spotlight." if do == "spotlight"
                            else f"There's your {seen['name']}.")
                    text = ("It moved, so I followed it. " if attempt else "") + text
                    return await self._goal_outcome(text, "happy", "nod")
                obj = seen  # it moved: re-aim once at the new position
            self._goal_step("failed", t0, why="kept moving")
            await self._goal_outcome(f"Your {want} won't hold still, I'm afraid.", "confused", "tilt")
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("goal failed")
            self._sfx("error")
            self.fsm.affect.set_emotion("confused")
            self.fsm.fire(Ev.ABORT)

    # ---------- body clock ----------
    async def _ticker(self) -> None:
        dt, n = 0.05, 0
        while True:
            await asyncio.sleep(dt)
            self.fsm.tick(dt)
            look = self._look if self._look is not None else self._gaze
            frame = self.body.update(self.fsm.state, self.fsm.affect, look, dt)
            n += 1
            if n % 2 == 0:  # 10 Hz to the browser (its 3D lamp interpolates between frames), 20 Hz to the sim
                a = self.fsm.affect
                self._emit({"type": "body", **frame, "mood": a.label(),
                            "va": [round(a.valence, 2), round(a.arousal, 2)], "face": self._face})
