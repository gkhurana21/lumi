"""State + affect + gaze -> joint targets and lamp light. Pure procedural animation, 20 Hz."""
from __future__ import annotations

import math
import os
import queue
import time
import xml.etree.ElementTree as ET

from ..core.fsm import Affect, State

# Semantic roles, bound in order to the first movable URDF joints.
# VERIFY against dummy_lamp_5dof.urdf (joint order + sign conventions) and reorder if needed.
ROLES = ("base_yaw", "shoulder", "elbow", "head_pitch", "head_roll")

# Radians, relative to URDF zero. Convention here: +head_pitch = droop down. Flip signs after first sim run.
POSES: dict[State, dict[str, float]] = {
    State.IDLE: dict(shoulder=-0.5, elbow=1.1, head_pitch=0.9),
    State.NOTICING: dict(shoulder=0.1, elbow=0.6, head_pitch=0.1),
    State.ENGAGED: dict(shoulder=0.2, elbow=0.5, head_pitch=0.0),
    State.LISTENING: dict(shoulder=0.3, elbow=0.4, head_pitch=-0.1, head_roll=0.15),
    State.THINKING: dict(shoulder=0.2, elbow=0.5, head_pitch=-0.35, head_roll=-0.2, base_yaw=0.25),
    State.SPEAKING: dict(shoulder=0.25, elbow=0.45, head_pitch=0.0),
    State.DISENGAGING: dict(shoulder=-0.2, elbow=0.8, head_pitch=0.4),
}
LOOKING = {State.NOTICING, State.ENGAGED, State.LISTENING, State.SPEAKING}

# name -> (role, amplitude, freq Hz (0 = hold), duration s)
GESTURES = {
    "nod": ("head_pitch", 0.25, 2.5, 1.0),
    "shake": ("base_yaw", 0.30, 3.0, 1.0),
    "tilt": ("head_roll", 0.35, 0.0, 1.4),
    "bounce": ("shoulder", 0.12, 4.0, 0.8),
    "droop": ("head_pitch", 0.50, 0.0, 1.6),
    "perk": ("shoulder", 0.25, 0.0, 0.8),
}

WARM, COOL = (1.0, 0.72, 0.42), (0.55, 0.75, 1.0)


def load_joints(urdf_path: str) -> list[tuple[str, float, float]]:
    out = []
    for j in ET.parse(urdf_path).getroot().findall("joint"):
        if j.get("type") not in ("revolute", "continuous"):
            continue
        lim = j.find("limit")
        lo = float(lim.get("lower", -math.pi)) if lim is not None else -math.pi
        hi = float(lim.get("upper", math.pi)) if lim is not None else math.pi
        if j.get("type") == "continuous" or lo >= hi:
            lo, hi = -math.pi, math.pi
        out.append((j.get("name"), lo, hi))
    return out


class Body:
    def __init__(self, urdf_path: str, sim_q=None, smoothing: float = 6.0):
        joints = load_joints(urdf_path) if os.path.exists(urdf_path) else [(r, -math.pi, math.pi) for r in ROLES]
        self.map = {role: j for role, j in zip(ROLES, joints, strict=False)}  # role -> (urdf_name, lo, hi)
        self.pos = {r: 0.0 for r in self.map}
        self.sim_q, self.k = sim_q, smoothing
        self.t0 = time.monotonic()
        self._gest: tuple[str, float] | None = None
        self.dance_bpm: float | None = None  # set while music plays

    def gesture(self, name: str) -> None:
        if name in GESTURES:
            self._gest = (name, time.monotonic())

    def update(self, state: State, affect: Affect, gaze: tuple[float, float], dt: float) -> dict:
        now = time.monotonic()
        t = now - self.t0
        ar = (affect.arousal + 1) / 2  # 0..1
        tgt = {r: POSES[state].get(r, 0.0) for r in self.map}

        def add(role: str, v: float) -> None:
            if role in tgt:
                tgt[role] += v

        if state in LOOKING:
            add("base_yaw", -0.6 * gaze[0])
            add("head_pitch", 0.4 * gaze[1])
        add("shoulder", (0.02 + 0.05 * ar) * math.sin(2 * math.pi * (0.15 + 0.35 * ar) * t))  # breathing
        add("head_pitch", 0.3 * max(0.0, -affect.valence))  # sadness droops
        if self.dance_bpm and state not in (State.LISTENING, State.THINKING):
            beat = 2 * math.pi * self.dance_bpm / 60 * t
            add("shoulder", 0.08 * abs(math.sin(beat / 2)))
            add("base_yaw", 0.18 * math.sin(beat / 4))
            add("head_roll", 0.12 * math.sin(beat / 2))
        if state == State.SPEAKING:
            add("head_pitch", 0.04 * math.sin(2 * math.pi * 3.2 * t))
        if self._gest:
            name, st = self._gest
            role, amp, f, dur = GESTURES[name]
            e = now - st
            if e > dur:
                self._gest = None
            else:
                env = math.sin(math.pi * e / dur)
                add(role, amp * env * (math.sin(2 * math.pi * f * e) if f else 1.0))

        a = 1 - math.exp(-self.k * (0.6 + 0.8 * ar) * dt)  # excited = snappier
        for r, (_name, lo, hi) in self.map.items():
            self.pos[r] = max(lo, min(hi, self.pos[r] + (tgt[r] - self.pos[r]) * a))

        v01 = (affect.valence + 1) / 2
        rgb = [round(c + (w - c) * v01, 3) for c, w in zip(COOL, WARM, strict=True)]
        bright = 0.15 if state == State.IDLE else 0.5 + 0.4 * ar
        if state == State.THINKING:
            bright *= 0.8 + 0.2 * math.sin(2 * math.pi * 1.5 * t)
        if self.dance_bpm:  # color cycles on the beat
            ph = 2 * math.pi * self.dance_bpm / 60 * t / 4
            rgb = [round(0.6 + 0.4 * math.sin(ph + o), 3) for o in (0, 2.1, 4.2)]

        frame = {"joints": {self.map[r][0]: round(v, 4) for r, v in self.pos.items()},
                 "light": {"rgb": rgb, "brightness": round(bright, 3)}}
        if self.sim_q is not None:
            try:
                self.sim_q.put_nowait(frame)
            except queue.Full:
                pass
        return frame
