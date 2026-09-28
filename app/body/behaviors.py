"""State + affect + gaze -> joint targets and lamp light. Pure procedural animation, 20 Hz."""
from __future__ import annotations

import math
import os
import queue
import time
import xml.etree.ElementTree as ET

from ..core.fsm import Affect, State

# Semantic roles, bound in order to the movable joints of dummy_lamp_5dof.urdf:
# base_yaw_joint, shoulder_pitch_joint, elbow_pitch_joint, neck_yaw_joint, head_pitch_joint.
ROLES = ("base_yaw", "shoulder", "elbow", "neck_yaw", "head_pitch")

# Radians, relative to URDF zero (arm straight up, head facing world -x, toward the user).
# Signs, verified with forward kinematics:
#   shoulder/elbow +  arm leans back, head tips up     (negative leans toward the user)
#   base/neck_yaw  +  turns to the lamp's own left     (negative turns right)
#   head_pitch     +  head looks down
# Head elevation = shoulder + elbow - head_pitch, so level gaze needs head_pitch = shoulder + elbow.
POSES: dict[State, dict[str, float]] = {
    State.IDLE: dict(shoulder=0.25, elbow=-1.60, head_pitch=-0.45),  # arm folded forward, head hangs toward the desk
    State.NOTICING: dict(shoulder=0.05, elbow=-0.45, head_pitch=-0.55),  # tall and alert, head slightly up
    State.ENGAGED: dict(shoulder=0.15, elbow=-0.70, head_pitch=-0.60),  # upright, head level toward the user
    # No roll joint: neck yaw on the forward-leaning upper arm rolls the head; opposite base yaw cancels the turn.
    # FK: 18 deg roll, -0.5 deg yaw (still facing the user).
    State.LISTENING: dict(shoulder=-0.15, elbow=-0.55, base_yaw=-0.40, neck_yaw=0.50, head_pitch=-0.65),
    State.THINKING: dict(shoulder=0.20, elbow=-0.45, base_yaw=-0.35, neck_yaw=-0.25, head_pitch=-0.75),  # up, away
    State.SPEAKING: dict(shoulder=0.10, elbow=-0.70, head_pitch=-0.65),
    State.DISENGAGING: dict(shoulder=0.20, elbow=-1.15, head_pitch=-0.35),  # sagging, head lowering
    State.ACTING: dict(shoulder=-0.30, elbow=-0.90, head_pitch=-0.60),  # leans over the desk, head 0.6 rad down
}
# States whose head follows a look point: the user's face, or the goal target while ACTING.
LOOKING = {State.NOTICING, State.ENGAGED, State.LISTENING, State.SPEAKING, State.ACTING}
SETTLED_TOL = 0.08  # rad; above the breathing overlay (<= 0.06), so "arrived" means the pose, not the sway

# name -> ({role: amplitude}, shape, freq Hz, duration s), all under a sin(pi*t/dur) envelope.
# Shapes: "osc" swings +-A (peak speed 2*pi*f*A), "dip" goes 0..A and back (pi*f*A), "hold" rises to A once (pi*A/dur).
# Peak speeds stay under the joint's URDF velocity limit; Body also rate-limits every command.
GESTURES: dict[str, tuple[dict[str, float], str, float, float]] = {
    "nod": ({"head_pitch": 0.30}, "dip", 1.1, 1.8),  # two dips down; 1.04 of 1.45 rad/s
    "shake": ({"neck_yaw": 0.26}, "osc", 0.85, 1.6),  # 1.39 of 1.60 rad/s
    "tilt": ({"neck_yaw": 0.50, "base_yaw": -0.40}, "hold", 0.0, 1.4),  # head cock, as in LISTENING (15 deg roll)
    "bounce": ({"elbow": 0.12, "head_pitch": 0.12}, "osc", 1.4, 1.2),  # head bobs, stays level; 1.06 of 1.15 rad/s
    "droop": ({"head_pitch": 0.45, "shoulder": 0.10}, "hold", 0.0, 1.6),
    "perk": ({"elbow": 0.25, "head_pitch": -0.10}, "hold", 0.0, 1.0),  # 0.79 of 1.15 rad/s
}
GAZE_YAW = {"base_yaw": -0.25, "neck_yaw": -0.30}  # rad per unit face offset; sums to ~half a laptop camera's FOV
GAZE_PITCH = 0.35

WARM, COOL = (1.0, 0.72, 0.42), (0.55, 0.75, 1.0)


def load_joints(urdf_path: str) -> list[tuple[str, float, float, float]]:
    """Movable joints in file order as (name, lower, upper, max velocity), preferring the soft safety limits."""
    out = []
    for j in ET.parse(urdf_path).getroot().findall("joint"):
        if j.get("type") not in ("revolute", "continuous"):
            continue
        lim, soft = j.find("limit"), j.find("safety_controller")
        lo = float(lim.get("lower", -math.pi)) if lim is not None else -math.pi
        hi = float(lim.get("upper", math.pi)) if lim is not None else math.pi
        vmax = float(lim.get("velocity", 2.0)) if lim is not None else 2.0
        if soft is not None:
            lo = max(lo, float(soft.get("soft_lower_limit", lo)))
            hi = min(hi, float(soft.get("soft_upper_limit", hi)))
        if j.get("type") == "continuous" or lo >= hi:
            lo, hi = -math.pi, math.pi
        out.append((j.get("name"), lo, hi, vmax))
    return out


class Body:
    def __init__(self, urdf_path: str, sim_q=None, smoothing: float = 6.0):
        joints = (load_joints(urdf_path) if os.path.exists(urdf_path)
                  else [(r, -math.pi, math.pi, 2.0) for r in ROLES])
        self.map = {role: j for role, j in zip(ROLES, joints, strict=False)}  # role -> (urdf_name, lo, hi, vmax)
        self.pos = {r: 0.0 for r in self.map}  # smoothed pose, before overlays
        self.out = {r: 0.0 for r in self.map}  # last emitted command, after overlays and limits
        self.sim_q, self.k = sim_q, smoothing
        self.t0 = time.monotonic()
        self._gest: tuple[str, float] | None = None
        self.dance_bpm: float | None = None  # set while music plays
        self.light_override: tuple[list[float], float] | None = None  # (rgb, brightness), e.g. a goal spotlight
        self.focus: str | None = None  # name of what the lamp is aiming at during a goal (display only)
        self._want = {r: 0.0 for r in self.map}  # clamped pose target from the last update

    @property
    def settled(self) -> bool:
        """True once every joint command is within SETTLED_TOL of its pose target (motion is velocity-limited)."""
        return all(abs(self.out[r] - self._want[r]) < SETTLED_TOL for r in self.map)

    def gesture(self, name: str) -> None:
        if name in GESTURES:
            self._gest = (name, time.monotonic())

    def _overlay(self, state: State, ar: float, now: float) -> dict[str, float]:
        """Designed trajectories (breathing, dance, speech bob, gestures) added after smoothing, so they keep shape."""
        t = now - self.t0
        o = {r: 0.0 for r in self.map}

        def add(role: str, v: float) -> None:
            if role in o:
                o[role] += v

        add("shoulder", (0.02 + 0.04 * ar) * math.sin(2 * math.pi * (0.15 + 0.35 * ar) * t))  # breathing
        if self.dance_bpm and state not in (State.LISTENING, State.THINKING):
            beat = 2 * math.pi * self.dance_bpm / 60 * t
            add("shoulder", 0.08 * abs(math.sin(beat / 2)))
            add("base_yaw", 0.18 * math.sin(beat / 4))
            add("neck_yaw", 0.12 * math.sin(beat / 2))
        if state == State.SPEAKING:
            add("head_pitch", 0.04 * math.sin(2 * math.pi * 3.2 * t))
        if self._gest:
            name, st = self._gest
            amps, shape, f, dur = GESTURES[name]
            e = now - st
            if e > dur:
                self._gest = None
            else:
                env = math.sin(math.pi * e / dur)
                wave = {"osc": math.sin(2 * math.pi * f * e),
                        "dip": (1 - math.cos(2 * math.pi * f * e)) / 2}.get(shape, 1.0)
                for role, amp in amps.items():
                    add(role, amp * env * wave)
        return o

    def update(self, state: State, affect: Affect, gaze: tuple[float, float], dt: float) -> dict:
        now = time.monotonic()
        t = now - self.t0
        ar = (affect.arousal + 1) / 2  # 0..1
        # While a goal target is in focus the lamp keeps leaning over it, even while it speaks the outcome line.
        pose = POSES[State.ACTING] if self.focus and state == State.SPEAKING else POSES[state]
        tgt = {r: pose.get(r, 0.0) for r in self.map}
        if state in LOOKING:
            for role, k in GAZE_YAW.items():
                if role in tgt:
                    tgt[role] += k * gaze[0]
            if "head_pitch" in tgt:
                tgt["head_pitch"] += GAZE_PITCH * gaze[1]
        if "head_pitch" in tgt:
            tgt["head_pitch"] += 0.3 * max(0.0, -affect.valence)  # sadness droops

        a = 1 - math.exp(-self.k * (0.6 + 0.8 * ar) * dt)  # excited = snappier
        overlay = self._overlay(state, ar, now)
        for r, (_name, lo, hi, vmax) in self.map.items():
            self._want[r] = max(lo, min(hi, tgt[r]))
            self.pos[r] = max(lo, min(hi, self.pos[r] + (tgt[r] - self.pos[r]) * a))
            want = max(lo, min(hi, self.pos[r] + overlay[r]))
            step = 0.95 * vmax * dt  # never command faster than the joint can move
            self.out[r] = self.out[r] + max(-step, min(step, want - self.out[r]))

        v01 = (affect.valence + 1) / 2
        rgb = [round(c + (w - c) * v01, 3) for c, w in zip(COOL, WARM, strict=True)]
        bright = 0.15 if state == State.IDLE else 0.5 + 0.4 * ar
        if state == State.THINKING:
            bright *= 0.8 + 0.2 * math.sin(2 * math.pi * 1.5 * t)
        if self.dance_bpm:  # color cycles on the beat
            ph = 2 * math.pi * self.dance_bpm / 60 * t / 4
            rgb = [round(0.6 + 0.4 * math.sin(ph + o), 3) for o in (0, 2.1, 4.2)]
        if self.light_override:
            rgb, bright = list(self.light_override[0]), self.light_override[1]

        frame = {"joints": {self.map[r][0]: round(v, 4) for r, v in self.out.items()},
                 "light": {"rgb": rgb, "brightness": round(bright, 3)}}
        if self.focus:
            frame["focus"] = self.focus
        if self.sim_q is not None:
            try:
                self.sim_q.put_nowait(frame)
            except queue.Full:
                pass
        return frame
