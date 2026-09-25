"""Tune the body without the server.

  python scripts/sim_poses.py              # GUI: cycle every state pose, then every gesture
  python scripts/sim_poses.py --sweep      # GUI: wiggle each joint alone to learn axes and signs
  python scripts/sim_poses.py --render out # headless: save one PNG per state/gesture to out/ (for agents)
"""
import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pybullet as p  # noqa: E402

from app.body import behaviors  # noqa: E402
from app.body.behaviors import GESTURES, Body  # noqa: E402
from app.core.fsm import Affect, State  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--urdf", default="robot/dummy_lamp_6dof.urdf")
ap.add_argument("--sweep", action="store_true")
ap.add_argument("--render", metavar="OUTDIR")
args = ap.parse_args()

urdf = os.path.abspath(args.urdf)
p.connect(p.DIRECT if args.render else p.GUI)
p.setGravity(0, 0, 0)
p.setAdditionalSearchPath(os.path.dirname(urdf))
robot = p.loadURDF(urdf, useFixedBase=True)
idx = {p.getJointInfo(robot, i)[1].decode(): i for i in range(p.getNumJoints(robot))}


class SimClock:
    """Body reads time.monotonic(); headless renders don't sleep, so advance a fake clock instead."""

    def __init__(self):
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t


clock = SimClock()
if args.render:
    behaviors.time = clock
body = Body(urdf)
CAM = dict(cameraDistance=0.9, cameraYaw=45, cameraPitch=-20, cameraTargetPosition=[0, 0, 0.25])
VIEWS = (45, 0)  # three-quarter and side view, side by side
if not args.render:
    p.resetDebugVisualizerCamera(**CAM)


def apply(frame):
    for name, v in frame["joints"].items():
        if name in idx:
            p.resetJointState(robot, idx[name], v)


def snap(outdir, label):
    import cv2
    import numpy as np
    proj = p.computeProjectionMatrixFOV(50, 4 / 3, 0.01, 10)
    panels = []
    for yaw in VIEWS:
        view = p.computeViewMatrixFromYawPitchRoll(CAM["cameraTargetPosition"], CAM["cameraDistance"],
                                                   yaw, CAM["cameraPitch"], 0, 2)
        _, _, rgba, _, _ = p.getCameraImage(640, 480, view, proj, renderer=p.ER_TINY_RENDERER)
        # rgba is a flat tuple when pybullet is built without numpy; make a contiguous uint8 BGR image.
        rgb = np.asarray(rgba, dtype=np.uint8).reshape(480, 640, 4)[:, :, :3]
        panel = np.ascontiguousarray(rgb[:, :, ::-1])
        cv2.putText(panel, f"yaw {yaw}", (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
        panels.append(panel)
    img = np.ascontiguousarray(np.hstack(panels))
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.imwrite(os.path.join(outdir, f"{label}.png"), img)


def step(state):
    frame = body.update(state, Affect(), (0.0, 0.0), 0.05)
    clock.t += 0.05
    if not args.render:
        apply(frame)
        p.stepSimulation()
        time.sleep(0.05)
    return frame


def settle(state, seconds):
    body.pos = {r: 0.0 for r in body.map}
    for _ in range(int(seconds * 20)):
        frame = step(state)
    apply(frame)
    return frame


def gesture_peak(name):
    """Settle ENGAGED, play the gesture, and return the frame where its joint deviates most."""
    role, _, _, dur = GESTURES[name]
    base = settle(State.ENGAGED, 3.0)
    joint = body.map[role][0] if role in body.map else None
    body.gesture(name)
    best, best_d = base, -1.0
    for _ in range(int(dur * 20) + 1):
        frame = step(State.ENGAGED)
        d = abs(frame["joints"].get(joint, 0.0) - base["joints"].get(joint, 0.0))
        if d > best_d:
            best, best_d = frame, d
    apply(best)
    return best, best_d


if args.sweep:
    for name, i in idx.items():
        if p.getJointInfo(robot, i)[2] == p.JOINT_FIXED:
            continue
        print("sweeping", name)
        for k in range(90):
            p.resetJointState(robot, i, 0.6 * math.sin(2 * math.pi * k / 45))
            time.sleep(1 / 30)
        p.resetJointState(robot, i, 0)
    sys.exit()

if args.render:
    os.makedirs(args.render, exist_ok=True)
for st in State:
    f = settle(st, 3.0)
    print(f"{st.value:12s} {f['joints']}")
    if args.render:
        snap(args.render, f"state_{st.value}")
for g in GESTURES:
    f, d = gesture_peak(g)
    print(f"gesture {g:8s} peak deviation {d:.3f} rad  {f['joints']}")
    if args.render:
        snap(args.render, f"gesture_{g}")
