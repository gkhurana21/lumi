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
body = Body(urdf)
CAM = dict(cameraDistance=0.9, cameraYaw=45, cameraPitch=-20, cameraTargetPosition=[0, 0, 0.25])
if not args.render:
    p.resetDebugVisualizerCamera(**CAM)


def apply(frame):
    for name, v in frame["joints"].items():
        if name in idx:
            p.resetJointState(robot, idx[name], v)


def snap(outdir, label):
    import cv2
    import numpy as np
    view = p.computeViewMatrixFromYawPitchRoll(CAM["cameraTargetPosition"], CAM["cameraDistance"],
                                               CAM["cameraYaw"], CAM["cameraPitch"], 0, 2)
    proj = p.computeProjectionMatrixFOV(50, 4 / 3, 0.01, 10)
    _, _, rgba, _, _ = p.getCameraImage(640, 480, view, proj, renderer=p.ER_TINY_RENDERER)
    img = np.reshape(rgba, (480, 640, 4))[:, :, :3][:, :, ::-1]
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.imwrite(os.path.join(outdir, f"{label}.png"), img)


def settle(state, seconds, gesture=None):
    body.pos = {r: 0.0 for r in body.map}
    if gesture:
        body.gesture(gesture)
    frame = None
    for _ in range(int(seconds * 20)):
        frame = body.update(state, Affect(), (0.0, 0.0), 0.05)
        if not args.render:
            apply(frame)
            p.stepSimulation()
            time.sleep(0.05)
    apply(frame)
    return frame


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
for g, (_, _, _, dur) in GESTURES.items():
    settle(State.ENGAGED, dur / 2, gesture=g)  # render at the gesture's peak
    print("gesture", g)
    if args.render:
        snap(args.render, f"gesture_{g}")
