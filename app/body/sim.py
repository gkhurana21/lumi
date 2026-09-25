"""PyBullet visualizer in its own process (GUI must own its main thread, esp. on macOS)."""
from __future__ import annotations

import os
import queue
import re
import struct
import time

CACHE = "build/sim"
# Frames the whole lamp: about 0.8 m tall at rest, head reaching about 0.3 m forward.
CAMERA = dict(cameraDistance=1.5, cameraYaw=45, cameraPitch=-15, cameraTargetPosition=[0.05, 0, 0.38])


def _binary_stl(src: str, dst: str) -> None:
    """PyBullet only parses binary STL; the supplied shade is ASCII. Write a binary copy."""
    text = open(src).read()
    normals = re.findall(r"facet normal\s+(\S+)\s+(\S+)\s+(\S+)", text)
    verts = re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", text)
    with open(dst, "wb") as f:
        f.write(b"binary copy of " + os.path.basename(src).encode().ljust(65, b" "))
        f.write(struct.pack("<I", len(normals)))
        for i, n in enumerate(normals):
            vals = [float(x) for x in (*n, *verts[3 * i], *verts[3 * i + 1], *verts[3 * i + 2])]
            f.write(struct.pack("<12fH", *vals, 0))


def sim_urdf(urdf_path: str) -> str:
    """Return a URDF PyBullet can load: ASCII STL meshes swapped for binary copies under build/sim/."""
    urdf_path = os.path.abspath(urdf_path)
    base = os.path.dirname(urdf_path)
    text = open(urdf_path).read()
    os.makedirs(CACHE, exist_ok=True)

    def swap(m: re.Match) -> str:
        src = os.path.join(base, m.group(1))
        with open(src, "rb") as f:
            ascii_stl = f.read(5) == b"solid"
        if not ascii_stl:
            return f'filename="{src}"'
        dst = os.path.abspath(os.path.join(CACHE, os.path.basename(src)))
        _binary_stl(src, dst)
        return f'filename="{dst}"'

    text = re.sub(r'filename="([^"]+\.stl)"', swap, text, flags=re.IGNORECASE)
    out = os.path.join(CACHE, os.path.basename(urdf_path))
    with open(out, "w") as f:
        f.write(text)
    return out


def run_sim(q, urdf_path: str) -> None:
    import pybullet as p

    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setGravity(0, 0, 0)
    robot = p.loadURDF(sim_urdf(urdf_path), useFixedBase=True)
    p.resetDebugVisualizerCamera(**CAMERA)

    joints, shade = {}, -1
    for i in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, i)
        joints[info[1].decode()] = i
        if "shade" in info[12].decode().lower():
            shade = i
    if shade < 0:
        shade = p.getNumJoints(robot) - 1

    while True:
        frame = None
        try:
            while True:
                frame = q.get_nowait()  # keep only the latest
        except queue.Empty:
            pass
        if frame:
            for name, v in frame["joints"].items():
                if name in joints:
                    p.setJointMotorControl2(robot, joints[name], p.POSITION_CONTROL,
                                            targetPosition=v, force=50, maxVelocity=4.0)
            r, g, b = frame["light"]["rgb"]
            k = 0.3 + 0.7 * frame["light"]["brightness"]
            p.changeVisualShape(robot, shade, rgbaColor=[r * k, g * k, b * k, 1])
        p.stepSimulation()
        time.sleep(1 / 240)
