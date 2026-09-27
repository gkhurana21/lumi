"""PyBullet visualizer in its own process (GUI must own its main thread, esp. on macOS)."""
from __future__ import annotations

import math
import os
import queue
import re
import struct
import time

CACHE = "build/sim"
# Front three-quarter view of the lamp (it faces world -x, toward the user; about 0.8 m tall) and the desk in front
# of it, where goal spotlights land.
CAMERA = dict(cameraDistance=1.3, cameraYaw=-55, cameraPitch=-22, cameraTargetPosition=[-0.3, 0.05, 0.28])
BACKGROUND = [0.07, 0.07, 0.09]  # a dim room, so the lamp's light reads as light


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


def link_names(p, robot) -> list[str]:
    return [p.getJointInfo(robot, i)[12].decode() for i in range(p.getNumJoints(robot))]


def light_link(names: list[str]) -> int:
    """Link to tint with the lamp light: the URDF's emitter ("light" in the name), else a "shade" link, else last."""
    for key in ("light", "emitter", "shade"):
        for i, n in enumerate(names):
            if key in n.lower():
                return i
    return len(names) - 1


def show_light(p, robot, link: int, light: dict) -> None:
    r, g, b = light["rgb"]
    k = 0.3 + 0.7 * light["brightness"]
    p.changeVisualShape(robot, link, rgbaColor=[r * k, g * k, b * k, 1])


def _cone_obj(path: str, length: float, half_angle: float, n: int = 24) -> None:
    """Open cone mesh, apex at the origin, axis +x. Faces in both windings so it shows from inside and out."""
    r = length * math.tan(half_angle)
    lines = ["v 0 0 0"] + [f"v {length} {r * math.cos(2 * math.pi * k / n):.5f} {r * math.sin(2 * math.pi * k / n):.5f}"
                           for k in range(n)]
    for k in range(n):
        a, b = 2 + k, 2 + (k + 1) % n
        lines += [f"f 1 {a} {b}", f"f 1 {b} {a}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


class LightViz:
    """Makes the lamp's light visible: a short glow when the head faces forward, and a spotlight cone that lands on a
    lit pool when it points down at the desk, with the goal target's name above it. Pure rendering, no behavior."""

    GLOW = (0.3, 0.4)  # length m, half-angle rad
    SPOT_LENGTHS, SPOT_HALF = (0.35, 0.6, 0.85, 1.1), 0.2  # the smallest one that reaches the desk is shown
    HIDDEN = [0, 0, -10]

    def __init__(self, p, robot: int, link: int, beam: bool = True):
        self.p, self.robot, self.link, self.show_beam = p, robot, link, beam
        desk = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.9, 0.75, 0.01], rgbaColor=[0.42, 0.31, 0.22, 1])
        p.createMultiBody(0, -1, desk, basePosition=[-0.45, 0, -0.01])
        os.makedirs(CACHE, exist_ok=True)
        self.glow = self._cone(*self.GLOW)
        self.spots = [self._cone(length, self.SPOT_HALF) for length in self.SPOT_LENGTHS]
        self.pools = [p.createMultiBody(0, -1, p.createVisualShape(
            p.GEOM_CYLINDER, radius=length * math.tan(self.SPOT_HALF), length=0.002, rgbaColor=[1, 1, 1, 0.3]),
            basePosition=self.HIDDEN) for length in self.SPOT_LENGTHS]
        self.label = -1

    def _cone(self, length: float, half: float) -> int:
        path = os.path.abspath(os.path.join(CACHE, f"light_cone_{length:.2f}_{half:.2f}.obj"))
        _cone_obj(path, length, half)
        shape = self.p.createVisualShape(self.p.GEOM_MESH, fileName=path, rgbaColor=[1, 1, 1, 0.1])
        return self.p.createMultiBody(0, -1, shape, basePosition=self.HIDDEN)

    def _place(self, body: int, pos, orn, rgba) -> None:
        self.p.resetBasePositionAndOrientation(body, pos, orn)
        self.p.changeVisualShape(body, -1, rgbaColor=rgba)

    def update(self, light: dict, focus: str | None = None) -> None:
        p, ident = self.p, [0, 0, 0, 1]
        pos, orn = p.getLinkState(self.robot, self.link, computeForwardKinematics=True)[4:6]
        k = light["brightness"]
        raw = list(light["rgb"])  # beams: raw color, brightness in alpha (blending over a dark room)
        r, g, b = (c * (0.3 + 0.7 * k) for c in light["rgb"])  # pools get the emitter's dimming
        fwd = p.getMatrixFromQuaternion(orn)[0::3]  # head forward axis (+x of the emitter frame)
        reach = -pos[2] / fwd[2] if fwd[2] < -0.05 else math.inf
        pick = next((i for i, length in enumerate(self.SPOT_LENGTHS) if length >= reach), None)
        for i, (spot, pool) in enumerate(zip(self.spots, self.pools, strict=True)):
            on = i == pick
            self._place(spot, pos if on and self.show_beam else self.HIDDEN, orn, [*raw, 0.08 + 0.3 * k])
            hit = [pos[0] + reach * fwd[0], pos[1] + reach * fwd[1], 0.002] if on else self.HIDDEN
            self._place(pool, hit, ident, [r, g, b, 0.2 + 0.6 * k])
        self._place(self.glow, pos if pick is None and self.show_beam else self.HIDDEN, orn,
                    [*raw, 0.1 + 0.45 * k])
        if focus and pick is not None:
            at = [pos[0] + reach * fwd[0], pos[1] + reach * fwd[1], 0.12]
            self.label = p.addUserDebugText(focus, at, textColorRGB=[1, 1, 1], textSize=1.6,
                                            replaceItemUniqueId=self.label if self.label >= 0 else -1)
        elif self.label >= 0:
            p.removeUserDebugItem(self.label)
            self.label = -1


def run_sim(q, urdf_path: str) -> None:
    import pybullet as p

    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, rgbBackground=BACKGROUND)
    p.setGravity(0, 0, 0)
    robot = p.loadURDF(sim_urdf(urdf_path), useFixedBase=True)
    p.resetDebugVisualizerCamera(**CAMERA)

    joints = {}  # name -> (index, max force, max velocity) from the URDF
    for i in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, i)
        joints[info[1].decode()] = (i, info[10], info[11])
    light = light_link(link_names(p, robot))
    viz = LightViz(p, robot, light)

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
                    i, force, vmax = joints[name]
                    p.setJointMotorControl2(robot, i, p.POSITION_CONTROL,
                                            targetPosition=v, force=force, maxVelocity=vmax)
            show_light(p, robot, light, frame["light"])
            viz.update(frame["light"], frame.get("focus"))
        p.stepSimulation()
        time.sleep(1 / 240)
