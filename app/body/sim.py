"""PyBullet visualizer in its own process (GUI must own its main thread, esp. on macOS)."""
from __future__ import annotations

import os
import queue
import time


def run_sim(q, urdf_path: str) -> None:
    import pybullet as p

    urdf_path = os.path.abspath(urdf_path)
    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setGravity(0, 0, 0)
    p.setAdditionalSearchPath(os.path.dirname(urdf_path))  # so lamp_shade.stl resolves
    robot = p.loadURDF(urdf_path, useFixedBase=True)
    p.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=45, cameraPitch=-25, cameraTargetPosition=[0, 0, 0.25])

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
