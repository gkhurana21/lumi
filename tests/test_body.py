from app.body.behaviors import Body
from app.core.fsm import Affect, State


def test_body_respects_limits_and_emits_light():
    b = Body("missing.urdf")
    b.gesture("shake")
    for _ in range(40):
        f = b.update(State.ENGAGED, Affect(), (1.0, -1.0), 0.05)
    assert set(f) == {"joints", "light"} and len(f["joints"]) == 5
    assert all(-3.1416 <= v <= 3.1416 for v in f["joints"].values())


def test_sim_urdf_loads_supplied_mesh(tmp_path, monkeypatch):
    import struct

    import pybullet as p

    from app.body import sim

    monkeypatch.setattr(sim, "CACHE", str(tmp_path))
    urdf = sim.sim_urdf("robot/dummy_lamp_5dof.urdf")
    stl = tmp_path / "lamp_shade.stl"
    facets = open("robot/assets/lamp_shade.stl").read().count("facet normal")
    (count,) = struct.unpack("<I", stl.read_bytes()[80:84])
    assert count == facets > 0 and stl.stat().st_size == 84 + 50 * facets
    assert str(stl) in open(urdf).read()

    cid = p.connect(p.DIRECT)
    try:
        # The raw ASCII file crashes PyBullet here (segfault); the binary copy must load with the mesh's extent.
        shape = p.createCollisionShape(p.GEOM_MESH, fileName=str(stl), physicsClientId=cid)
        lo, hi = p.getAABB(p.createMultiBody(0, shape, physicsClientId=cid), physicsClientId=cid)
        assert abs(hi[0] - 0.12) < 0.01 and abs(lo[0] + 0.03) < 0.01 and abs(hi[2] - 0.108) < 0.01
        assert p.loadURDF(urdf, useFixedBase=True, physicsClientId=cid) >= 0
    finally:
        p.disconnect(cid)


URDF = "robot/dummy_lamp_5dof.urdf"


def _elevation(pose):  # head elevation in rad, + = up (see behaviors.POSES comment)
    return pose.get("shoulder", 0) + pose.get("elbow", 0) - pose.get("head_pitch", 0)


def test_roles_bind_to_supplied_joints():
    b = Body(URDF)
    assert {r: j[0] for r, j in b.map.items()} == {
        "base_yaw": "base_yaw_joint", "shoulder": "shoulder_pitch_joint", "elbow": "elbow_pitch_joint",
        "neck_yaw": "neck_yaw_joint", "head_pitch": "head_pitch_joint"}


def test_pose_intent():
    from app.body.behaviors import POSES
    assert _elevation(POSES[State.IDLE]) < -0.6  # slumped, facing the desk
    for s in (State.ENGAGED, State.SPEAKING, State.LISTENING):
        assert abs(_elevation(POSES[s])) < 0.1  # level, toward the user
    assert _elevation(POSES[State.THINKING]) > 0.3  # looks up
    assert POSES[State.THINKING].get("base_yaw", 0) != 0  # and away
    assert -0.8 < _elevation(POSES[State.ACTING]) < -0.4  # leans over the desk to aim at objects


def test_commands_respect_soft_and_velocity_limits():
    from app.body import behaviors
    from app.body.behaviors import GESTURES

    clock = type("Clock", (), {"t": 0.0, "monotonic": lambda self: self.t})()
    orig, behaviors.time = behaviors.time, clock
    try:
        b = Body(URDF)
        b.dance_bpm = 120
        prev = None
        for i, st in enumerate([*State, *State]):
            b.gesture(list(GESTURES)[i % len(GESTURES)])
            for _ in range(30):
                f = b.update(st, Affect(), (1.0, 1.0), 0.05)
                clock.t += 0.05
                for role, (name, lo, hi, vmax) in b.map.items():
                    v = f["joints"][name]
                    assert lo - 1e-6 <= v <= hi + 1e-6, (st, role, v)
                    if prev:
                        assert abs(v - prev[name]) <= vmax * 0.05 + 1e-3, (st, role)
                prev = f["joints"]
    finally:
        behaviors.time = orig


def test_gestures_reach_most_of_their_amplitude():
    from app.body import behaviors
    from app.body.behaviors import GESTURES

    clock = type("Clock", (), {"t": 0.0, "monotonic": lambda self: self.t})()
    orig, behaviors.time = behaviors.time, clock
    try:
        for name, (amps, _, _, dur) in GESTURES.items():
            b = Body(URDF)
            for _ in range(60):  # settle ENGAGED
                base = b.update(State.ENGAGED, Affect(), (0.0, 0.0), 0.05)
                clock.t += 0.05
            role = max(amps, key=lambda r: abs(amps[r]))
            joint = b.map[role][0]
            b.gesture(name)
            peak = 0.0
            for _ in range(int(dur / 0.05) + 1):
                f = b.update(State.ENGAGED, Affect(), (0.0, 0.0), 0.05)
                clock.t += 0.05
                peak = max(peak, abs(f["joints"][joint] - base["joints"][joint]))
            assert peak >= 0.65 * abs(amps[role]), (name, peak)
    finally:
        behaviors.time = orig


def test_light_link_is_the_emitter():
    import pybullet as p

    from app.body.sim import light_link, sim_urdf

    cid = p.connect(p.DIRECT)
    try:
        robot = p.loadURDF(sim_urdf(URDF), useFixedBase=True, physicsClientId=cid)
        names = [p.getJointInfo(robot, i, physicsClientId=cid)[12].decode()
                 for i in range(p.getNumJoints(robot, physicsClientId=cid))]
        assert names[light_link(names)] == "light_emitter_link"
    finally:
        p.disconnect(cid)
    assert light_link(["arm", "lamp_shade"]) == 1 and light_link(["a", "b"]) == 1


def test_goal_focus_keeps_the_lean_while_speaking():
    from app.body import behaviors
    from app.body.behaviors import POSES

    clock = type("Clock", (), {"t": 0.0, "monotonic": lambda self: self.t})()
    orig, behaviors.time = behaviors.time, clock
    try:
        b = Body(URDF)
        b.focus = "mug"
        for _ in range(80):
            f = b.update(State.SPEAKING, Affect(), (-0.5, 0.6), 0.05)
            clock.t += 0.05
        assert abs(f["joints"]["elbow_pitch_joint"] - POSES[State.ACTING]["elbow"]) < 0.05
        b.focus = None
        for _ in range(80):
            f = b.update(State.SPEAKING, Affect(), (0.0, 0.0), 0.05)
            clock.t += 0.05
        assert abs(f["joints"]["elbow_pitch_joint"] - POSES[State.SPEAKING]["elbow"]) < 0.05
    finally:
        behaviors.time = orig


def test_idle_life_glances_while_engaged_but_not_while_listening():
    from app.body import behaviors

    clock = type("Clock", (), {"t": 0.0, "monotonic": lambda self: self.t})()
    orig, behaviors.time = behaviors.time, clock
    try:
        def neck_range(state, seconds):
            b = Body(URDF)
            vals = []
            for i in range(int(seconds / 0.05)):
                v = b.update(state, Affect(), (0.0, 0.0), 0.05)["joints"]["neck_yaw_joint"]
                clock.t += 0.05
                if i * 0.05 > 3.0:  # after the pose has settled (LISTENING itself cocks the neck 0.5 rad)
                    vals.append(v)
            return max(vals) - min(vals)

        assert neck_range(State.ENGAGED, 30) > 0.25  # glanced at least once in 30 s
        assert neck_range(State.LISTENING, 30) < 0.05  # holds still while listening
    finally:
        behaviors.time = orig


def test_sleeping_light_breathes():
    from app.body import behaviors

    clock = type("Clock", (), {"t": 0.0, "monotonic": lambda self: self.t})()
    orig, behaviors.time = behaviors.time, clock
    try:
        b, levels = Body(URDF), []
        for _ in range(240):
            levels.append(b.update(State.IDLE, Affect(), (0.0, 0.0), 0.05)["light"]["brightness"])
            clock.t += 0.05
        assert max(levels) <= 0.15 and max(levels) - min(levels) > 0.05
    finally:
        behaviors.time = orig
