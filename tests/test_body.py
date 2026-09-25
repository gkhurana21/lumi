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
