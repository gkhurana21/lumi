from app.body.behaviors import Body
from app.core.fsm import Affect, State


def test_body_respects_limits_and_emits_light():
    b = Body("missing.urdf")
    b.gesture("shake")
    for _ in range(40):
        f = b.update(State.ENGAGED, Affect(), (1.0, -1.0), 0.05)
    assert set(f) == {"joints", "light"} and len(f["joints"]) == 5
    assert all(-3.1416 <= v <= 3.1416 for v in f["joints"].values())
