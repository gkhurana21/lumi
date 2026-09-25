from app.core.fsm import TIMEOUTS, Ev, Machine, State


def run(m, *evs):
    for ev in evs:
        assert m.fire(ev), f"{ev} rejected in {m.state}"


def test_full_turn():
    m = Machine()
    run(m, Ev.FACE_SEEN, Ev.ATTN_ON, Ev.SPEECH_START, Ev.SPEECH_END, Ev.REPLY_READY, Ev.TTS_DONE)
    assert m.state == State.ENGAGED


def test_barge_in():
    m = Machine()
    run(m, Ev.SPEECH_START, Ev.SPEECH_END, Ev.REPLY_READY, Ev.SPEECH_START)
    assert m.state == State.LISTENING


def test_disengage_then_sleep():
    m = Machine()
    run(m, Ev.FACE_SEEN, Ev.ATTN_ON, Ev.ATTN_OFF)
    assert m.state == State.DISENGAGING
    m.entered -= TIMEOUTS[State.DISENGAGING] + 0.1
    m.tick(0.05)
    assert m.state == State.IDLE


def test_irrelevant_event_ignored():
    m = Machine()
    assert not m.fire(Ev.TTS_DONE) and m.state == State.IDLE


def test_affect_follows_emotion_then_baseline():
    m = Machine()
    m.affect.set_emotion("excited", hold=10)
    for _ in range(60):
        m.affect.step(0.05)
    assert m.affect.label() == "excited"


def test_face_without_attention_does_not_reengage():
    m = Machine()
    run(m, Ev.FACE_SEEN, Ev.ATTN_ON, Ev.ATTN_OFF)
    assert m.state == State.DISENGAGING
    assert not m.fire(Ev.FACE_LOST) and not m.fire(Ev.FACE_SEEN)  # e.g. a 45 deg glance: face, no attention
    assert m.state == State.DISENGAGING
    run(m, Ev.ATTN_ON)
    assert m.state == State.ENGAGED


def test_attention_hysteresis_and_debounce():
    from app.vision.attention import AttentionTracker
    from app.vision.types import Reading

    a = AttentionTracker(on_thresh=0.25, off_thresh=0.40)
    evs = [a.decide(Reading(face=True, yaw=0.1)) for _ in range(3)]
    assert evs[1] == [Ev.FACE_SEEN] and evs[2] == [Ev.ATTN_ON] and a.attending
    for _ in range(20):  # 0.3 is outside the on-threshold but inside the off-threshold: stays attending
        assert a.decide(Reading(face=True, yaw=0.3)) == []
    evs = [a.decide(Reading(face=True, yaw=0.5)) for _ in range(6)]
    assert evs[-1] == [Ev.ATTN_OFF] and not any(evs[:-1]) and not a.attending
    evs = [a.decide(Reading(face=True, yaw=0.3)) for _ in range(5)]
    assert not any(evs)  # 0.3 no longer counts once attention is off


def test_goal_acts_then_speaks_the_outcome():
    m = Machine()
    run(m, Ev.FACE_SEEN, Ev.ATTN_ON, Ev.GOAL, Ev.REPLY_READY, Ev.TTS_DONE)
    assert m.state == State.ENGAGED
    run(m, Ev.GOAL, Ev.SPEECH_START)  # talking over the lamp cancels the goal
    assert m.state == State.LISTENING
    m2 = Machine()
    run(m2, Ev.FACE_SEEN, Ev.ATTN_ON, Ev.GOAL)
    assert not m2.fire(Ev.ATTN_OFF)  # looking at the target instead of the lamp does not abort the goal
    m2.entered -= TIMEOUTS[State.ACTING] + 0.1
    m2.tick(0.05)
    assert m2.state == State.ENGAGED
