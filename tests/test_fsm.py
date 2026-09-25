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
