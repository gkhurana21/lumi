# Tasks

Work top to bottom. A milestone is done only when its checks pass. Mark `[x]` as you go.

## M0 Setup
- [x] `make install`, `make test`, `make lint` all green.
- [ ] Supplied files placed at `robot/dummy_lamp_6dof.urdf` and `robot/assets/lamp_shade.stl`; `make urdf` shows every mesh `OK`.
- [ ] `make render` produces PNGs. (Unverified script: fix it if it fails.)

## M1 Body mapping
- [ ] From `make urdf` + renders, correct `ROLES` order and `POSES` signs in `app/body/behaviors.py`.
- [ ] Check: in renders, IDLE is visibly slumped, ENGAGED upright and facing forward, THINKING looks up, LISTENING tilts. Every gesture reads clearly at its peak.
- [ ] Shade link found by `sim.py` (name contains "shade", else fix the lookup).
- [ ] HUMAN: `make sweep` then `make poses` look right.

## M2 Engagement (HUMAN live)
- [ ] Look at camera: NOTICING then ENGAGED in under 1 s, greeting fires once.
- [ ] Look away 90 degrees: DISENGAGING, then IDLE after about 5 s, sleep SFX.
- [ ] No state flapping during 60 s of normal sitting. Tune debounce/thresholds if needed; log values chosen.

## M3 Voice loop on cloud providers
- [ ] `.env` with keys, `PROVIDERS=cloud`. Unit-test providers with a recorded WAV where practical.
- [ ] HUMAN: 10 spoken turns. Collect `metrics` (stt_ms, llm_ms, first_audio_ms). Put median and p90 in README.
- [ ] Only if first_audio_ms median > 2500: stream LLM text and synthesize the first sentence early.

## M4 Barge-in and echo
- [ ] HUMAN: talking over the lamp stops its audio within about 200 ms.
- [ ] HUMAN: 5 turns on laptop speakers with no false barge-ins. If it fails, tune `VAD.strict`, else document "headphones recommended".

## M5 Scene memory
- [ ] HUMAN: show a mug, remove it, a minute later ask "where did I leave my mug?" Answer names the location and recency.
- [ ] Repeat with 3 objects. Add a smoke test covering an object that is no longer in `scene_now`.

## M6 Character pass
- [ ] Walk the README demo script end to end. Every moment in the spec appears, plus motion, light, voice, one SFX, and music.
- [ ] Greeting, SFX, and motion feel intentional together (timing, not just presence).

## M7 Submission
- [ ] Fill the supplied `SUBMISSION.md`. README: measurements, hardware used, cloud data table accurate, completed vs left out.
- [ ] `make test` and `make lint` green. Fresh clone + README instructions works.

## Stretch (only after M7)
- Sentence-level TTS pipelining. Streaming STT. Greeting that mentions a remembered object.
