# Tasks

Work top to bottom. A milestone is done only when its checks pass. Mark `[x]` as you go.

## M0 Setup
- [x] `make install`, `make test`, `make lint` all green.
- [x] Supplied files placed at `robot/dummy_lamp_5dof.urdf` and `robot/assets/lamp_shade.stl`; `make urdf` shows every mesh `OK`.
- [x] `make render` produces PNGs. (Unverified script: fix it if it fails.)

## M1 Body mapping
- [x] From `make urdf` + renders, correct `ROLES` order and `POSES` signs in `app/body/behaviors.py`.
- [x] Check: in renders, IDLE is visibly slumped, ENGAGED upright and facing forward, THINKING looks up, LISTENING tilts. Every gesture reads clearly at its peak.
- [x] Shade link found by `sim.py` (name contains "shade", else fix the lookup).
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

## M5b Goal-directed action (from CHALLENGE.md, moment 5)
- [x] Action representation: a small closed set of lamp actions (e.g. look at a scene point, aim the light at an object, nod, return to user). Document the boundary: models choose from the set, the body owns kinematics, limits, and timing.
- [x] Spoken goal about the live scene (e.g. "shine your light on my mug") -> locate the object with vision -> plan a sequence of actions -> execute -> re-observe the scene -> confirm or retry once -> speak the outcome.
- [x] Keep invariant 3: cloud parses the goal and describes the scene; a local planner picks and runs the actions. If that proves insufficient, ask before changing the invariant.
- [x] Fake provider covers it; smoke test drives a goal end to end.
- [ ] HUMAN: run one goal live.

## M6 Character pass
- [ ] Walk the README demo script end to end. Every moment in the spec appears, plus motion, light, voice, one SFX, and music.
- [ ] Greeting, SFX, and motion feel intentional together (timing, not just presence).

## M7 Submission
- [ ] Fill the supplied `SUBMISSION.md`. README: measurements, hardware used, cloud data table accurate, completed vs left out.
- [ ] `make test` and `make lint` green. Fresh clone + README instructions works.
- [ ] Ubuntu 24.04 target (4 cores, 8 GB, no GPU): setup and run instructions, system packages, Python version (24.04 ships 3.12; Makefile hard-codes `python3.11`), headless sim option, audio and camera notes.
- [ ] Measure CPU and memory during a full demo run; engagement reliability (false and missed engagements).
- [ ] Technical note, at most 2 pages: architecture and data-flow diagram; protocol, model-to-action, simulation, deployment, key choices; reliability, latency, CPU and memory numbers; known limitations.

## Stretch (only after M7)
- Sentence-level TTS pipelining. Streaming STT. Greeting that mentions a remembered object.
