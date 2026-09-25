# CLAUDE.md

Lumi is a live character built around a simulated 5-DOF desk lamp (supplied URDF) for the Human Computer Lab "Live Character Robot Software Challenge". Laptop camera = eyes, mic = ears, speaker = voice/SFX/music, PyBullet = body (motion + shade light).

## The spec, in one breath
One continuous demo showing: (1) engagement: notices a person looking, disengages when attention leaves; (2) character response: acknowledges with motion + light + sound + voice; (3) spoken interaction via mic and speaker; (4) scene memory: sees an object, later answers a spoken question about it. Across the demo: motion, light, voice, a sound effect, and music, all purposeful. Timebox 6 to 8 hours. Coherence beats breadth. The README must explain decisions, trade-offs, cloud data flows, measurements, hardware, completed vs left out.

## Commands
- `make install` / `make test` / `make lint`
- `make fake` runs the server with canned STT/LLM/TTS (no keys). `make run` uses `.env`.
- `make urdf` prints joints, limits, mesh paths, and the role mapping the body uses.
- `make render` saves PNGs of every state pose and gesture to `out/poses/` (headless; view these to judge poses).
- `make sweep` / `make poses` open the PyBullet GUI (human only).
- Headless e2e: `tests/test_session_smoke.py` drives the real WebSocket with fakes. Extend it for any new behavior.

## Architecture
- `app/protocol.py` one WebSocket: binary `[kind byte][payload]` for audio/video, JSON text for control.
- `app/session.py` the orchestrator. Inputs -> FSM events. `_on_transition` is the ONLY place state side effects live.
- `app/core/fsm.py` engagement states (discrete) + Affect valence/arousal (continuous).
- `app/vision/attention.py` local MediaPipe face + nose-offset gaze heuristic, debounced.
- `app/audio/vad.py` webrtcvad segmentation, stricter while the lamp is speaking.
- `app/providers/` Protocols with `fakes.py` and `cloud.py` (OpenAI STT/TTS, Anthropic LLM + vision, tool-forced JSON).
- `app/mind/memory.py` scene memory (Chroma or dependency-free in-memory).
- `app/body/behaviors.py` state + affect + gaze + gestures + dance -> joint targets + light. `sim.py` only renders.
- `client/index.html` capture, voice playback, synthesized SFX and music, typed-input fallback.

## Invariants (do not break)
1. State changes only via `Machine.fire`. New behavior = new event/transition + side effect in `_on_transition` + a test.
2. Never block the event loop. CPU work -> `asyncio.to_thread`. Network -> async clients.
3. Behavioral decisions stay local. Cloud only supplies words and scene descriptions. Update the README data table if that changes.
4. Every provider has a fake, and `PROVIDERS=fake` must keep working end to end.
5. Protocol changes update `protocol.py`, the client, and the smoke test together.
6. Body code produces targets; sim code renders. No logic in `sim.py`.

## Conventions
Python 3.11, type hints, ruff clean, small functions. Prefer editing over adding files. Add a dependency only with a one-line reason in `docs/DECISIONS.md`. In all prose (README, docs, commit messages): concise, no em dashes.

## Known unknowns (resolve early)
- Real URDF joint order, axes, signs. Roles bind to movable joints in file order. Verify with `make urdf` + `make render`.
- Attention thresholds (`yaw_thresh`, `min_face_width`) depend on seating distance and camera.
- Speaker-to-mic echo causing false barge-in without headphones.
- `mediapipe` is pinned to 0.10.14 for the legacy `mp.solutions` API.

## What a human must verify
You cannot see the camera, hear audio, or watch the GUI. For anything live, prepare it, state exactly what to do and what should happen, then stop and ask. Record their results in `docs/DECISIONS.md`.
