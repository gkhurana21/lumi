# Decisions log

Append one entry per decision or measurement. Format: date, decision, why, evidence.

- Local attention (MediaPipe) + cloud scene VLM every 4 s. Engagement cannot wait on the network; object naming tolerates seconds.
- One multiplexed WebSocket. Ordered delivery makes barge-in (purge + stop) simple.
- Turn-based STT via VAD segmentation. Simpler than streaming STT; costs roughly 300 to 600 ms.
- Tool-forced JSON from the LLM (say, emotion, gesture, music) so the body always gets parseable intent.
- SFX and music synthesized in the browser. No licensing; tempo locked to the dance BPM.
- PyBullet in a spawned process. GUI owns its main thread (macOS) and cannot crash the conversation.
- Default LLM and vision model: claude-haiku-4-5 for latency and cost.

## 2026-09-24 M0 setup

- Machine: MacBook Air, Apple Silicon (arm64), macOS 26 SDK, Apple clang 21, Python 3.11.15.
- pybullet builds from a patched sdist on macOS. Why: PyPI has no macOS arm64 wheel, and the source build fails on the macOS 26 SDK because the bundled zlib `zutil.h` defines `fdopen` as a macro (a classic Mac OS stub) that breaks `_stdio.h`. Fix: `make pybullet-mac` downloads pybullet 3.2.7, deletes that 3-line stub, and builds. `make install` runs it only on Darwin when pybullet is not importable. Evidence: fresh `make install` from no venv, exit 0 in 135 s; 8 tests pass. No new dependency.
- `make install` no longer ends in `|| true`. Why: it hid the pybullet failure, and since pip installs atomically nothing was installed while the command still exited 0. Only the `.env` copy is allowed to fail (file exists).
- The pybullet build has numpy disabled, so `getCameraImage` returns a flat tuple. `sim_poses.py` converts it to a contiguous uint8 array before OpenCV touches it.
- mediapipe 0.10.14 works with the numpy 2.4 that pip resolves. Evidence: `FaceDetection.process` on a blank 640x480 frame returns cleanly. Not pinning numpy.
- `sim_poses.py --render` fixes (never run before). (1) Body times gestures and breathing with `time.monotonic()`, and render mode never sleeps, so gestures rendered at zero amplitude. The script now swaps in a simulated clock advanced 50 ms per step (render mode only; Body unchanged). (2) Rendering at `dur/2` missed oscillating peaks (shake at 3 Hz, 0.5 s is sin(3 pi) = 0), and the ENGAGED pose was only about 75% settled. Now: settle ENGAGED 3 s, play the gesture, render the frame with the largest deviation on the gesture's joint. (3) Each PNG shows two views side by side, yaw 45 and yaw 0 (side view, to judge pitch). Evidence: 13 PNGs rendered against a throwaway 5-joint stand-in URDF (not committed); the supplied URDF was not yet available.
- Finding for M1: at their peak, oscillating gestures reach a fraction of their amplitude because Body's smoothing low-passes them. Stand-in measurements: nod 0.07 of 0.25 rad, shake 0.07 of 0.30, bounce 0.04 of 0.12. Held gestures come through: tilt 0.32 of 0.35, droop 0.46 of 0.50, perk 0.22 of 0.25.
