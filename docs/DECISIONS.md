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
- `sim_poses.py --render` fixes (never run before). (1) Body times gestures and breathing with `time.monotonic()`, and render mode never sleeps, so gestures rendered at zero amplitude. The script now swaps in a simulated clock advanced 50 ms per step (render mode only; Body unchanged). (2) Rendering at `dur/2` missed oscillating peaks (shake at 3 Hz, 0.5 s is sin(3 pi) = 0), and the ENGAGED pose was only about 75% settled. Now: settle ENGAGED 3 s, play the gesture, render the frame with the largest deviation on the gesture's joint. (3) Each PNG shows two views side by side, yaw 45 and yaw 0 (side view, to judge pitch). Evidence: 13 PNGs rendered, first against a throwaway stand-in URDF, then against the supplied URDF.
- Finding for M1: at their peak, oscillating gestures reach a fraction of their amplitude because Body's smoothing low-passes them. Measured on the supplied URDF (same on the stand-in): nod 0.07 of 0.25 rad, shake 0.07 of 0.30, bounce 0.04 of 0.12. Held gestures come through: tilt 0.32 of 0.35, droop 0.46 of 0.50, perk 0.22 of 0.25.
- Supplied starter pack is `dummy_lamp_5dof.urdf`, not the `6dof` name the scaffold guessed. Kept the supplied name and updated config, `.env.example`, scripts, and docs. Also committed `robot/dummy-lamp.png` (reference look), `CHALLENGE.md`, and `SUBMISSION.md` (it links to `CHALLENGE.md`).
- The supplied `lamp_shade.stl` is ASCII (360 facets). PyBullet only parses binary STL: `loadURDF` warns "cannot extract anything useful from mesh" and drops the shade, and `createCollisionShape` on it segfaults. `app/body/sim.py:sim_urdf` writes a binary copy (18084 bytes = 84 + 50 x 360) and a URDF pointing at it under `build/sim/`. Supplied files stay untouched. `sim.py` and `sim_poses.py` share this loader. Evidence: binary copy's collision AABB is x -0.032..0.122, y and z +-0.11, matching the vertex extent; test `test_sim_urdf_loads_supplied_mesh`.
- Camera reframed and shared (`sim.CAMERA`, used by the GUI and renders): distance 1.5 m, target z 0.38 m. The old 0.9 m / z 0.25 m framing cut off the head of the roughly 0.8 m lamp.
- `make urdf` joint order: base_yaw, shoulder_pitch, elbow_pitch, neck_yaw, head_pitch. The scaffold's ROLES assume head_pitch then head_roll for the last two, so today `head_pitch` drives neck_yaw and `head_roll` drives head_pitch. No link name contains "shade" (the shade is `lamp_head_link`; there is also `light_emitter_link`), so `sim.py`'s shade lookup falls back to the last joint (`speaker_link`, no visual). Both are M1 work.
- Spec gaps found in `CHALLENGE.md` / `SUBMISSION.md`, not covered by `docs/TASKS.md`: (1) a fifth required moment, goal-directed action: a spoken goal about the live scene drives a sequence of lamp actions chosen from vision + language, and the lamp re-observes before completing it; (2) deployment target is Ubuntu 24.04, 4 cores, 8 GB, no GPU, and setup instructions must address it; (3) deliverable is a technical note of at most 2 pages with a diagram and CPU and memory measurements. Raised with the user before planning any work.

## 2026-09-24 M1 body mapping

- Roles now match the URDF: base_yaw, shoulder, elbow, neck_yaw, head_pitch. There is no roll joint, so the scaffold's `head_roll` role is gone.
- Sign conventions from forward kinematics (`light_emitter_link` pose per joint at +0.4 rad). At zero the arm is vertical and the head (z 0.725 m) faces world -x, which we treat as toward the user. shoulder/elbow + lean back and tip the head up; base/neck yaw + turn to the lamp's left; head_pitch + looks down. Head elevation = shoulder + elbow - head_pitch. Gaze mapping `-k * gaze_x` was already the right sign.
- Limits: Body clamps to the URDF `safety_controller` soft limits (tighter than hard limits) and rate-limits every command to 0.95 x the URDF velocity limit. The sim motors now use the URDF force and velocity limits instead of a flat 50 N m and 4 rad/s. Consequence: a full IDLE to ENGAGED perk-up takes about 1 s (elbow moves 0.9 rad at 1.09 rad/s). Motion starts on the first 50 ms frame, so the reaction still feels immediate.
- The scaffold gestures were physically impossible: peak speed 2 pi f A for nod was 3.9 rad/s (limit 1.45), shake 5.7 (1.6), bounce 3.0 (0.95). Retuned under the limits, with nod as a one-sided "dip" (a nod goes down and back; the level-head ENGAGED pose leaves only 0.2 rad of upward range before the -0.80 soft limit).
- Gestures, breathing, dance, and speech bob are now added after pose smoothing. Before, the smoothing low-pass cut oscillating gestures to about 30%. Measured peak deviation on the main joint at the gesture's peak (renders): nod 0.224 rad (13 deg), shake 0.255, tilt 0.50, bounce 0.118, droop 0.45, perk 0.25. Test asserts at least 65% of design amplitude.
- Head cock without a roll joint: neck yaw on the forward-leaning upper arm rolls the head, and an opposite base yaw cancels the turn. FK: neck 0.5 + base -0.4 gives 18 deg roll at -0.5 deg yaw in LISTENING (15 deg in ENGAGED for the `tilt` gesture). Neck yaw alone (0.3) gave 11 deg roll but also a 14 deg turn away.
- Pose intent (head elevation, rad): IDLE -0.90 (slumped toward desk), DISENGAGING -0.60, ENGAGED +0.05, SPEAKING +0.05, LISTENING -0.05, NOTICING +0.15, THINKING +0.50 and turned 0.6 rad to its right (away from both the user and the GUI camera). Asserted in `test_pose_intent`.
- Gaze split across base yaw (-0.25) and neck yaw (-0.30) per unit face offset, total 0.55 rad, about half a typical 60 to 70 deg laptop camera FOV. Vertical 0.35 rad. Not yet checked live (M2).
- Light: `sim.py` looked for "shade" in link names and fell back to `speaker_link` (no visual). Now `light_link` prefers "light"/"emitter", so it tints `light_emitter_link`. Evidence: sim loop run headless in a thread; emitter rgba (0.93, 0.67, 0.39) for warm light at brightness 0.9; joints tracked commands within 0.02 rad after 3 s.
- Camera: front three-quarter view, yaw -60 (the lamp faces -x), plus a side view (yaw 0) in renders.
- `make sweep` now sweeps each joint over 80% of its URDF range; before it used +-0.6 rad, past the elbow's +0.30 upper limit.

## 2026-09-25 M2 engagement prep (before live testing)

- M1 GUI check (`make sweep`, `make poses`) not yet reported by the user; folded into the M2 live session.
- Bug: ENGAGED could get stuck with nobody looking. Attention events fire only on change, and the FSM drops `ATTN_OFF` in SPEAKING/THINKING. Look away during a reply, then `TTS_DONE` returns to ENGAGED with attention already off and no new `ATTN_OFF` coming. Fix: after each frame, if ENGAGED and debounced attention is off, fire `ATTN_OFF` (level check in `Session._on_video`; `NullAttention.attending = None` means unknown, never forces it). Smoke test reproduces the sequence over the real WebSocket; it fails without the fix.
- Bug: `DISENGAGING + FACE_SEEN -> ENGAGED` re-engaged on a visible but not-looking face (e.g. a 45 deg glance), which hit the stuck state above. Removed; re-engagement comes from `ATTN_ON`. README state diagram updated.
- Attention hysteresis: attending starts below |yaw| 0.25 and ends above 0.40 (was one 0.25 threshold). Starting values, to be tuned from live traces.
- MediaPipe first use costs 1115 ms (model load) and ran on the first real frame. Now warmed in a thread at session start. Measured on this Mac (M3) with the fake server: first frame 23 ms, then 6.6 to 12 ms per 640x480 frame; body updates held 9 of 10 expected during warm-up.
- Session trace: `out/traces/<time>.jsonl` with per-frame attention numbers (face, yaw, box, processing ms), transitions, and turn latency. Numbers only, never images or audio; stays on disk. `make trace` summarizes the latest: engagement and disengagement latency, re-engagements (flapping), |yaw| spread, frame processing, turn latency. Set `TRACE_DIR=` to disable.
- Expected engagement latency from the design: 3 attending frames at 5 fps = 400 to 600 ms after the first attending frame, plus up to 200 ms capture interval and processing. Target under 1 s.
