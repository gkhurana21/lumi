# Submission

## Deliverables

Submit the following:

1. **Source code** with its dependency declarations.
2. **Setup and run instructions** for the Ubuntu target described in
   [`CHALLENGE.md`](CHALLENGE.md).
3. **Short technical note** of no more than two pages containing:
   - One architecture and data-flow diagram.
   - Protocol, model-to-action, simulation, deployment, and key design choices.
   - Engagement reliability, response latency, CPU and memory measurements.
   - Known limitations.

The repository organization, documentation format, installation method, and
launch interface are candidate decisions.

## General evaluation criteria

| Area | Weight | What matters |
|---|---:|---|
| Architecture and protocol | 25% | Clear boundaries, ownership, and data flow. |
| Character integration | 20% | Coherent motion, light, voice, SFX, and music. |
| Perception, action, and memory | 20% | Grounded live decisions and recall. |
| Deployment | 15% | Practical Linux and embedded understanding. |
| Physical reasoning | 10% | Awareness of real robot constraints. |
| Evidence and communication | 10% | Measurements, tradeoffs, and ownership. |

## Where each deliverable is

1. **Source code with dependency declarations:** `app/` (server), `client/index.html` (browser I/O), `scripts/` (tools), `tests/`; `requirements.txt` (pinned direct dependencies) and `constraints.txt` (full pinned set).
2. **Setup and run instructions for Ubuntu 24.04:** `README.md` (Run), `deploy/setup_ubuntu.sh`, `deploy/lumi.service`.
3. **Technical note (at most 2 pages):** `docs/TECHNICAL_NOTE.md`. Longer rationale in `README.md`; every decision and measurement with evidence in `docs/DECISIONS.md`.
