#!/usr/bin/env bash
# One-time setup on a clean Ubuntu 24.04 LTS laptop (4 cores, 8 GB RAM, integrated GPU, no CUDA).
# Run from the repo root: bash deploy/setup_ubuntu.sh
set -euo pipefail

sudo apt-get update
# python3.12-venv/-dev + build-essential: venv, and a source build of pybullet if PyPI has no wheel for this CPU.
# libgl1, libglib2.0-0t64: runtime libs for opencv-contrib-python (pulled in by mediapipe).
# libegl1, mesa-utils: OpenGL on the integrated GPU for the PyBullet window (Mesa; `glxinfo -B` to check).
sudo apt-get install -y python3.12-venv python3-dev build-essential \
    libgl1 libglib2.0-0t64 libegl1 mesa-utils
# The client needs a Chromium-based browser (it captures the mic into a 16 kHz AudioContext; Firefox refuses
# to connect a mic stream to a context at a different sample rate).
command -v chromium >/dev/null || command -v google-chrome >/dev/null || sudo snap install chromium

make install   # venv on python3.12, pinned deps (constraints.txt), Chroma embedder prefetch, .env from example
make test      # unit + end-to-end tests with fake providers, no network
make render    # headless PyBullet: proves the supplied URDF and mesh load (PNGs in out/poses/)

echo "Done. Put API keys in .env (PROVIDERS=cloud), then: make run  (or: deploy/lumi.service)"
