"""CPU and memory of the server process tree under a client-like load, with fake providers (no network).

Load: 640x480 JPEG frames at 5 fps (like the browser), 20 ms mic chunks at 50 Hz, a typed turn every 10 s
(exercises LLM + TTS streaming). Run on the target machine for deployment numbers.

  python scripts/measure_load.py [seconds] [--sim]     # --sim also runs the PyBullet GUI process
"""
import asyncio
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import websockets

SECONDS = int(next((a for a in sys.argv[1:] if a.isdigit()), 30))
SIM = "--sim" in sys.argv
PORT = 8765
IMAGE = "out/poses/state_engaged.png"


def tree(pid: int) -> list[int]:
    out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()
    return [pid] + [p for c in out for p in tree(int(c))]


def cpu_seconds(pid: int) -> float:
    try:
        if os.path.exists(f"/proc/{pid}/stat"):  # Linux: utime + stime in clock ticks
            f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
            return (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")
        t = subprocess.run(["ps", "-o", "time=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        parts = [float(x) for x in t.replace("-", ":").split(":")]  # [[dd-]hh:]mm:ss.ss
        return sum(v * m for v, m in zip(reversed(parts), (1, 60, 3600, 86400), strict=False))
    except (OSError, ValueError, IndexError):
        return 0.0


def rss_mb(pid: int) -> float:
    r = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(r) / 1024 if r else 0.0


async def client(stop: asyncio.Event) -> None:
    img = cv2.imread(IMAGE) if os.path.exists(IMAGE) else np.full((480, 640, 3), 128, np.uint8)
    jpeg = cv2.imencode(".jpg", cv2.resize(img, (640, 480)), [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
    noise = (np.random.default_rng(0).normal(0, 30, 320)).astype(np.int16).tobytes()  # 20 ms at 16 kHz
    async with websockets.connect(f"ws://127.0.0.1:{PORT}/ws", max_size=None) as ws:
        async def drain():
            async for m in ws:
                if isinstance(m, str) and json.loads(m).get("type") == "tts_end":
                    await ws.send(json.dumps({"type": "playback_done"}))

        async def video():
            while not stop.is_set():
                await ws.send(b"\x02" + jpeg)
                await asyncio.sleep(0.2)

        async def audio():
            while not stop.is_set():
                await ws.send(b"\x01" + noise)
                await asyncio.sleep(0.02)

        async def turns():
            while not stop.is_set():
                await asyncio.sleep(10)
                await ws.send(json.dumps({"type": "text", "text": "what do you see on my desk?"}))

        tasks = [asyncio.create_task(c()) for c in (drain, video, audio, turns)]
        await stop.wait()
        for t in tasks:
            t.cancel()


async def main() -> None:
    env = {**os.environ, "PROVIDERS": "fake", "SIM": "true" if SIM else "false", "TRACE_DIR": ""}
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT),
                               "--log-level", "warning"], env=env)
    try:
        for _ in range(60):
            try:
                async with websockets.connect(f"ws://127.0.0.1:{PORT}/ws"):
                    break
            except OSError:
                await asyncio.sleep(0.5)
        base_rss = sum(rss_mb(p) for p in tree(server.pid))
        stop = asyncio.Event()
        load = asyncio.create_task(client(stop))
        await asyncio.sleep(3)  # warm-up (model load) excluded from the averages
        pids = tree(server.pid)
        c0, t0, samples = {p: cpu_seconds(p) for p in pids}, time.monotonic(), []
        while time.monotonic() - t0 < SECONDS:
            await asyncio.sleep(1)
            samples.append(sum(rss_mb(p) for p in tree(server.pid)))
        cores = sum(cpu_seconds(p) - c0[p] for p in pids) / (time.monotonic() - t0)
        stop.set()
        await load
        print(f"processes: {len(pids)} (server{' + sim' if SIM else ''} + helpers), sim={'GUI' if SIM else 'off'}")
        print(f"CPU: {cores:.2f} cores average over {SECONDS} s ({100 * cores / os.cpu_count():.0f}% "
              f"of {os.cpu_count()} cores)")
        print(f"RSS: idle after start {base_rss:.0f} MB, under load median {sorted(samples)[len(samples) // 2]:.0f} MB,"
              f" peak {max(samples):.0f} MB")
    finally:
        server.terminate()
        server.wait(timeout=10)


asyncio.run(main())
