"""Call each cloud provider (PROVIDERS=gemini or cloud) with fixed inputs and time it. Sends only synthetic data:
the `say`-generated test utterance, its transcript, and a rendered lamp pose (never camera or mic data).

  python scripts/check_cloud.py [runs]
"""
import asyncio
import os
import statistics as st
import sys
import time
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2  # noqa: E402

from app.config import settings  # noqa: E402
from app.providers import build_providers  # noqa: E402

RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
IMAGE = "out/poses/state_engaged.png"  # from `make render`


def stats(xs):
    xs = sorted(xs)
    return f"median {st.median(xs):5.0f} ms  p90 {xs[min(len(xs) - 1, round(0.9 * (len(xs) - 1)))]:5.0f} ms"


async def main() -> None:
    if settings.providers == "gemini" and not settings.gemini_api_key:
        sys.exit("Set GEMINI_API_KEY in .env first (https://aistudio.google.com/apikey).")
    if settings.providers == "cloud" and not (settings.anthropic_api_key and settings.openai_api_key):
        sys.exit("Set ANTHROPIC_API_KEY and OPENAI_API_KEY in .env first.")
    if settings.providers not in ("gemini", "cloud"):
        sys.exit("Set PROVIDERS=gemini (or cloud) in .env; fake providers make no network calls to measure.")
    print(f"providers: {settings.providers}")
    with wave.open("tests/data/utterance.wav") as w:
        pcm = w.readframes(w.getnframes())
    if not os.path.exists(IMAGE):
        sys.exit(f"{IMAGE} missing: run `make render` first.")
    jpeg = cv2.imencode(".jpg", cv2.imread(IMAGE), [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
    p = build_providers()
    stt, llm, tts, scene = p.stt, p.llm, p.tts, p.scene
    t = {"stt": [], "llm": [], "tts_first": [], "scene": [], "one_call": []}
    positions: dict[str, list[tuple[float, float]]] = {}
    for i in range(RUNS):
        t0 = time.perf_counter()
        text = await stt.transcribe(pcm)
        t["stt"].append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        reply = await llm.respond(user_text=text, history=[], memories=["mug: left of the keyboard, 2 min ago"],
                                  scene=[], mood="curious")
        t["llm"].append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        async for _ in tts.stream(reply["say"]):
            t["tts_first"].append((time.perf_counter() - t0) * 1000)
            break
        if hasattr(llm, "respond_to_audio"):  # gemini: hear and answer in one call
            t0 = time.perf_counter()
            one = await llm.respond_to_audio(audio=pcm, sr=16000, history=[], scene=[], mood="curious",
                                             memories=["mug was left of the keyboard (2 min ago)"])
            t["one_call"].append((time.perf_counter() - t0) * 1000)
            if i == 0:
                print(f"one-call heard: {one.get('heard')!r} -> {one.get('say')!r}")
        t0 = time.perf_counter()
        objs = await scene.describe(jpeg)
        t["scene"].append((time.perf_counter() - t0) * 1000)
        firsts = {}
        for o in objs:  # one box per name per frame: duplicates (two "lamp" boxes) would inflate the drift
            if "x" in o:
                firsts.setdefault(o["name"], (o["x"], o["y"]))
        for name, xy in firsts.items():
            positions.setdefault(name, []).append(xy)
        if i == 0:
            print(f"transcript: {text!r}\nreply: {reply}\nscene: {[o['name'] for o in objs]}\n")
    for k, xs in t.items():
        if xs:
            print(f"{k:10s} {stats(xs)}   (n={len(xs)})")
    total = [a + b + c for a, b, c in zip(t["stt"], t["llm"], t["tts_first"], strict=True)]
    print(f"{'two-step':10s} {stats(total)}   stt + llm + first TTS audio (add ~0.6 s VAD hangover)")
    if t["one_call"]:
        one_total = [a + b for a, b in zip(t["one_call"], t["tts_first"], strict=True)]
        print(f"{'one-call':10s} {stats(one_total)}   heard+reply + first TTS audio (what PROVIDERS=gemini uses)")

    print("\nscene position jitter on an identical image (calibrates MOVE_TOL in app/session.py):")
    for name, ps in positions.items():
        if len(ps) > 1:
            cx, cy = st.mean(p[0] for p in ps), st.mean(p[1] for p in ps)
            worst = max(((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 for x, y in ps)
            print(f"  {name:20s} seen {len(ps)}/{RUNS}  center ({cx:+.2f}, {cy:+.2f})  max drift {worst:.3f}")

    scene_now = [{"name": "desk lamp", "location": "center of the frame", "x": 0.0, "y": 0.1}]
    reply = await llm.respond(user_text="Can you shine your light on the desk lamp?", history=[], memories=[],
                              scene=scene_now, mood="curious")
    print(f"\ngoal request -> action {reply.get('action')}  say {reply['say']!r}")


asyncio.run(main())
