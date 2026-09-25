"""Call each cloud provider with fixed inputs and time it. Needs keys in .env. Sends only synthetic data:
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
from app.providers.cloud import ClaudeLLM, ClaudeScene, OpenAISTT, OpenAITTS  # noqa: E402

RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
IMAGE = "out/poses/state_engaged.png"  # from `make render`


def stats(xs):
    xs = sorted(xs)
    return f"median {st.median(xs):5.0f} ms  p90 {xs[min(len(xs) - 1, round(0.9 * (len(xs) - 1)))]:5.0f} ms"


async def main() -> None:
    if not (settings.anthropic_api_key and settings.openai_api_key):
        sys.exit("Set ANTHROPIC_API_KEY and OPENAI_API_KEY in .env first.")
    with wave.open("tests/data/utterance.wav") as w:
        pcm = w.readframes(w.getnframes())
    if not os.path.exists(IMAGE):
        sys.exit(f"{IMAGE} missing: run `make render` first.")
    jpeg = cv2.imencode(".jpg", cv2.imread(IMAGE), [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
    stt, llm, tts, scene = OpenAISTT(), ClaudeLLM(), OpenAITTS(), ClaudeScene()
    t = {"stt": [], "llm": [], "tts_first": [], "scene": []}
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
        t0 = time.perf_counter()
        objs = await scene.describe(jpeg)
        t["scene"].append((time.perf_counter() - t0) * 1000)
        if i == 0:
            print(f"transcript: {text!r}\nreply: {reply}\nscene: {[o['name'] for o in objs]}\n")
    for k, xs in t.items():
        print(f"{k:10s} {stats(xs)}   (n={len(xs)})")
    total = [a + b + c for a, b, c in zip(t["stt"], t["llm"], t["tts_first"], strict=True)]
    print(f"{'sum':10s} {stats(total)}   stt + llm + first TTS audio; add ~0.6 s VAD hangover for end-of-speech")


asyncio.run(main())
