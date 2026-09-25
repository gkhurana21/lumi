"""Summarize a session trace (out/traces/*.jsonl): engagement latency, flapping, yaw spread, pipeline latency.

  python scripts/analyze_trace.py            # latest trace
  python scripts/analyze_trace.py FILE.jsonl
"""
import glob
import json
import os
import statistics as st
import sys


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else float("nan")


def summary(name, xs, unit=""):
    if not xs:
        return f"{name}: none"
    return f"{name}: n={len(xs)} median={st.median(xs):.0f}{unit} p90={pct(xs, 90):.0f}{unit} max={max(xs):.0f}{unit}"


def streak_start(frames, t_end, want):
    """Time of the first frame in the unbroken run of frames with attending == want that ends at t_end."""
    start = None
    for f in frames:
        if f["t"] > t_end:
            break
        start = (start or f["t"]) if f["attending"] == want else None
    return start


path = sys.argv[1] if len(sys.argv) > 1 else max(glob.glob("out/traces/*.jsonl"), key=os.path.getmtime)
recs = [json.loads(line) for line in open(path)]
frames = [r for r in recs if r["kind"] == "frame"]
trans = [r for r in recs if r["kind"] == "transition"]
metrics = [r for r in recs if r["kind"] == "metrics"]
print(f"trace {path}: {len(frames)} frames, {len(trans)} transitions, {len(metrics)} turns")
if frames:
    span = frames[-1]["t"] - frames[0]["t"]
    gaps = [(b["t"] - a["t"]) * 1000 for a, b in zip(frames, frames[1:], strict=False)]
    print(f"duration {span:.1f} s, processed {len(frames) / max(span, 1e-9):.1f} fps")
    print(summary("frame interval", gaps, " ms"))
    print(summary("attention processing", [f["ms"] for f in frames], " ms"))

print("\nTRANSITIONS")
t0 = frames[0]["t"] if frames else (trans[0]["t"] if trans else 0)
for r in trans:
    print(f"  {r['t'] - t0:7.2f} s  {r['prev']:>11s} -> {r['next']:<11s} ({r['event']})")

eng = [r for r in trans if r["next"] == "engaged" and r["prev"] in ("idle", "noticing", "disengaging")]
lat = [(r["t"] - s) * 1000 for r in eng if (s := streak_start(frames, r["t"], True)) is not None]
print("\nENGAGEMENT")
print(summary("first attending frame -> ENGAGED", lat, " ms"), "(add up to 200 ms capture interval + transport)")
dis = [r for r in trans if r["next"] == "disengaging"]
dlat = [(r["t"] - s) * 1000 for r in dis if (s := streak_start(frames, r["t"], False)) is not None]
print(summary("first non-attending frame -> DISENGAGING", dlat, " ms"))
sleeps = [(b["t"] - a["t"]) for a, b in zip(trans, trans[1:], strict=False)
          if a["next"] == "disengaging" and b["next"] == "idle"]
print(f"DISENGAGING -> IDLE: {[round(s, 1) for s in sleeps]} s")
flaps = [(b["t"] - a["t"]) for a, b in zip(trans, trans[1:], strict=False)
         if a["next"] == "disengaging" and b["next"] == "engaged"]
print(f"re-engaged from DISENGAGING: {len(flaps)} times, after {[round(f, 1) for f in flaps]} s")

faces = [f for f in frames if f["face"]]
if faces:
    yaws = [abs(f["yaw"]) for f in faces]
    print("\nATTENTION SIGNAL (frames with a face)")
    width = st.median(f["w"] for f in faces)
    print(f"  face in {len(faces) / len(frames):.0%} of frames; face width median {width:.3f}")
    print(f"  |yaw| p10 {pct(yaws, 10):.2f}  p50 {pct(yaws, 50):.2f}  p90 {pct(yaws, 90):.2f}  max {max(yaws):.2f}")
    for th in (0.25, 0.40):
        print(f"  |yaw| >= {th:.2f} in {sum(y >= th for y in yaws) / len(yaws):.0%} of face frames")

vad = [r for r in recs if r["kind"] == "vad"]
if vad:
    print("\nSPEECH")
    starts = [r for r in vad if r["ev"] == "start"]
    ends = [r["sec"] for r in vad if r["ev"] == "end"]
    print(f"  utterances started: {len(starts)} ({sum(r['strict'] for r in starts)} while the lamp was speaking)")
    if ends:
        print(f"  utterance length: median {st.median(ends):.1f} s, shortest {min(ends):.1f} s")
    barge = [r for r in trans if r["prev"] == "speaking" and r["event"] == "speech_start"]
    print(f"  barge-ins (lamp audio stopped): {len(barge)} at {[round(r['t'] - t0, 1) for r in barge]} s")
    aborted = [r for r in trans if r["prev"] == "thinking" and r["event"] == "abort"]
    print(f"  turns with an empty transcript: {len(aborted)}")

if metrics:
    print("\nTURN LATENCY")
    for k in ("stt_ms", "llm_ms", "first_audio_ms"):
        print(" ", summary(k, [m[k] for m in metrics if k in m], " ms"))
