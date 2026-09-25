"""Frame-level VAD with pre-roll and hangover. Emits ("start", None) and ("end", utterance_pcm)."""
from __future__ import annotations

import collections

import webrtcvad


class VAD:
    FRAME_MS = 30

    def __init__(self, sr: int = 16000, aggressiveness: int = 2, start_frames: int = 4, end_frames: int = 20):
        self.sr = sr
        self.vad = webrtcvad.Vad(aggressiveness)
        self.frame_bytes = sr * self.FRAME_MS // 1000 * 2
        self.start_frames = start_frames   # 120 ms voiced to start
        self.end_frames = end_frames       # 600 ms silence to end
        self.strict = False                # set while SPEAKING: harder to trigger, limits echo false barge-ins
        self._buf = bytearray()
        self._pre: collections.deque[bytes] = collections.deque(maxlen=10)  # 300 ms pre-roll
        self._utt = bytearray()
        self._run = 0
        self.speaking = False

    def feed(self, pcm: bytes) -> list[tuple[str, bytes | None]]:
        events: list[tuple[str, bytes | None]] = []
        self._buf += pcm
        fb = self.frame_bytes
        need = self.start_frames * (3 if self.strict else 1)
        while len(self._buf) >= fb:
            frame = bytes(self._buf[:fb])
            del self._buf[:fb]
            voiced = self.vad.is_speech(frame, self.sr)
            if not self.speaking:
                self._pre.append(frame)
                self._run = self._run + 1 if voiced else 0
                if self._run >= need:
                    self.speaking, self._run = True, 0
                    self._utt = bytearray(b"".join(self._pre))
                    self._pre.clear()
                    events.append(("start", None))
            else:
                self._utt += frame
                self._run = 0 if voiced else self._run + 1
                if self._run >= self.end_frames:
                    events.append(("end", self.flush()))
        return events

    def flush(self) -> bytes:
        utt = bytes(self._utt) if self.speaking else b""
        self._utt, self._run, self.speaking = bytearray(), 0, False
        return utt
