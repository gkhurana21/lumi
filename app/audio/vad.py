"""Frame-level VAD with pre-roll and hangover. Emits ("start", None) and ("end", utterance_pcm)."""
from __future__ import annotations

import collections

import numpy as np
import webrtcvad


class VAD:
    FRAME_MS = 30

    def __init__(self, sr: int = 16000, aggressiveness: int = 2, start_frames: int = 4, end_frames: int = 15,
                 snr: float = 2.5, min_peaks: int = 4):
        self.sr = sr
        self.vad = webrtcvad.Vad(aggressiveness)
        self.frame_bytes = sr * self.FRAME_MS // 1000 * 2
        self.start_frames = start_frames   # 120 ms voiced to start
        self.end_frames = end_frames       # 450 ms of silence ends a turn (was 600; replies start sooner)
        self.strict = False                # set while SPEAKING: harder to trigger, limits echo false barge-ins
        self._buf = bytearray()
        self._pre: collections.deque[bytes] = collections.deque(maxlen=10)  # 300 ms pre-roll
        self._utt = bytearray()
        self._run = 0
        self.speaking = False
        # webrtcvad adapts during speech and can keep calling steady room noise "speech": utterances never end, or
        # restart on noise. A frame also has to beat the running noise RMS (unvoiced frames) by `snr` to count.
        self.snr = snr
        self.noise: float | None = None  # EMA of RMS over unvoiced frames outside utterances
        # Tones (beeps, the lamp's own synthesized music) pass webrtcvad and the energy gate but have 1 to 3 spectral
        # peaks; voiced speech has many harmonics. An utterance only starts once its voiced run holds a frame with
        # at least `min_peaks` peaks within 20 dB of the strongest (0 disables). Continuing utterances are unaffected.
        self.min_peaks = min_peaks
        self._rich = 0
        self._win = np.hanning(self.frame_bytes // 2)

    def feed(self, pcm: bytes) -> list[tuple[str, bytes | None]]:
        events: list[tuple[str, bytes | None]] = []
        self._buf += pcm
        fb = self.frame_bytes
        need = self.start_frames * (3 if self.strict else 1)
        while len(self._buf) >= fb:
            frame = bytes(self._buf[:fb])
            del self._buf[:fb]
            rms = float(np.sqrt(np.mean(np.frombuffer(frame, np.int16).astype(np.float32) ** 2)))
            voiced = self.vad.is_speech(frame, self.sr) and rms > self.snr * max(self.noise or 0.0, 20.0)
            if not self.speaking:
                if not voiced:
                    self.noise = rms if self.noise is None else 0.95 * self.noise + 0.05 * rms
                self._pre.append(frame)
                self._run = self._run + 1 if voiced else 0
                self._rich = (self._rich + self._harmonic(frame)) if voiced else 0
                if self._run >= need and (not self.min_peaks or self._rich):
                    self.speaking, self._run, self._rich = True, 0, 0
                    self._utt = bytearray(b"".join(self._pre))
                    self._pre.clear()
                    events.append(("start", None))
            else:
                self._utt += frame
                self._run = 0 if voiced else self._run + 1
                if self._run >= self.end_frames:
                    events.append(("end", self.flush()))
        return events

    def _harmonic(self, frame: bytes) -> bool:
        if not self.min_peaks:
            return True
        x = np.frombuffer(frame, np.int16).astype(np.float32) * self._win
        spec = np.abs(np.fft.rfft(x)) ** 2
        hz = self.sr / len(x)
        spec = spec[int(100 / hz):int(4000 / hz)]  # voice band
        mid = spec[1:-1]
        peaks = (mid > spec[:-2]) & (mid >= spec[2:]) & (mid > spec.max() / 100)  # within 20 dB of the strongest
        return int(peaks.sum()) >= self.min_peaks

    def flush(self) -> bytes:
        utt = bytes(self._utt) if self.speaking else b""
        self._utt, self._run, self._rich, self.speaking = bytearray(), 0, 0, False
        return utt
