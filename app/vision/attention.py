"""Local, per-frame attention. Cheap enough for 5 fps; cloud VLM is only used for scene understanding."""
from __future__ import annotations

import threading

import cv2
import numpy as np

from ..core.fsm import Ev
from .types import Reading


class _Debounce:
    def __init__(self, on_n: int, off_n: int):
        self.on_n, self.off_n, self.value, self.run = on_n, off_n, False, 0

    def update(self, raw: bool) -> bool | None:
        if raw == self.value:
            self.run = 0
            return None
        self.run += 1
        if self.run >= (self.on_n if raw else self.off_n):
            self.value, self.run = raw, 0
            return raw
        return None


class AttentionTracker:
    """`read` detects (MediaPipe); `decide` turns readings into debounced edge events. Tests override `read`."""

    def __init__(self, min_face_width: float = 0.08, on_thresh: float = 0.25, off_thresh: float = 0.40):
        self._det = None  # created by warm(): mediapipe is heavy
        self._lock = threading.Lock()
        self.min_w = min_face_width
        self.on_thresh, self.off_thresh = on_thresh, off_thresh  # hysteresis on |yaw|
        self._face = _Debounce(2, 8)   # quick to notice, slow to forget
        self._attn = _Debounce(3, 6)

    @property
    def attending(self) -> bool:
        """Debounced attention level (the edge events only report changes)."""
        return self._attn.value

    def warm(self) -> None:
        """Load the model before the first frame matters (first use costs about 1.1 s). Call off the event loop."""
        with self._lock:  # warm-up thread and the first frame may race
            if self._det is None:
                import mediapipe as mp
                det = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)
                det.process(np.zeros((480, 640, 3), np.uint8))
                self._det = det

    def read(self, jpeg: bytes) -> Reading:
        self.warm()
        r = Reading()
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return r
        res = self._det.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if res.detections:
            d = max(res.detections, key=lambda d: d.location_data.relative_bounding_box.width)
            bb = d.location_data.relative_bounding_box
            if bb.width >= self.min_w:
                kp = d.location_data.relative_keypoints  # right_eye, left_eye, nose, mouth, r_ear, l_ear
                mid = (kp[0].x + kp[1].x) / 2
                eye = abs(kp[0].x - kp[1].x) or 1e-6
                r.face, r.w, r.h = True, bb.width, bb.height
                r.yaw = (kp[2].x - mid) / eye
                r.x = (bb.xmin + bb.width / 2) * 2 - 1
                r.y = (bb.ymin + bb.height / 2) * 2 - 1
        return r

    def decide(self, r: Reading) -> list[Ev]:
        thresh = self.off_thresh if self._attn.value else self.on_thresh
        r.attending = r.face and abs(r.yaw) < thresh
        events: list[Ev] = []
        f = self._face.update(r.face)
        if f is True:
            events.append(Ev.FACE_SEEN)
        elif f is False:
            events.append(Ev.FACE_LOST)
        a = self._attn.update(r.attending)
        if a is True:
            events.append(Ev.ATTN_ON)
        elif a is False:
            events.append(Ev.ATTN_OFF)
        return events

    def update(self, jpeg: bytes) -> tuple[list[Ev], Reading]:
        r = self.read(jpeg)
        return self.decide(r), r
