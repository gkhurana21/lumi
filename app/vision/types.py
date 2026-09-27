from dataclasses import dataclass


@dataclass
class Reading:
    face: bool = False
    attending: bool = False
    x: float = 0.0    # face center in [-1, 1], camera frame (not mirrored)
    y: float = 0.0
    yaw: float = 0.0  # nose offset from eye midpoint, in eye-distances
    w: float = 0.0    # face box width as a fraction of the frame (proxy for distance)
    h: float = 0.0    # face box height as a fraction of the frame
