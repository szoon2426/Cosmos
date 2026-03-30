from dataclasses import dataclass


@dataclass
class WebcamCapture:
    camera_index: int = 0
    width: int = 1280
    height: int = 720
