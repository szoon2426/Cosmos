from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np


FINGER_PAIRS = [
    (8, 6),
    (12, 10),
    (16, 14),
    (20, 18),
]


def _is_finger_curled(lms: list, tip: int, pip: int) -> bool:
    return lms[tip].y > lms[pip].y


def _hand_is_fist(hand_lms: list) -> bool:
    return all(_is_finger_curled(hand_lms, tip, pip) for tip, pip in FINGER_PAIRS)


def _hand_is_open(hand_lms: list) -> bool:
    return all(not _is_finger_curled(hand_lms, tip, pip) for tip, pip in FINGER_PAIRS)


class HandEstimator:
    def __init__(self, model_path: str | None = None, num_hands: int = 2):
        if model_path is None:
            model_path = str(
                Path(__file__).resolve().parents[2] / "interaction" / "hand_landmarker.task"
            )

        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Hand model not found: {model_path}")

        model_bytes = model_file.read_bytes()

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._ts_ms = 0

    def process(self, frame_rgb: np.ndarray) -> dict:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._ts_ms += 33
        result = self.landmarker.detect_for_video(mp_image, self._ts_ms)

        hands: dict[str, dict | None] = {"Left": None, "Right": None}
        if not result.hand_landmarks:
            return hands

        for i, lms in enumerate(result.hand_landmarks):
            if i >= len(result.handedness):
                continue
            side = result.handedness[i][0].display_name
            wrist = lms[0]
            hands[side] = {
                "fist": _hand_is_fist(lms),
                "open": _hand_is_open(lms),
                "wrist": (wrist.x, wrist.y),
                "landmarks": [(lm.x, lm.y) for lm in lms],
            }
        return hands

    def close(self) -> None:
        self.landmarker.close()
