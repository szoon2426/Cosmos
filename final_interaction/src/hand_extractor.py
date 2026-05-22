from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


HAND_KEYPOINTS = {
    "WRIST": 0,
    "THUMB_TIP": 4,
    "INDEX_MCP": 5,
    "INDEX_TIP": 8,
    "MIDDLE_MCP": 9,
    "MIDDLE_TIP": 12,
    "RING_MCP": 13,
    "RING_TIP": 16,
    "PINKY_MCP": 17,
    "PINKY_TIP": 20,
}


class HandExtractor:
    def __init__(
        self,
        model_path: str | None = None,
        *,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        enable_image_mode: bool = False,
    ):
        if model_path is None:
            model_path = str(
                Path(__file__).resolve().parents[1] / "hand_landmarker.task"
            )
        model_bytes = Path(model_path).read_bytes()
        self._model_bytes = model_bytes
        self._min_hand_detection_confidence = min_hand_detection_confidence
        self._min_hand_presence_confidence = min_hand_presence_confidence
        self._min_tracking_confidence = min_tracking_confidence
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self.image_landmarker = self._create_image_landmarker() if enable_image_mode else None
        self._ts_ms = 0

    def _create_image_landmarker(self):
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=self._model_bytes),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=self._min_hand_detection_confidence,
            min_hand_presence_confidence=self._min_hand_presence_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
        )
        return mp_vision.HandLandmarker.create_from_options(options)

    def process(self, frame_rgb: np.ndarray):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._ts_ms += 33
        return self.landmarker.detect_for_video(mp_image, self._ts_ms)

    def detect_image(self, frame_rgb: np.ndarray):
        if self.image_landmarker is None:
            self.image_landmarker = self._create_image_landmarker()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        return self.image_landmarker.detect(mp_image)

    def extract_keypoints(self, results) -> dict[str, tuple[float, float]]:
        output: dict[str, tuple[float, float]] = {}
        if not results.hand_landmarks:
            return output

        for hand_idx, landmarks in enumerate(results.hand_landmarks):
            if hand_idx >= len(results.handedness):
                continue
            handed = results.handedness[hand_idx][0]
            side = str(getattr(handed, "display_name", "") or getattr(handed, "category_name", "")).upper()
            if side not in ("LEFT", "RIGHT"):
                continue

            for key_name, lm_idx in HAND_KEYPOINTS.items():
                lm = landmarks[lm_idx]
                output[f"{side}_HAND_{key_name}"] = (lm.x, lm.y)
        return output

    def close(self) -> None:
        self.landmarker.close()
        if self.image_landmarker is not None:
            self.image_landmarker.close()
