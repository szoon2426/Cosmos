from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np


LANDMARK_NAMES = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR",
    "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_PINKY", "RIGHT_PINKY",
    "LEFT_INDEX", "RIGHT_INDEX",
    "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]


class PoseEstimator:
    def __init__(self, model_path: str | None = None):
        if model_path is None:
            model_path = str(
                Path(__file__).resolve().parents[2] / "interaction" / "pose_landmarker.task"
            )

        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Pose model not found: {model_path}")

        model_bytes = model_file.read_bytes()

        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._ts_ms = 0

    def process(self, frame_rgb: np.ndarray):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._ts_ms += 33
        return self.landmarker.detect_for_video(mp_image, self._ts_ms)

    def get_landmarks_as_dict(self, results, frame_width: int, frame_height: int) -> list[dict] | None:
        if not results.pose_landmarks:
            return None

        items: list[dict] = []
        for idx, lm in enumerate(results.pose_landmarks[0]):
            name = LANDMARK_NAMES[idx] if idx < len(LANDMARK_NAMES) else str(idx)
            items.append(
                {
                    "name": name,
                    "x": lm.x * frame_width,
                    "y": lm.y * frame_height,
                    "z": lm.z,
                    "nx": lm.x,
                    "ny": lm.y,
                    "nz": lm.z,
                    "visibility": lm.visibility if hasattr(lm, "visibility") else 1.0,
                }
            )
        return items

    def close(self) -> None:
        self.landmarker.close()
