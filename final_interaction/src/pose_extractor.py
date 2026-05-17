from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


UPPER_BODY_LANDMARKS = (
    "NOSE",
    "LEFT_EYE",
    "RIGHT_EYE",
    "LEFT_EAR",
    "RIGHT_EAR",
    "MOUTH_LEFT",
    "MOUTH_RIGHT",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_INDEX",
    "RIGHT_INDEX",
    "LEFT_THUMB",
    "RIGHT_THUMB",
)

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


class PoseExtractor:
    def __init__(self, model_path: str | None = None):
        if model_path is None:
            model_path = str(
                Path(__file__).resolve().parents[1] / "pose_landmarker.task"
            )
        model_bytes = Path(model_path).read_bytes()
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

    def extract_upper_body(self, results) -> dict[str, tuple[float, float]] | None:
        if not results.pose_landmarks:
            return None
        pose = results.pose_landmarks[0]
        output: dict[str, tuple[float, float]] = {}
        for idx, lm in enumerate(pose):
            if idx >= len(LANDMARK_NAMES):
                continue
            name = LANDMARK_NAMES[idx]
            if name in UPPER_BODY_LANDMARKS:
                output[name] = (lm.x, lm.y)
        return output

    @staticmethod
    def detect_face_presence(landmarks: dict[str, tuple[float, float]] | None) -> bool:
        if not landmarks:
            return False

        if "NOSE" not in landmarks:
            return False
        if "LEFT_EYE" not in landmarks or "RIGHT_EYE" not in landmarks:
            return False

        nose = landmarks["NOSE"]
        left_eye = landmarks["LEFT_EYE"]
        right_eye = landmarks["RIGHT_EYE"]

        eye_left_x = min(left_eye[0], right_eye[0])
        eye_right_x = max(left_eye[0], right_eye[0])
        eye_span = abs(right_eye[0] - left_eye[0])

        if eye_span < 0.015 or eye_span > 0.35:
            return False
        if not (eye_left_x <= nose[0] <= eye_right_x):
            return False

        eye_y = (left_eye[1] + right_eye[1]) * 0.5
        if nose[1] < eye_y - 0.08:
            return False
        if nose[1] > eye_y + 0.18:
            return False

        return True

    def close(self) -> None:
        self.landmarker.close()
