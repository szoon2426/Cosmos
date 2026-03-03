import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.components.containers import landmark as mp_landmark
import numpy as np
import os


# MediaPipe 포즈 랜드마크 이름 (33개, 순서 고정)
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

# MediaPipe 포즈 연결선 정의 (landmark index 쌍)
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(27,31),(29,31),
    (24,26),(26,28),(28,30),(28,32),(30,32),
]


class PoseEstimator:
    """MediaPipe Tasks PoseLandmarker를 사용해 인체 랜드마크를 추출하는 클래스."""

    def __init__(self, model_path: str = "pose_landmarker.task"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"모델 파일을 찾을 수 없습니다: {model_path}\n"
                "다음 명령어로 다운로드하세요:\n"
                "Invoke-WebRequest -Uri https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task -OutFile pose_landmarker.task"
            )

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._frame_ts_ms = 0
        print("[Pose] MediaPipe PoseLandmarker 초기화 완료")

    def process(self, frame_rgb: np.ndarray):
        """
        RGB 프레임을 입력받아 포즈 결과를 반환합니다.

        Args:
            frame_rgb: RGB 포맷의 numpy 프레임

        Returns:
            PoseLandmarkerResult 객체
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._frame_ts_ms += 33  # ~30fps 가정
        return self.landmarker.detect_for_video(mp_image, self._frame_ts_ms)

    def get_landmarks_as_dict(self, results, frame_width: int, frame_height: int) -> list[dict] | None:
        """
        랜드마크를 픽셀 좌표 딕셔너리 리스트로 반환합니다.

        Returns:
            [{"name": str, "x": float, "y": float, "z": float, "visibility": float}, ...] 또는 None
        """
        if not results.pose_landmarks:
            return None

        landmarks = []
        for idx, lm in enumerate(results.pose_landmarks[0]):
            name = LANDMARK_NAMES[idx] if idx < len(LANDMARK_NAMES) else str(idx)
            landmarks.append({
                "name": name,
                "x": lm.x * frame_width,
                "y": lm.y * frame_height,
                "z": lm.z,
                "visibility": lm.visibility if hasattr(lm, "visibility") else 1.0,
            })
        return landmarks

    def close(self):
        """MediaPipe 자원을 해제합니다."""
        self.landmarker.close()
        print("[Pose] MediaPipe PoseLandmarker 자원 해제 완료")
