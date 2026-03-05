"""
hand.py — MediaPipe Hand Landmarker 기반 손 상태 감지

21개 손 랜드마크를 사용해 주먹(fist) vs 손바닥(open palm)을 구분합니다.

손가락 인덱스:
  엄지: 1(CMC) 2(MCP) 3(IP) 4(TIP)
  검지: 5(MCP) 6(PIP) 7(DIP) 8(TIP)
  중지: 9(MCP) 10(PIP) 11(DIP) 12(TIP)
  약지: 13(MCP) 14(PIP) 15(DIP) 16(TIP)
  새끼: 17(MCP) 18(PIP) 19(DIP) 20(TIP)
"""

import os
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np


# (TIP index, PIP index) — 4개 손가락 (엄지 제외)
FINGER_PAIRS = [
    (8,  6),   # 검지
    (12, 10),  # 중지
    (16, 14),  # 약지
    (20, 18),  # 새끼
]


def _is_finger_curled(lms: list, tip: int, pip: int) -> bool:
    """TIP이 PIP보다 아래(화면 기준 y 증가)에 있으면 접힌 것."""
    return lms[tip].y > lms[pip].y


def _hand_is_fist(hand_lms: list) -> bool:
    """4개 손가락이 모두 접혀 있으면 주먹."""
    return all(_is_finger_curled(hand_lms, tip, pip) for tip, pip in FINGER_PAIRS)


def _hand_is_open(hand_lms: list) -> bool:
    """4개 손가락이 모두 펴져 있으면 손바닥."""
    return all(not _is_finger_curled(hand_lms, tip, pip) for tip, pip in FINGER_PAIRS)


class HandEstimator:
    """
    MediaPipe Hand Landmarker를 사용해 양손의 상태를 감지합니다.

    result_for_frame() 호출 시 반환값:
        {
            "Left":  {"fist": bool, "open": bool, "wrist": (nx, ny)} | None,
            "Right": {"fist": bool, "open": bool, "wrist": (nx, ny)} | None,
        }
    """

    def __init__(self, model_path: str = "hand_landmarker.task", num_hands: int = 2):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Hand Landmarker 모델 파일을 찾을 수 없습니다: {model_path}"
            )

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._ts_ms = 0
        print("[Hand] MediaPipe HandLandmarker 초기화 완료")

    def process(self, frame_rgb: np.ndarray) -> dict:
        """
        RGB 프레임을 분석해 양손 상태를 반환합니다.

        Returns:
            {"Left": {...} | None, "Right": {...} | None}
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._ts_ms += 33
        result = self.landmarker.detect_for_video(mp_image, self._ts_ms)

        h_info: dict[str, dict | None] = {"Left": None, "Right": None}

        if not result.hand_landmarks:
            return h_info

        for i, lms in enumerate(result.hand_landmarks):
            # handedness: "Left" or "Right" (MediaPipe 기준 — 거울상이므로 실제 반대)
            if i < len(result.handedness):
                side = result.handedness[i][0].display_name   # "Left" | "Right"
            else:
                continue

            wrist = lms[0]   # 0번이 손목
            h_info[side] = {
                "fist": _hand_is_fist(lms),
                "open": _hand_is_open(lms),
                "wrist": (wrist.x, wrist.y),          # 정규화 (0~1)
                "landmarks": [(lm.x, lm.y) for lm in lms],  # 21개 관절 정규화 좌표
            }

        return h_info

    def close(self):
        self.landmarker.close()
        print("[Hand] MediaPipe HandLandmarker 자원 해제 완료")
