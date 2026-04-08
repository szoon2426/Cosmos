from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "interaction" / "hand_landmarker.task"

# Four fingers except thumb: (TIP, PIP)
FINGER_PAIRS = [
    (8, 6),
    (12, 10),
    (16, 14),
    (20, 18),
]


def is_finger_curled(landmarks: list, tip_index: int, pip_index: int) -> bool:
    return landmarks[tip_index].y > landmarks[pip_index].y


def hand_is_fist(landmarks: list) -> bool:
    return all(is_finger_curled(landmarks, tip, pip) for tip, pip in FINGER_PAIRS)


def hand_is_open(landmarks: list) -> bool:
    return all(not is_finger_curled(landmarks, tip, pip) for tip, pip in FINGER_PAIRS)


class DebugHandEstimator:
    def __init__(self, model_path: Path, num_hands: int = 2):
        if not model_path.exists():
            raise FileNotFoundError(f"Hand Landmarker model not found: {model_path}")

        model_bytes = model_path.read_bytes()
        base_options = mp_python.BaseOptions(model_asset_buffer=model_bytes)
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

    def process(self, frame_rgb: np.ndarray) -> dict[str, dict | None]:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._ts_ms += 33
        result = self.landmarker.detect_for_video(mp_image, self._ts_ms)

        hand_info: dict[str, dict | None] = {"Left": None, "Right": None}
        if not result.hand_landmarks:
            return hand_info

        for index, hand_landmarks in enumerate(result.hand_landmarks):
            if index >= len(result.handedness):
                continue

            side = result.handedness[index][0].display_name
            wrist = hand_landmarks[0]
            hand_info[side] = {
                "fist": hand_is_fist(hand_landmarks),
                "open": hand_is_open(hand_landmarks),
                "wrist": (wrist.x, wrist.y),
                "landmarks": [(lm.x, lm.y) for lm in hand_landmarks],
            }

        return hand_info

    def close(self):
        self.landmarker.close()


def classify_hand_shape(hand_info: dict | None) -> str:
    if hand_info is None:
        return "missing"
    if hand_info["fist"]:
        return "fist"
    if hand_info["open"]:
        return "open"
    return "ambiguous"


def classify_gather_state(hand_info: dict[str, dict | None], gather_threshold: float = 0.18):
    left = hand_info["Left"]
    right = hand_info["Right"]

    if left is None or right is None:
        return "need_both_hands", None

    left_wrist = np.array(left["wrist"], dtype=np.float32)
    right_wrist = np.array(right["wrist"], dtype=np.float32)
    wrist_distance = float(np.linalg.norm(left_wrist - right_wrist))

    if wrist_distance > gather_threshold:
        return "hands_apart", wrist_distance

    left_shape = classify_hand_shape(left)
    right_shape = classify_hand_shape(right)

    if "fist" in (left_shape, right_shape):
        return "gather_fist", wrist_distance
    if "open" in (left_shape, right_shape):
        return "gather_open", wrist_distance
    return "gather_ambiguous", wrist_distance


def draw_hand_landmarks(frame, hand_info: dict[str, dict | None]):
    height, width = frame.shape[:2]
    for side in ("Left", "Right"):
        info = hand_info[side]
        if not info:
            continue

        for x, y in info["landmarks"]:
            px = int(x * width)
            py = int(y * height)
            cv2.circle(frame, (px, py), 3, (0, 255, 255), -1)

        wrist_x, wrist_y = info["wrist"]
        cv2.putText(
            frame,
            side,
            (int(wrist_x * width), int(wrist_y * height) - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def draw_status(frame, hand_info: dict[str, dict | None], gather_state: str, wrist_distance: float | None):
    lines = [
        f"Gather state: {gather_state}",
        f"Wrist dist:   {wrist_distance:.3f}" if wrist_distance is not None else "Wrist dist:   -",
        f"Left hand:    {classify_hand_shape(hand_info['Left'])}",
        f"Right hand:   {classify_hand_shape(hand_info['Right'])}",
        "ESC or Q to quit",
    ]

    y = 30
    for index, line in enumerate(lines):
        color = (0, 255, 0) if index == 0 and gather_state.startswith("gather_") else (255, 255, 255)
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 30


def main():
    hand_estimator = DebugHandEstimator(model_path=MODEL_PATH, num_hands=2)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    last_logged_state = None

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hand_info = hand_estimator.process(rgb)
            gather_state, wrist_distance = classify_gather_state(hand_info)

            if gather_state != last_logged_state:
                distance_text = f"{wrist_distance:.3f}" if wrist_distance is not None else "-"
                print(
                    f"[hand-debug] state={gather_state} "
                    f"left={classify_hand_shape(hand_info['Left'])} "
                    f"right={classify_hand_shape(hand_info['Right'])} "
                    f"wrist_distance={distance_text}"
                )
                last_logged_state = gather_state

            draw_hand_landmarks(frame, hand_info)
            draw_status(frame, hand_info, gather_state, wrist_distance)

            cv2.imshow("Hand State Debug", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand_estimator.close()


if __name__ == "__main__":
    main()
