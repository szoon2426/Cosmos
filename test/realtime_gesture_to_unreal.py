import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HAND_MODEL_PATH = PROJECT_ROOT / "interaction" / "hand_landmarker.task"
POSE_MODEL_PATH = PROJECT_ROOT / "interaction" / "pose_landmarker.task"
FINGER_PAIRS = [(8, 6), (12, 10), (16, 14), (20, 18)]
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]


@dataclass
class GestureRuntimeState:
    is_active: bool = False
    last_trigger_time: float = 0.0


@dataclass
class GestureResult:
    is_detected: bool
    debug_values: dict[str, float | str] = field(default_factory=dict)


@dataclass
class GestureDefinition:
    name: str
    detector: Callable[[np.ndarray], GestureResult]
    cooldown: float = 1.0


@dataclass
class ContinuousControlState:
    phase: str = "idle"
    hold_started_at: float | None = None
    active_started_at: float | None = None
    value: float = 0.0
    branch: str = "missing"


@dataclass
class VADState:
    valence: float = 0.25
    arousal: float = 0.75
    dominance: float = 0.25


class LandmarkSmoother:
    def __init__(self, window_size: int = 5):
        self.window = deque(maxlen=window_size)

    def update(self, landmarks):
        if landmarks is None:
            self.window.clear()
            return None

        frame_points = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks],
            dtype=np.float32,
        )
        self.window.append(frame_points)
        return np.mean(np.stack(self.window, axis=0), axis=0)


class HandEstimator:
    def __init__(self, model_path: Path, num_hands: int = 2):
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
            }

        return hand_info

    def close(self):
        self.landmarker.close()


class PoseEstimator:
    def __init__(self, model_path: Path):
        model_bytes = model_path.read_bytes()
        base_options = mp_python.BaseOptions(model_asset_buffer=model_bytes)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
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

    def close(self):
        self.landmarker.close()


def calculate_angle(point_a, point_b, point_c) -> float:
    a = np.array(point_a[:2], dtype=np.float32)
    b = np.array(point_b[:2], dtype=np.float32)
    c = np.array(point_c[:2], dtype=np.float32)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0.0 or norm_bc == 0.0:
        return 0.0

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def trigger_unreal_effect(event_name: str):
    print(f"Triggered gesture: {event_name}")


def log_open_state(event_name: str, open_control_result: GestureResult):
    debug = open_control_result.debug_values
    print(
        "[open-debug] "
        f"event={event_name} "
        f"phase={debug.get('phase', '-')} "
        f"branch={debug.get('branch', '-')} "
        f"release={debug.get('release_requested', 0.0)} "
        f"value={float(debug.get('open_value', 0.0)):.2f}"
    )


def send_open_control_value(value: float):
    print(f"Open value: {value:.2f}")


def send_rise_control_value(value: float):
    print(f"Rise value: {value:.2f}")


def log_gather_state(event_name: str, gather_result: GestureResult, vad_state: VADState):
    debug = gather_result.debug_values
    print(
        "[gather-debug] "
        f"event={event_name} "
        f"phase={debug.get('phase', '-')} "
        f"V={vad_state.valence:.2f} "
        f"A={vad_state.arousal:.2f} "
        f"D={vad_state.dominance:.2f}"
    )


def move_toward_target(current: float, target: float, rate: float) -> float:
    return current + (target - current) * rate


def _get_pose_points(smoothed_landmarks):
    return {
        "nose": smoothed_landmarks[0],
        "left_shoulder": smoothed_landmarks[11],
        "right_shoulder": smoothed_landmarks[12],
        "left_elbow": smoothed_landmarks[13],
        "right_elbow": smoothed_landmarks[14],
        "left_wrist": smoothed_landmarks[15],
        "right_wrist": smoothed_landmarks[16],
        "left_hip": smoothed_landmarks[23],
        "right_hip": smoothed_landmarks[24],
    }


def _has_min_visibility(points, visibility_threshold: float) -> bool:
    return min(point[3] for point in points) >= visibility_threshold


def _shoulder_span(points) -> float:
    return abs(points["left_shoulder"][0] - points["right_shoulder"][0])


def _wrist_span(points) -> float:
    return abs(points["left_wrist"][0] - points["right_wrist"][0])


def is_finger_curled(landmarks: list, tip_index: int, pip_index: int) -> bool:
    return landmarks[tip_index].y > landmarks[pip_index].y


def hand_is_fist(landmarks: list) -> bool:
    return all(is_finger_curled(landmarks, tip, pip) for tip, pip in FINGER_PAIRS)


def hand_is_open(landmarks: list) -> bool:
    return all(not is_finger_curled(landmarks, tip, pip) for tip, pip in FINGER_PAIRS)


def classify_hand_shape(hand_info: dict | None) -> str:
    if hand_info is None:
        return "missing"
    if hand_info["fist"]:
        return "fist"
    if hand_info["open"]:
        return "open"
    return "missing"


def classify_open_branch(hand_info: dict[str, dict | None]) -> str:
    left_state = classify_hand_shape(hand_info["Left"])
    right_state = classify_hand_shape(hand_info["Right"])
    states = {left_state, right_state}

    if "fist" in states and "open" in states:
        return "missing"
    if "fist" in states:
        return "gather"
    if "open" in states:
        return "open"
    return "missing"


def has_open_hand(hand_info: dict[str, dict | None]) -> bool:
    return "open" in {
        classify_hand_shape(hand_info["Left"]),
        classify_hand_shape(hand_info["Right"]),
    }


def detect_open_trigger_pose(
    smoothed_landmarks, visibility_threshold: float = 0.6
) -> GestureResult:
    points = _get_pose_points(smoothed_landmarks)
    tracked_points = [
        points["left_shoulder"],
        points["right_shoulder"],
        points["left_elbow"],
        points["right_elbow"],
        points["left_wrist"],
        points["right_wrist"],
        points["left_hip"],
        points["right_hip"],
    ]
    if not _has_min_visibility(tracked_points, visibility_threshold):
        return GestureResult(is_detected=False)

    shoulder_span = _shoulder_span(points)
    wrist_distance = np.linalg.norm(points["left_wrist"][:2] - points["right_wrist"][:2])
    wrist_midpoint_x = (points["left_wrist"][0] + points["right_wrist"][0]) / 2.0
    wrist_midpoint_y = (points["left_wrist"][1] + points["right_wrist"][1]) / 2.0
    left_hip_y = points["left_hip"][1]
    right_hip_y = points["right_hip"][1]
    nose_y = points["nose"][1]

    upper_y = nose_y
    lower_y = max(left_hip_y, right_hip_y)

    left_in_y_range = upper_y <= points["left_wrist"][1] <= lower_y
    right_in_y_range = upper_y <= points["right_wrist"][1] <= lower_y
    midpoint_in_y_range = upper_y <= wrist_midpoint_y <= lower_y
    is_detected = (
        wrist_distance <= shoulder_span * 0.55
        and left_in_y_range
        and right_in_y_range
        and midpoint_in_y_range
    )

    return GestureResult(
        is_detected=is_detected,
        debug_values={
            "shoulder_span": shoulder_span,
            "wrist_distance": wrist_distance,
            "wrist_distance_ratio": wrist_distance / max(shoulder_span, 1e-6),
            "wrist_midpoint_x": wrist_midpoint_x,
            "midpoint_y": wrist_midpoint_y,
            "upper_y": upper_y,
            "lower_y": lower_y,
            "left_in_y_range": float(left_in_y_range),
            "right_in_y_range": float(right_in_y_range),
            "midpoint_in_y_range": float(midpoint_in_y_range),
            "is_upper_lower_gathered": float(is_detected),
        },
    )


def compute_open_control_value(
    smoothed_landmarks, visibility_threshold: float = 0.6
) -> GestureResult:
    points = _get_pose_points(smoothed_landmarks)
    tracked_points = [
        points["left_shoulder"],
        points["right_shoulder"],
        points["left_wrist"],
        points["right_wrist"],
    ]
    if not _has_min_visibility(tracked_points, visibility_threshold):
        return GestureResult(is_detected=False)

    shoulder_span = _shoulder_span(points)
    wrist_span = _wrist_span(points)
    min_span = shoulder_span * 0.45
    max_span = shoulder_span * 2.2
    open_value = (wrist_span - min_span) / max(max_span - min_span, 1e-6)
    open_value = float(np.clip(open_value, 0.0, 1.0))

    return GestureResult(
        is_detected=True,
        debug_values={
            "open_value": open_value,
            "wrist_span_ratio": wrist_span / max(shoulder_span, 1e-6),
        },
    )


def is_open_release_pose(smoothed_landmarks, visibility_threshold: float = 0.6) -> bool:
    points = _get_pose_points(smoothed_landmarks)
    tracked_points = [
        points["left_shoulder"],
        points["right_shoulder"],
        points["left_wrist"],
        points["right_wrist"],
        points["left_hip"],
        points["right_hip"],
    ]
    if not _has_min_visibility(tracked_points, visibility_threshold):
        return True

    shoulder_y = (points["left_shoulder"][1] + points["right_shoulder"][1]) / 2.0
    hip_y = (points["left_hip"][1] + points["right_hip"][1]) / 2.0
    release_band_y = shoulder_y + (hip_y - shoulder_y) * 0.45
    return (
        points["left_wrist"][1] > release_band_y
        and points["right_wrist"][1] > release_band_y
    )


def detect_rise_trigger_pose(
    smoothed_landmarks, visibility_threshold: float = 0.6
) -> GestureResult:
    points = _get_pose_points(smoothed_landmarks)
    tracked_points = [
        points["left_shoulder"],
        points["right_shoulder"],
        points["left_elbow"],
        points["right_elbow"],
        points["left_wrist"],
        points["right_wrist"],
        points["left_hip"],
        points["right_hip"],
    ]
    if not _has_min_visibility(tracked_points, visibility_threshold):
        return GestureResult(is_detected=False)

    shoulder_y = (points["left_shoulder"][1] + points["right_shoulder"][1]) / 2.0
    hip_y = (points["left_hip"][1] + points["right_hip"][1]) / 2.0
    shoulder_center_x = (points["left_shoulder"][0] + points["right_shoulder"][0]) / 2.0
    shoulder_span = _shoulder_span(points)
    left_wrist_y = points["left_wrist"][1]
    right_wrist_y = points["right_wrist"][1]
    lower_band_max_y = shoulder_y + (hip_y - shoulder_y) * 0.7

    left_wrist_ready = shoulder_y <= left_wrist_y <= lower_band_max_y
    right_wrist_ready = shoulder_y <= right_wrist_y <= lower_band_max_y
    wrist_span = _wrist_span(points)
    left_is_side_ready = (
        left_wrist_ready
        and abs(points["left_wrist"][0] - shoulder_center_x) >= shoulder_span * 0.28
    )
    right_is_side_ready = (
        right_wrist_ready
        and abs(points["right_wrist"][0] - shoulder_center_x) >= shoulder_span * 0.28
    )
    ready_hand_count = int(left_is_side_ready) + int(right_is_side_ready)
    hands_not_grouped = wrist_span >= shoulder_span * 0.75
    is_detected = ready_hand_count == 1 and hands_not_grouped

    return GestureResult(
        is_detected=is_detected,
        debug_values={
            "ready_hand_count": float(ready_hand_count),
            "left_wrist_ready": float(left_wrist_ready),
            "right_wrist_ready": float(right_wrist_ready),
            "left_is_side_ready": float(left_is_side_ready),
            "right_is_side_ready": float(right_is_side_ready),
            "hands_not_grouped": float(hands_not_grouped),
            "wrist_span_ratio": wrist_span / max(shoulder_span, 1e-6),
        },
    )


def compute_rise_control_value(
    smoothed_landmarks, visibility_threshold: float = 0.6
) -> GestureResult:
    points = _get_pose_points(smoothed_landmarks)
    tracked_points = [
        points["nose"],
        points["left_shoulder"],
        points["right_shoulder"],
        points["left_wrist"],
        points["right_wrist"],
    ]
    if not _has_min_visibility(tracked_points, visibility_threshold):
        return GestureResult(is_detected=False)

    shoulder_y = (points["left_shoulder"][1] + points["right_shoulder"][1]) / 2.0
    nose_y = points["nose"][1]
    active_wrist_y = min(points["left_wrist"][1], points["right_wrist"][1])
    max_low_y = shoulder_y + abs(shoulder_y - nose_y) * 0.8
    min_high_y = nose_y - abs(shoulder_y - nose_y) * 0.5
    rise_value = (max_low_y - active_wrist_y) / max(max_low_y - min_high_y, 1e-6)
    rise_value = float(np.clip(rise_value, 0.0, 1.0))

    return GestureResult(
        is_detected=True,
        debug_values={"rise_value": rise_value},
    )


def update_gather_healing(
    open_control_state: ContinuousControlState,
    vad_state: VADState,
    trigger_result: GestureResult,
):
    event_name = None

    if open_control_state.phase == "gather_active":
        if not trigger_result.is_detected:
            event_name = "gather_active_ended"
            reset_control(open_control_state)
        else:
            vad_state.valence = move_toward_target(vad_state.valence, 0.60, 0.03)
            vad_state.dominance = move_toward_target(vad_state.dominance, 0.50, 0.02)

    gather_debug = GestureResult(
        is_detected=open_control_state.phase == "gather_active",
        debug_values={
            "phase": open_control_state.phase,
            "gather_active": float(open_control_state.phase == "gather_active"),
            "valence": vad_state.valence,
            "arousal": vad_state.arousal,
            "dominance": vad_state.dominance,
        },
    )
    return event_name, gather_debug


def update_open_control(
    smoothed_landmarks,
    frame_rgb: np.ndarray,
    open_control_state: ContinuousControlState,
    hand_estimator: HandEstimator,
    hold_duration: float = 1.0,
    grace_duration: float = 0.35,
):
    now = time.time()
    trigger_result = detect_open_trigger_pose(smoothed_landmarks)
    value_result = GestureResult(is_detected=False)
    hand_branch = open_control_state.branch
    event_name = None
    hand_status = "not_checked"
    release_requested = False

    if open_control_state.phase == "idle":
        if trigger_result.is_detected:
            open_control_state.phase = "arming"
            open_control_state.hold_started_at = now

    elif open_control_state.phase == "arming":
        if not trigger_result.is_detected:
            reset_control(open_control_state)
        elif open_control_state.hold_started_at is not None and (now - open_control_state.hold_started_at) >= hold_duration:
            hand_info = hand_estimator.process(frame_rgb)
            hand_branch = classify_open_branch(hand_info)
            hand_status = hand_branch
            open_control_state.branch = hand_branch
            if hand_branch == "open":
                open_control_state.phase = "controlling"
                open_control_state.active_started_at = now
                event_name = "open_control_started"
            elif hand_branch == "gather":
                open_control_state.phase = "gather_active"
                open_control_state.active_started_at = now
                event_name = "gather_active_started"
            else:
                reset_control(open_control_state)

    elif open_control_state.phase == "gather_active":
        if not trigger_result.is_detected:
            reset_control(open_control_state)

    if open_control_state.phase == "controlling":
        hand_branch = open_control_state.branch
        hand_status = hand_branch
        value_result = compute_open_control_value(smoothed_landmarks)
        time_since_active = 0.0
        if open_control_state.active_started_at is not None:
            time_since_active = now - open_control_state.active_started_at
        release_requested = is_open_release_pose(smoothed_landmarks)
        if time_since_active < grace_duration:
            release_requested = False
        if release_requested or not value_result.is_detected:
            event_name = event_name or "open_control_ended"
            reset_control(open_control_state)
            hand_branch = "missing"
            value_result = GestureResult(is_detected=False)
        else:
            open_control_state.value = float(value_result.debug_values["open_value"])
            open_control_state.branch = "open"

    hold_elapsed = 0.0
    if open_control_state.hold_started_at is not None:
        hold_elapsed = now - open_control_state.hold_started_at

    debug_values = {
        "hold_elapsed": hold_elapsed,
        "hold_duration_target": hold_duration,
        "hold_ready": float(hold_elapsed >= hold_duration),
        "grace_duration": grace_duration,
        "active_elapsed": (
            0.0
            if open_control_state.active_started_at is None
            else now - open_control_state.active_started_at
        ),
        "open_value": open_control_state.value,
        "branch": hand_branch,
        "hand_status": hand_status,
        "trigger_detected": float(trigger_result.is_detected),
        "release_requested": float(release_requested),
        "phase": open_control_state.phase,
    }
    debug_values.update(trigger_result.debug_values)
    debug_values.update(value_result.debug_values)
    return event_name, GestureResult(
        is_detected=open_control_state.phase in {"controlling", "gather_active"},
        debug_values=debug_values,
    )


def update_rise_control(
    smoothed_landmarks,
    rise_control_state: ContinuousControlState,
    hold_duration: float = 0.5,
):
    now = time.time()
    trigger_result = detect_rise_trigger_pose(smoothed_landmarks)
    value_result = GestureResult(is_detected=False)
    event_name = None

    if rise_control_state.phase == "idle":
        if trigger_result.is_detected:
            rise_control_state.phase = "arming"
            rise_control_state.hold_started_at = now

    elif rise_control_state.phase == "arming":
        if not trigger_result.is_detected:
            reset_control(rise_control_state)
        elif rise_control_state.hold_started_at is not None and (now - rise_control_state.hold_started_at) >= hold_duration:
            rise_control_state.phase = "controlling"
            event_name = "rise_control_started"

    if rise_control_state.phase == "controlling":
        value_result = compute_rise_control_value(smoothed_landmarks)
        if value_result.is_detected:
            rise_control_state.value = float(value_result.debug_values["rise_value"])
        else:
            event_name = event_name or "rise_control_ended"
            reset_control(rise_control_state)

    hold_elapsed = 0.0
    if rise_control_state.hold_started_at is not None:
        hold_elapsed = now - rise_control_state.hold_started_at

    debug_values = {
        "hold_elapsed": hold_elapsed,
        "rise_value": rise_control_state.value,
        "trigger_detected": float(trigger_result.is_detected),
    }
    debug_values.update(trigger_result.debug_values)
    debug_values.update(value_result.debug_values)
    return event_name, GestureResult(
        is_detected=rise_control_state.phase == "controlling",
        debug_values=debug_values,
    )


def detect_fold_pose(smoothed_landmarks, visibility_threshold: float = 0.6) -> GestureResult:
    points = _get_pose_points(smoothed_landmarks)
    tracked_points = [
        points["left_shoulder"],
        points["right_shoulder"],
        points["left_elbow"],
        points["right_elbow"],
        points["left_wrist"],
        points["right_wrist"],
        points["left_hip"],
        points["right_hip"],
    ]
    if not _has_min_visibility(tracked_points, visibility_threshold):
        return GestureResult(is_detected=False)

    shoulder_center_x = (points["left_shoulder"][0] + points["right_shoulder"][0]) / 2.0
    shoulder_center_y = (points["left_shoulder"][1] + points["right_shoulder"][1]) / 2.0
    hip_center_y = (points["left_hip"][1] + points["right_hip"][1]) / 2.0
    wrist_distance = np.linalg.norm(points["left_wrist"][:2] - points["right_wrist"][:2])
    shoulder_span = _shoulder_span(points)
    left_wrist_to_center = abs(points["left_wrist"][0] - shoulder_center_x)
    right_wrist_to_center = abs(points["right_wrist"][0] - shoulder_center_x)
    wrists_in_torso_band = (
        shoulder_center_y <= points["left_wrist"][1] <= hip_center_y
        and shoulder_center_y <= points["right_wrist"][1] <= hip_center_y
    )

    is_detected = (
        wrist_distance <= shoulder_span * 0.9
        and left_wrist_to_center <= shoulder_span * 0.6
        and right_wrist_to_center <= shoulder_span * 0.6
        and wrists_in_torso_band
    )
    return GestureResult(is_detected=is_detected)


def build_gesture_definitions() -> list[GestureDefinition]:
    return []


def evaluate_gestures(smoothed_landmarks, gesture_definitions, gesture_states):
    now = time.time()
    triggered_gestures = []
    gesture_results = {}

    for definition in gesture_definitions:
        result = definition.detector(smoothed_landmarks)
        gesture_results[definition.name] = result
        state = gesture_states[definition.name]
        if result.is_detected and (
            not state.is_active or (now - state.last_trigger_time) > definition.cooldown
        ):
            triggered_gestures.append(definition.name)
            state.is_active = True
            state.last_trigger_time = now
        elif not result.is_detected:
            state.is_active = False

    return triggered_gestures, gesture_results


def reset_gesture_states(gesture_states):
    for state in gesture_states.values():
        state.is_active = False


def reset_control(control_state: ContinuousControlState):
    control_state.phase = "idle"
    control_state.hold_started_at = None
    control_state.active_started_at = None
    control_state.value = 0.0
    control_state.branch = "missing"


def draw_debug_text(
    frame,
    gesture_results,
    open_control_state,
    open_control_result,
    gather_result,
    vad_state,
    rise_control_state,
    rise_control_result,
):
    active_names = [name for name, result in gesture_results.items() if result.is_detected]
    open_debug = open_control_result.debug_values
    wrist_distance_ratio = float(open_debug.get("wrist_distance_ratio", 0.0))
    hold_elapsed = float(open_debug.get("hold_elapsed", 0.0))
    hold_target = float(open_debug.get("hold_duration_target", 0.5))
    grace_duration = float(open_debug.get("grace_duration", 0.35))
    active_elapsed = float(open_debug.get("active_elapsed", 0.0))
    hand_status = str(open_debug.get("hand_status", open_control_state.branch))
    release_requested = bool(open_debug.get("release_requested", 0.0))
    trigger_detected = bool(open_debug.get("trigger_detected", 0.0))
    open_value = float(open_debug.get("open_value", open_control_state.value))
    left_in_y_range = bool(open_debug.get("left_in_y_range", 0.0))
    right_in_y_range = bool(open_debug.get("right_in_y_range", 0.0))
    midpoint_in_y_range = bool(open_debug.get("midpoint_in_y_range", 0.0))
    midpoint_y = float(open_debug.get("midpoint_y", 0.0))
    upper_y = float(open_debug.get("upper_y", 0.0))
    lower_y = float(open_debug.get("lower_y", 0.0))

    lines = [
        f"Open phase: {open_control_state.phase}",
        f"1. Same-place gather: {'YES' if trigger_detected else 'NO'}",
        f"   wrist/shoulder ratio: {wrist_distance_ratio:0.2f} (<= 0.55 target)",
        f"   left in height band:  {'YES' if left_in_y_range else 'NO'}",
        f"   right in height band: {'YES' if right_in_y_range else 'NO'}",
        f"   waist~eye band:       {'YES' if midpoint_in_y_range else 'NO'} ({midpoint_y:0.2f} in {upper_y:0.2f}..{lower_y:0.2f})",
        f"2. Hold ready: {'YES' if hold_elapsed >= hold_target else 'NO'}",
        f"   hold elapsed:         {hold_elapsed:0.2f}s / {hold_target:0.2f}s",
        f"3. Hand branch: {hand_status}",
        f"4. Grace:      {'YES' if active_elapsed < grace_duration else 'NO'} ({active_elapsed:0.2f}s / {grace_duration:0.2f}s)",
        f"5. Open value:  {open_value:0.2f}",
        f"6. Release req: {'YES' if release_requested else 'NO'}",
        f"Gather active: {'YES' if gather_result.is_detected else 'NO'}",
        f"VAD: V={vad_state.valence:0.2f} A={vad_state.arousal:0.2f} D={vad_state.dominance:0.2f}",
        f"Rise phase: {rise_control_state.phase}",
        f"Rise value: {float(rise_control_result.debug_values.get('rise_value', rise_control_state.value)):0.2f}",
        f"Active:     {', '.join(active_names) if active_names else '-'}",
    ]

    for name, result in gesture_results.items():
        lines.append(f"{name:<10} {'YES' if result.is_detected else 'NO'}")

    y = 30
    for line in lines:
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 0) if "YES" in line or line.startswith("3. Hand branch: open") else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 24


def draw_pose_landmarks(frame, pose_landmarks):
    frame_height, frame_width = frame.shape[:2]

    for start_idx, end_idx in POSE_CONNECTIONS:
        start = pose_landmarks[start_idx]
        end = pose_landmarks[end_idx]
        if start.visibility < 0.5 or end.visibility < 0.5:
            continue

        start_point = (int(start.x * frame_width), int(start.y * frame_height))
        end_point = (int(end.x * frame_width), int(end.y * frame_height))
        cv2.line(frame, start_point, end_point, (255, 120, 50), 2, cv2.LINE_AA)

    for landmark in pose_landmarks:
        if landmark.visibility < 0.5:
            continue
        point = (int(landmark.x * frame_width), int(landmark.y * frame_height))
        cv2.circle(frame, point, 2, (0, 255, 255), -1)


def draw_open_debug_overlay(frame, smoothed_landmarks, open_control_result):
    if smoothed_landmarks is None:
        return

    points = _get_pose_points(smoothed_landmarks)
    frame_height, frame_width = frame.shape[:2]
    open_debug = open_control_result.debug_values

    left_shoulder = points["left_shoulder"]
    right_shoulder = points["right_shoulder"]
    left_wrist = points["left_wrist"]
    right_wrist = points["right_wrist"]

    def to_pixel(point):
        return (int(point[0] * frame_width), int(point[1] * frame_height))

    left_shoulder_px = to_pixel(left_shoulder)
    right_shoulder_px = to_pixel(right_shoulder)
    left_wrist_px = to_pixel(left_wrist)
    right_wrist_px = to_pixel(right_wrist)
    midpoint_px = (
        int(((left_wrist[0] + right_wrist[0]) / 2.0) * frame_width),
        int(((left_wrist[1] + right_wrist[1]) / 2.0) * frame_height),
    )

    upper_y = float(open_debug.get("upper_y", 0.0))
    lower_y = float(open_debug.get("lower_y", 0.0))
    top_left = (0, int(upper_y * frame_height))
    bottom_right = (frame_width - 1, int(lower_y * frame_height))

    cv2.rectangle(frame, top_left, bottom_right, (80, 180, 255), 2)
    cv2.putText(
        frame,
        "OPEN HEIGHT BAND",
        (top_left[0], max(20, top_left[1] - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (80, 180, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.circle(frame, left_shoulder_px, 6, (255, 0, 0), -1)
    cv2.circle(frame, right_shoulder_px, 6, (255, 0, 0), -1)
    cv2.circle(frame, left_wrist_px, 6, (0, 255, 0), -1)
    cv2.circle(frame, right_wrist_px, 6, (0, 255, 0), -1)
    cv2.circle(frame, midpoint_px, 7, (0, 165, 255), -1)

    cv2.putText(frame, "LS", (left_shoulder_px[0] + 8, left_shoulder_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "RS", (right_shoulder_px[0] + 8, right_shoulder_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "LW", (left_wrist_px[0] + 8, left_wrist_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "RW", (right_wrist_px[0] + 8, right_wrist_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "MID", (midpoint_px[0] + 8, midpoint_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    smoother = LandmarkSmoother(window_size=5)
    hand_estimator = HandEstimator(HAND_MODEL_PATH, num_hands=2)
    pose_estimator = PoseEstimator(POSE_MODEL_PATH)
    vad_state = VADState()
    gesture_definitions = build_gesture_definitions()
    gesture_states = {
        definition.name: GestureRuntimeState() for definition in gesture_definitions
    }
    open_control_state = ContinuousControlState()
    rise_control_state = ContinuousControlState()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_estimator.process(rgb)

            gesture_results = {}
            open_control_result = GestureResult(is_detected=False)
            gather_result = GestureResult(is_detected=False)
            rise_control_result = GestureResult(is_detected=False)

            if results.pose_landmarks:
                pose_landmarks = results.pose_landmarks[0]
                draw_pose_landmarks(frame, pose_landmarks)
                smoothed_landmarks = smoother.update(pose_landmarks)

                open_event_name, open_control_result = update_open_control(
                    smoothed_landmarks,
                    rgb,
                    open_control_state,
                    hand_estimator,
                )
                if open_event_name:
                    trigger_unreal_effect(open_event_name)
                    if open_event_name.startswith("open_control"):
                        log_open_state(open_event_name, open_control_result)
                gather_event_name, gather_result = update_gather_healing(
                    open_control_state,
                    vad_state,
                    detect_open_trigger_pose(smoothed_landmarks),
                )
                if open_event_name and open_event_name.startswith("gather_active"):
                    log_gather_state(open_event_name, gather_result, vad_state)
                if gather_event_name:
                    trigger_unreal_effect(gather_event_name)
                    log_gather_state(gather_event_name, gather_result, vad_state)
                if open_control_state.phase == "controlling":
                    send_open_control_value(open_control_state.value)

                rise_event_name, rise_control_result = update_rise_control(
                    smoothed_landmarks,
                    rise_control_state,
                )
                if rise_event_name:
                    trigger_unreal_effect(rise_event_name)
                if rise_control_state.phase == "controlling":
                    send_rise_control_value(rise_control_state.value)

                draw_open_debug_overlay(frame, smoothed_landmarks, open_control_result)

                triggered_gestures, gesture_results = evaluate_gestures(
                    smoothed_landmarks, gesture_definitions, gesture_states
                )
                for gesture_name in triggered_gestures:
                    trigger_unreal_effect(gesture_name)
            else:
                smoother.update(None)
                reset_gesture_states(gesture_states)
                reset_control(open_control_state)
                reset_control(rise_control_state)

                draw_debug_text(
                    frame,
                    gesture_results,
                    open_control_state,
                    open_control_result,
                    gather_result,
                    vad_state,
                    rise_control_state,
                    rise_control_result,
                )

            cv2.imshow("Realtime Gesture To Unreal", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        hand_estimator.close()
        pose_estimator.close()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
