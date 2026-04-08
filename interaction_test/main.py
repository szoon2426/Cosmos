import argparse
import math
import time
from dataclasses import dataclass

import cv2

from src.capture import WebcamCapture
from src.gesture import InteractionEngine, InteractionSignals
from src.hand import HandEstimator
from src.pose import POSE_CONNECTIONS, PoseEstimator
from src.vad_mapper import compute_unreal_payload


VISIBILITY_THRESHOLD = 0.45


@dataclass
class Mission:
    label: str
    duration: float = 0.0


MISSIONS = [
    Mission("1. open 열어서 늘리기"),
    Mission("2. open 열어서 줄이기"),
    Mission("3. rise 올리기"),
    Mission("4. rise 줄이기"),
    Mission("5. gather 5초 유지", duration=5.0),
    Mission("6. deep breath 5초 유지", duration=5.0),
]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def remap(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    if in_max == in_min:
        return out_min
    t = (value - in_min) / (in_max - in_min)
    return out_min + (out_max - out_min) * t


def angle_deg(a: dict | None, b: dict | None, c: dict | None) -> float | None:
    if a is None or b is None or c is None:
        return None

    bax = a["nx"] - b["nx"]
    bay = a["ny"] - b["ny"]
    bcx = c["nx"] - b["nx"]
    bcy = c["ny"] - b["ny"]

    mag_ba = (bax * bax + bay * bay) ** 0.5
    mag_bc = (bcx * bcx + bcy * bcy) ** 0.5
    if mag_ba <= 1e-6 or mag_bc <= 1e-6:
        return None

    dot = bax * bcx + bay * bcy
    cos_theta = clamp(dot / (mag_ba * mag_bc), -1.0, 1.0)
    return float(math.degrees(math.acos(cos_theta)))


def get_landmark(landmarks: list[dict] | None, name: str) -> dict | None:
    if not landmarks:
        return None
    for item in landmarks:
        if item["name"] == name and item.get("visibility", 1.0) >= VISIBILITY_THRESHOLD:
            return item
    return None


def classify_open_branch(hand_info: dict) -> str:
    left = hand_info.get("Left")
    right = hand_info.get("Right")

    def state(hand: dict | None) -> str:
        if hand is None:
            return "missing"
        if hand.get("fist"):
            return "fist"
        if hand.get("open"):
            return "open"
        return "missing"

    left_state = state(left)
    right_state = state(right)
    pair = {left_state, right_state}

    if pair == {"fist", "missing"} or pair == {"fist"}:
        return "gather"
    if pair == {"open", "missing"} or pair == {"open"}:
        return "open"
    return "missing"


def build_webcam_signals(dt: float, landmarks: list[dict] | None, hand_info: dict) -> tuple[InteractionSignals, dict]:
    left_wrist = get_landmark(landmarks, "LEFT_WRIST")
    right_wrist = get_landmark(landmarks, "RIGHT_WRIST")
    left_shoulder = get_landmark(landmarks, "LEFT_SHOULDER")
    right_shoulder = get_landmark(landmarks, "RIGHT_SHOULDER")
    left_hip = get_landmark(landmarks, "LEFT_HIP")
    right_hip = get_landmark(landmarks, "RIGHT_HIP")
    left_elbow = get_landmark(landmarks, "LEFT_ELBOW")
    right_elbow = get_landmark(landmarks, "RIGHT_ELBOW")
    left_ankle = get_landmark(landmarks, "LEFT_ANKLE")
    right_ankle = get_landmark(landmarks, "RIGHT_ANKLE")
    nose = get_landmark(landmarks, "NOSE")
    left_eye = get_landmark(landmarks, "LEFT_EYE")
    right_eye = get_landmark(landmarks, "RIGHT_EYE")
    mouth_left = get_landmark(landmarks, "MOUTH_LEFT")
    mouth_right = get_landmark(landmarks, "MOUTH_RIGHT")

    debug = {
        "open_ready": False,
        "open_branch": "missing",
        "open_spread": 0.0,
        "open_mid_y": 0.0,
        "open_upper_band": False,
        "open_close_enough": False,
        "open_release": False,
        "rise_ready": False,
        "rise_left_ready": False,
        "rise_right_ready": False,
        "rise_face_front": False,
        "rise_side": "none",
        "rise_left_in_band": False,
        "rise_right_in_band": False,
        "rise_left_inside": False,
        "rise_right_inside": False,
        "rise_left_other_low": False,
        "rise_right_other_low": False,
        "rise_left_higher": False,
        "rise_right_higher": False,
        "rise_height": 0.0,
        "gather_ready": False,
        "breath_ready": False,
        "breath_left_armpit": None,
        "breath_right_armpit": None,
        "breath_left_armpit_state": "missing",
        "breath_right_armpit_state": "missing",
        "breath_left_elbow": None,
        "breath_right_elbow": None,
        "breath_left_bent": False,
        "breath_right_bent": False,
    }

    if not all([left_wrist, right_wrist, left_shoulder, right_shoulder, left_hip, right_hip]):
        return InteractionSignals(dt=dt), debug

    shoulder_width = max(0.05, abs(right_shoulder["nx"] - left_shoulder["nx"]))
    shoulder_y = (left_shoulder["ny"] + right_shoulder["ny"]) / 2.0
    hip_y = (left_hip["ny"] + right_hip["ny"]) / 2.0
    torso_height = max(0.08, hip_y - shoulder_y)
    eye_y = min(
        item["ny"]
        for item in (left_eye, right_eye, nose)
        if item is not None
    ) if any((left_eye, right_eye, nose)) else shoulder_y - torso_height * 0.6
    mouth_y = max(
        item["ny"]
        for item in (mouth_left, mouth_right)
        if item is not None
    ) if any((mouth_left, mouth_right)) else shoulder_y + torso_height * 0.15

    wrist_mid_y = (left_wrist["ny"] + right_wrist["ny"]) / 2.0
    wrist_dist_ratio = abs(right_wrist["nx"] - left_wrist["nx"]) / shoulder_width
    in_upper_band = eye_y <= wrist_mid_y <= hip_y

    open_close_enough = wrist_dist_ratio <= 0.70
    open_ready = in_upper_band and open_close_enough
    open_branch = classify_open_branch(hand_info) if open_ready else "missing"
    open_spread = clamp(remap(wrist_dist_ratio, 0.25, 1.75, -1.0, 1.0), -1.0, 1.0)
    open_mid_y = wrist_mid_y
    open_release = False

    face_front = (
        nose is not None
        and left_shoulder["nx"] - shoulder_width * 0.15 <= nose["nx"] <= right_shoulder["nx"] + shoulder_width * 0.15
    )

    shoulder_min_x = min(left_shoulder["nx"], right_shoulder["nx"])
    shoulder_max_x = max(left_shoulder["nx"], right_shoulder["nx"])
    shoulder_inner_margin = shoulder_width * 0.08
    left_inside_shoulders = (
        shoulder_min_x - shoulder_inner_margin
        <= left_wrist["nx"]
        <= shoulder_max_x + shoulder_inner_margin
    )
    right_inside_shoulders = (
        shoulder_min_x - shoulder_inner_margin
        <= right_wrist["nx"]
        <= shoulder_max_x + shoulder_inner_margin
    )

    rise_band_top = mouth_y - torso_height * 0.06
    rise_band_bottom = hip_y + torso_height * 0.02
    left_in_rise_band = rise_band_top <= left_wrist["ny"] <= rise_band_bottom
    right_in_rise_band = rise_band_top <= right_wrist["ny"] <= rise_band_bottom

    left_low = left_wrist["ny"] >= shoulder_y + torso_height * 0.45
    right_low = right_wrist["ny"] >= shoulder_y + torso_height * 0.45
    left_higher = left_wrist["ny"] < right_wrist["ny"] - torso_height * 0.01
    right_higher = right_wrist["ny"] < left_wrist["ny"] - torso_height * 0.01

    left_rise_ready = (
        left_in_rise_band
        and left_inside_shoulders
        and (right_low or left_higher)
    )
    right_rise_ready = (
        right_in_rise_band
        and right_inside_shoulders
        and (left_low or right_higher)
    )
    rise_ready = left_rise_ready ^ right_rise_ready

    if left_rise_ready:
        active_wrist = left_wrist
        active_side = "left"
    elif right_rise_ready:
        active_wrist = right_wrist
        active_side = "right"
    else:
        active_wrist = right_wrist if right_wrist["ny"] < left_wrist["ny"] else left_wrist
        active_side = "right" if active_wrist is right_wrist else "left"

    rise_height = clamp(remap(active_wrist["ny"], hip_y, mouth_y, -1.0, 1.0), -1.0, 1.0)
    rise_release = (
        active_wrist["nx"] < shoulder_min_x - shoulder_width * 0.40
        or active_wrist["nx"] > shoulder_max_x + shoulder_width * 0.40
        or active_wrist["ny"] > hip_y + torso_height * 0.18
    )

    any_fist = any(hand is not None and hand.get("fist") for hand in hand_info.values())
    gather_ready = open_ready and any_fist
    gather_release = not gather_ready

    left_armpit_angle = angle_deg(left_hip, left_shoulder, left_elbow)
    right_armpit_angle = angle_deg(right_hip, right_shoulder, right_elbow)
    left_elbow_angle = angle_deg(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = angle_deg(right_shoulder, right_elbow, right_wrist)

    def armpit_state(angle: float | None) -> str:
        if angle is None:
            return "missing"
        if angle < 15.0:
            return "small"
        if angle > 70.0:
            return "large"
        return "ok"

    left_armpit_state = armpit_state(left_armpit_angle)
    right_armpit_state = armpit_state(right_armpit_angle)
    left_armpit_ok = left_armpit_state == "ok"
    right_armpit_ok = right_armpit_state == "ok"

    left_bent = left_elbow_angle is not None and 110.0 <= left_elbow_angle <= 172.0
    right_bent = right_elbow_angle is not None and 110.0 <= right_elbow_angle <= 172.0

    breath_ready = left_armpit_ok and right_armpit_ok and left_bent and right_bent
    breath_release = not breath_ready

    debug.update(
        {
            "open_ready": open_ready,
            "open_branch": open_branch,
            "open_spread": open_spread,
            "open_mid_y": open_mid_y,
            "open_upper_band": in_upper_band,
            "open_close_enough": open_close_enough,
            "open_release": open_release,
            "rise_ready": rise_ready,
            "rise_left_ready": left_rise_ready,
            "rise_right_ready": right_rise_ready,
            "rise_face_front": face_front,
            "rise_side": active_side,
            "rise_left_in_band": left_in_rise_band,
            "rise_right_in_band": right_in_rise_band,
            "rise_left_inside": left_inside_shoulders,
            "rise_right_inside": right_inside_shoulders,
            "rise_left_other_low": right_low,
            "rise_right_other_low": left_low,
            "rise_left_higher": left_higher,
            "rise_right_higher": right_higher,
            "rise_height": rise_height,
            "gather_ready": gather_ready,
            "breath_ready": breath_ready,
            "breath_left_armpit": left_armpit_angle,
            "breath_right_armpit": right_armpit_angle,
            "breath_left_armpit_state": left_armpit_state,
            "breath_right_armpit_state": right_armpit_state,
            "breath_left_elbow": left_elbow_angle,
            "breath_right_elbow": right_elbow_angle,
            "breath_left_bent": left_bent,
            "breath_right_bent": right_bent,
        }
    )

    return (
        InteractionSignals(
            dt=dt,
            open_ready=open_ready,
            open_branch=open_branch,
            open_spread=open_spread,
            open_mid_y=open_mid_y,
            open_drop_margin=torso_height * 0.28,
            open_release=open_release,
            rise_ready=rise_ready,
            rise_height=rise_height,
            rise_release=rise_release,
            gather_ready=gather_ready,
            gather_release=gather_release,
            breath_ready=breath_ready,
            breath_release=breath_release,
        ),
        debug,
    )


def evaluate_mission_condition(index: int, engine: InteractionEngine) -> bool:
    if index == 0:
        return engine.interaction.open.phase == "active" and engine.interaction.open.amount >= 0.6
    if index == 1:
        return engine.interaction.open.phase == "active" and engine.interaction.open.amount <= -0.6
    if index == 2:
        return engine.interaction.rise.phase == "active" and engine.interaction.rise.amount >= 0.6
    if index == 3:
        return engine.interaction.rise.phase == "active" and engine.interaction.rise.amount <= -0.6
    if index == 4:
        return engine.interaction.gather.phase == "active"
    if index == 5:
        return engine.interaction.deep_breath.phase == "active"
    return False


def update_mission_progress(
    mission_index: int,
    hold_elapsed: float,
    dt: float,
    engine: InteractionEngine,
) -> tuple[int, float, str | None]:
    if mission_index >= len(MISSIONS):
        return mission_index, hold_elapsed, None

    mission = MISSIONS[mission_index]
    if evaluate_mission_condition(mission_index, engine):
        hold_elapsed += dt
    else:
        hold_elapsed = 0.0

    required = mission.duration if mission.duration > 0.0 else 0.15
    if hold_elapsed >= required:
        return mission_index + 1, 0.0, mission.label

    return mission_index, hold_elapsed, None


def draw_pose(frame, landmarks: list[dict] | None) -> None:
    if not landmarks:
        return

    points = {item["name"]: (int(item["x"]), int(item["y"])) for item in landmarks}
    name_by_index = {i: name for i, name in enumerate([item["name"] for item in landmarks])}

    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx >= len(landmarks) or end_idx >= len(landmarks):
            continue
        start_name = name_by_index[start_idx]
        end_name = name_by_index[end_idx]
        if start_name in points and end_name in points:
            cv2.line(frame, points[start_name], points[end_name], (80, 160, 255), 2)

    for key in ("LEFT_WRIST", "RIGHT_WRIST", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"):
        if key in points:
            cv2.circle(frame, points[key], 6, (40, 255, 120), -1)


def draw_hand_info(frame, hand_info: dict) -> None:
    for side, hand in hand_info.items():
        if hand is None:
            continue
        x, y = hand["wrist"]
        px = int(x * frame.shape[1])
        py = int(y * frame.shape[0])
        label = f"{side}:"
        if hand.get("fist"):
            label += " fist"
        elif hand.get("open"):
            label += " open"
        else:
            label += " unknown"
        cv2.putText(frame, label, (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 5, (255, 255, 0), -1)


def draw_debug(
    frame,
    engine: InteractionEngine,
    target,
    payload: dict,
    debug: dict,
    events: list[str],
    mission_index: int,
    mission_hold: float,
) -> None:
    if mission_index < len(MISSIONS):
        mission = MISSIONS[mission_index]
        mission_text = mission.label
        if mission.duration > 0.0:
            mission_text += f" ({mission_hold:.1f}/{mission.duration:.1f}s)"
    else:
        mission_text = "모든 미션 완료"

    lines = [
        f"mission: {mission_text}",
        f"target V {target.valence:+.2f}  A {target.arousal:+.2f}  D {target.dominance:+.2f}",
        f"payload V {payload['V']:+.2f}  A {payload['A']:+.2f}  D {payload['D']:+.2f}",
        f"open  ready={debug['open_ready']} end={debug['open_release']} band={debug['open_upper_band']} close={debug['open_close_enough']} branch={debug['open_branch']} amount={engine.interaction.open.amount:+.2f}",
        f"rise  ready={debug['rise_ready']} L={debug['rise_left_ready']} R={debug['rise_right_ready']} side={debug['rise_side']} amount={engine.interaction.rise.amount:+.2f}",
        f"      inBand L={debug['rise_left_in_band']} R={debug['rise_right_in_band']} inside L={debug['rise_left_inside']} R={debug['rise_right_inside']}",
        f"      otherLow L={debug['rise_left_other_low']} R={debug['rise_right_other_low']} higher L={debug['rise_left_higher']} R={debug['rise_right_higher']}",
        f"gather ready={debug['gather_ready']} active={engine.interaction.gather.phase == 'active'}",
        f"breath ready={debug['breath_ready']} active={engine.interaction.deep_breath.phase == 'active'}",
        (
            "       "
            f"Lpit={debug['breath_left_armpit'] if debug['breath_left_armpit'] is not None else 'NA'}({debug['breath_left_armpit_state']}) "
            f"Rpit={debug['breath_right_armpit'] if debug['breath_right_armpit'] is not None else 'NA'}({debug['breath_right_armpit_state']})"
        ),
        (
            "       "
            f"Lelbow={debug['breath_left_elbow'] if debug['breath_left_elbow'] is not None else 'NA'} bent={debug['breath_left_bent']} "
            f"Relbow={debug['breath_right_elbow'] if debug['breath_right_elbow'] is not None else 'NA'} bent={debug['breath_right_bent']}"
        ),
        f"density={payload['density']:.3f} decay={payload['decay']:.3f} min={payload['min_speed']:.1f} max={payload['max_speed']:.1f}",
        f"events={events if events else []}",
        "q or ESC: quit",
    ]

    y = 28
    for line in lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        y += 24

    badge_x_left = max(40, frame.shape[1] - 360)
    badge_x_right = max(200, frame.shape[1] - 190)
    badges = [
        ("OPEN READY", debug["open_ready"], (badge_x_left, 170)),
        ("OPEN ACTIVE", engine.interaction.open.phase == "active", (badge_x_right, 170)),
        ("RISE READY", debug["rise_ready"], (badge_x_left, 210)),
        ("RISE ACTIVE", engine.interaction.rise.phase == "active", (badge_x_right, 210)),
        ("GATHER", engine.interaction.gather.phase == "active", (badge_x_left, 250)),
        ("BREATH", engine.interaction.deep_breath.phase == "active", (badge_x_right, 250)),
    ]

    for label, is_on, (x, y0) in badges:
        color = (60, 200, 60) if is_on else (70, 70, 70)
        cv2.rectangle(frame, (x, y0), (x + 150, y0 + 28), color, -1)
        cv2.putText(frame, label, (x + 8, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (15, 15, 15), 1, cv2.LINE_AA)


def parse_args():
    parser = argparse.ArgumentParser(description="interaction_test webcam debug")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to open")
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Probe a few camera indices and print which ones open",
    )
    return parser.parse_args()


def list_cameras(max_index: int = 6) -> None:
    print("[interaction_test] available camera probe")
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)
        opened = cap.isOpened()
        if opened:
            ok, _ = cap.read()
            status = "ok" if ok else "opened-no-frame"
        else:
            status = "unavailable"
        print(f"  camera {idx}: {status}")
        cap.release()


def main() -> None:
    args = parse_args()
    if args.list_cameras:
        list_cameras()
        return

    capture = WebcamCapture(camera_index=args.camera, width=1280, height=720)
    pose_estimator = PoseEstimator()
    hand_estimator = HandEstimator()
    engine = InteractionEngine()
    engine.open_hold_duration = 1.0
    engine.rise_hold_duration = 0.8

    capture.open()
    previous = time.time()
    last_events: list[str] = []
    mission_index = 0
    mission_hold = 0.0

    print(f"[interaction_test] webcam interaction debug | camera={args.camera}")
    print("OpenCV window will show live values. Press q or ESC to quit.")

    try:
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            now = time.time()
            dt = max(0.001, now - previous)
            previous = now

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose_estimator.process(frame_rgb)
            hand_info = hand_estimator.process(frame_rgb)
            h, w = frame.shape[:2]
            landmarks = pose_estimator.get_landmarks_as_dict(pose_results, w, h)

            signals, debug = build_webcam_signals(dt, landmarks, hand_info)
            events = engine.update(signals)
            target_emotion = engine.get_target_emotion()
            payload = compute_unreal_payload(engine.emotion).as_dict()
            mission_index, mission_hold, completed = update_mission_progress(
                mission_index, mission_hold, dt, engine
            )

            if events.triggered:
                last_events = events.triggered
                print(
                    "[webcam-log] "
                    f"target=({target_emotion.valence:+.2f}, {target_emotion.arousal:+.2f}, {target_emotion.dominance:+.2f}) "
                    f"payload=({payload['V']:+.2f}, {payload['A']:+.2f}, {payload['D']:+.2f}) "
                    f"events={events.triggered}"
                )
            if completed is not None:
                print(f"[mission] completed: {completed}")
                last_events = [f"mission complete: {completed}"]

            draw_pose(frame, landmarks)
            draw_hand_info(frame, hand_info)
            draw_debug(
                frame,
                engine,
                target_emotion,
                payload,
                debug,
                last_events,
                mission_index,
                mission_hold,
            )

            cv2.imshow("interaction_test webcam", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
    finally:
        capture.release()
        pose_estimator.close()
        hand_estimator.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
