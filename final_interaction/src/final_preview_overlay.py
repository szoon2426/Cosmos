from __future__ import annotations

import cv2

try:
    from .final_hud_state import HudFrameState
except ImportError:
    from final_hud_state import HudFrameState

HAND_LANDMARK_LABELS = [
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_MCP",
    "INDEX_PIP",
    "INDEX_DIP",
    "INDEX_TIP",
    "MIDDLE_MCP",
    "MIDDLE_PIP",
    "MIDDLE_DIP",
    "MIDDLE_TIP",
    "RING_MCP",
    "RING_PIP",
    "RING_DIP",
    "RING_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
]

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)

POSE_CONNECTIONS = (
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"),
    ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP", "RIGHT_HIP"),
)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _point(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    return int(clamp(x, 0.0, 1.0) * width), int(clamp(y, 0.0, 1.0) * height)


def _draw_text(frame, text: str, origin: tuple[int, int], scale: float, color: tuple[int, int, int]) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _draw_bar(
    frame,
    *,
    label: str,
    value: float,
    origin: tuple[int, int],
    size: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    x, y = origin
    width, height = size
    cv2.rectangle(frame, (x, y), (x + width, y + height), (60, 60, 60), 1, cv2.LINE_AA)
    center = x + width // 2
    fill = int((width // 2) * clamp(abs(value), 0.0, 1.0))
    if value >= 0:
        cv2.rectangle(frame, (center, y + 2), (center + fill, y + height - 2), color, -1, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (center - fill, y + 2), (center, y + height - 2), color, -1, cv2.LINE_AA)
    cv2.line(frame, (center, y), (center, y + height), (160, 160, 160), 1, cv2.LINE_AA)
    _draw_text(frame, f"{label} {value:+.2f}", (x + width + 8, y + height - 3), 0.36, color)


def _draw_hand_skeleton(frame, hand_results, *, debug: bool) -> None:
    if not getattr(hand_results, "hand_landmarks", None):
        return

    height, width = frame.shape[:2]
    for hand_idx, hand in enumerate(hand_results.hand_landmarks):
        user_side = ""
        if hand_idx < len(hand_results.handedness):
            mp_side = hand_results.handedness[hand_idx][0].display_name.upper()
            if mp_side in ("LEFT", "RIGHT"):
                user_side = "RIGHT" if mp_side == "LEFT" else "LEFT"

        points = [_point(landmark.x, landmark.y, width, height) for landmark in hand]
        line_color = (90, 220, 255) if user_side == "RIGHT" else (210, 160, 255)
        dot_color = (245, 245, 245)
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], line_color, 2, cv2.LINE_AA)
        for landmark_idx, point in enumerate(points):
            cv2.circle(frame, point, 3, dot_color, -1, cv2.LINE_AA)
            if debug:
                label = f"{landmark_idx}:{HAND_LANDMARK_LABELS[landmark_idx]}"
                label_x = min(width - 170, point[0] + 6)
                label_y = max(14, point[1] - 4)
                _draw_text(frame, label, (label_x, label_y), 0.30, (255, 255, 255))

        if user_side:
            wrist = points[0]
            _draw_text(frame, f"HAND={user_side}", (min(width - 120, wrist[0] + 8), min(height - 10, wrist[1] + 18)), 0.42, line_color)


def _draw_pose_skeleton(frame, pose_landmarks: dict[str, tuple[float, float]] | None) -> None:
    if not pose_landmarks:
        return

    height, width = frame.shape[:2]
    for start, end in POSE_CONNECTIONS:
        if start not in pose_landmarks or end not in pose_landmarks:
            continue
        p1 = _point(pose_landmarks[start][0], pose_landmarks[start][1], width, height)
        p2 = _point(pose_landmarks[end][0], pose_landmarks[end][1], width, height)
        cv2.line(frame, p1, p2, (120, 255, 120), 2, cv2.LINE_AA)
        cv2.circle(frame, p1, 4, (120, 255, 120), -1, cv2.LINE_AA)
        cv2.circle(frame, p2, 4, (120, 255, 120), -1, cv2.LINE_AA)


def _draw_pointer(frame, state: HudFrameState) -> None:
    if not state.interaction_active or not state.both_visible:
        return
    if state.pointer_x < 0.0 or state.pointer_y < 0.0:
        return

    height, width = frame.shape[:2]
    cx, cy = _point(state.pointer_x, state.pointer_y, width, height)
    color = (70, 240, 160) if state.grab_strength >= 0.5 else (255, 210, 120)
    radius = max(10, int(18 + 0.5 * (state.left.palm_radius + state.right.palm_radius) * 28))
    cv2.circle(frame, (cx, cy), radius, color, 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), max(4, radius // 5), color, -1, cv2.LINE_AA)


def _draw_core_status(frame, state: HudFrameState, *, debug: bool) -> None:
    status_color = (70, 240, 160) if state.interaction_active else (180, 180, 180)
    world_label = state.world_id or "none"
    if state.world_number is not None:
        world_label = f"{world_label}#{state.world_number}"

    _draw_text(
        frame,
        f"{state.mode.upper()} active={state.interaction_active} world={world_label}",
        (20, 36),
        0.58,
        status_color,
    )
    _draw_bar(frame, label="V", value=state.target_v, origin=(20, 52), size=(120, 12), color=(255, 120, 190))
    _draw_bar(frame, label="A", value=state.target_a, origin=(20, 72), size=(120, 12), color=(80, 220, 255))
    _draw_bar(frame, label="D", value=state.target_d, origin=(20, 92), size=(120, 12), color=(130, 255, 130))

    hand_line = (
        f"L hand={state.left.hand_visible} vis={state.left.visible} open={state.left.open_strength:.2f} grab={state.left.grab_strength:.2f} "
        f"R hand={state.right.hand_visible} vis={state.right.visible} open={state.right.open_strength:.2f} grab={state.right.grab_strength:.2f}"
    )
    _draw_text(frame, hand_line, (20, 126), 0.42, (220, 245, 220))

    if debug:
        detail = (
            f"both_visible={state.both_visible} both_open={state.both_open} both_grab={state.both_grab} "
            f"LP={state.left_pointer_active} RP={state.right_pointer_active} "
            f"Lyz=({state.left_pointer_y:.1f},{state.left_pointer_z:.1f}) "
            f"Ryz=({state.right_pointer_y:.1f},{state.right_pointer_z:.1f})"
        )
        _draw_text(frame, detail, (20, 150), 0.38, (210, 210, 255))
        quality = (
            f"luma={state.luma_mean:.1f}/{state.luma_std:.1f} pre={state.preprocess_applied} "
            f"hands={state.hand_count} rescue={state.roi_rescue_count} "
            f"Lscore={state.left.handedness_score:.2f} Lage={state.left.fallback_age:.2f} "
            f"Lpalm={state.left.palm_radius:.2f} "
            f"Rscore={state.right.handedness_score:.2f} Rage={state.right.fallback_age:.2f} "
            f"Rpalm={state.right.palm_radius:.2f}"
        )
        _draw_text(frame, quality, (20, 174), 0.38, (255, 230, 180))
        gesture = (
            f"L-grab galaxy active={state.left_solo_grab_active} "
            f"hold={state.left_solo_grab_hold_progress * 100.0:.0f}% "
            f"restore={state.left_solo_vad_restore_active} "
            f"dx={state.left_solo_swipe_delta_x:+.2f} "
            f"vx={state.left_solo_swipe_velocity_x:+.2f} "
            f"ready={state.left_solo_world_move_ready} "
            f"fired={state.left_solo_world_move_fired}"
        )
        color = (120, 255, 150) if state.left_solo_vad_restore_active else (210, 210, 255)
        _draw_text(frame, gesture, (20, 198), 0.38, color)
        if state.camera_props:
            _draw_text(frame, state.camera_props, (20, 222), 0.34, (200, 230, 255))


def draw_preview_overlay(
    frame,
    *,
    hand_results,
    pose_landmarks: dict[str, tuple[float, float]] | None,
    state: HudFrameState,
    debug: bool = False,
) -> None:
    _draw_pose_skeleton(frame, pose_landmarks)
    _draw_hand_skeleton(frame, hand_results, debug=debug)
    _draw_pointer(frame, state)
    _draw_core_status(frame, state, debug=debug)
