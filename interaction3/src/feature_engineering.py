from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


def normalize_landmarks(landmarks: dict[str, tuple[float, float]]) -> dict[str, np.ndarray] | None:
    required = ("LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST")
    if any(name not in landmarks for name in required):
        return None

    left_shoulder = landmarks["LEFT_SHOULDER"]
    right_shoulder = landmarks["RIGHT_SHOULDER"]
    shoulder_center = (
        (left_shoulder[0] + right_shoulder[0]) * 0.5,
        (left_shoulder[1] + right_shoulder[1]) * 0.5,
    )
    shoulder_width = max(abs(right_shoulder[0] - left_shoulder[0]), 1e-5)

    normalized: dict[str, np.ndarray] = {}
    for name, (x, y) in landmarks.items():
        normalized[name] = np.array(
            [
                (x - shoulder_center[0]) / shoulder_width,
                (y - shoulder_center[1]) / shoulder_width,
            ],
            dtype=np.float32,
        )
    return normalized


def detect_snap_gate(normalized_landmarks: dict[str, np.ndarray]) -> dict[str, bool | str | float]:
    le = normalized_landmarks["LEFT_ELBOW"]
    re = normalized_landmarks["RIGHT_ELBOW"]
    lw = normalized_landmarks["LEFT_WRIST"]
    rw = normalized_landmarks["RIGHT_WRIST"]
    left_has_hand = all(
        key in normalized_landmarks
        for key in (
            "LEFT_HAND_THUMB_TIP",
            "LEFT_HAND_INDEX_TIP",
            "LEFT_HAND_INDEX_MCP",
            "LEFT_HAND_MIDDLE_TIP",
        )
    )
    right_has_hand = all(
        key in normalized_landmarks
        for key in (
            "RIGHT_HAND_THUMB_TIP",
            "RIGHT_HAND_INDEX_TIP",
            "RIGHT_HAND_INDEX_MCP",
            "RIGHT_HAND_MIDDLE_TIP",
        )
    )

    # Normalized coordinates are relative to shoulder center and scaled by shoulder width.
    # We keep this gate intentionally loose:
    # - one hand should be clearly more raised than the other
    # - the active hand should still be near the upper body center
    # - the corresponding hand landmarks must be visible
    left_above = lw[1] < 0.85 and le[1] < 1.0
    right_above = rw[1] < 0.85 and re[1] < 1.0
    left_near_center = abs(float(lw[0])) < 1.25
    right_near_center = abs(float(rw[0])) < 1.25
    left_clear_side = (rw[1] - lw[1]) > 0.22
    right_clear_side = (lw[1] - rw[1]) > 0.22

    left_thumb_middle_distance = 99.0
    right_thumb_middle_distance = 99.0
    if left_has_hand:
        left_thumb = normalized_landmarks["LEFT_HAND_THUMB_TIP"]
        left_middle = normalized_landmarks["LEFT_HAND_MIDDLE_TIP"]
        left_thumb_middle_distance = float(np.linalg.norm(left_thumb - left_middle))
    if right_has_hand:
        right_thumb = normalized_landmarks["RIGHT_HAND_THUMB_TIP"]
        right_middle = normalized_landmarks["RIGHT_HAND_MIDDLE_TIP"]
        right_thumb_middle_distance = float(np.linalg.norm(right_thumb - right_middle))

    thumb_middle_contact_threshold = 0.16
    left_prep_contact = left_thumb_middle_distance < thumb_middle_contact_threshold
    right_prep_contact = right_thumb_middle_distance < thumb_middle_contact_threshold

    left_gate = bool(left_has_hand and left_above and left_near_center and left_clear_side and left_prep_contact)
    right_gate = bool(right_has_hand and right_above and right_near_center and right_clear_side and right_prep_contact)

    active_side = "none"
    if left_gate:
        active_side = "left"
    elif right_gate:
        active_side = "right"

    return {
        "left_gate": left_gate,
        "right_gate": right_gate,
        "active_side": active_side,
        "gate_open": active_side != "none",
        "left_above": left_above,
        "right_above": right_above,
        "left_clear_side": left_clear_side,
        "right_clear_side": right_clear_side,
        "left_has_hand": left_has_hand,
        "right_has_hand": right_has_hand,
        "left_thumb_middle_distance": left_thumb_middle_distance,
        "right_thumb_middle_distance": right_thumb_middle_distance,
        "left_prep_contact": left_prep_contact,
        "right_prep_contact": right_prep_contact,
        "thumb_middle_contact_threshold": thumb_middle_contact_threshold,
    }


def detect_snap_event(
    normalized_landmarks: dict[str, np.ndarray],
    previous_landmarks: dict[str, np.ndarray] | None,
    active_side: str,
) -> dict[str, float | bool]:
    if active_side not in {"left", "right"}:
        return {
            "event": False,
            "thumb_index_distance": 0.0,
            "thumb_index_delta": 0.0,
            "thumb_middle_distance": 0.0,
            "thumb_middle_delta": 0.0,
            "thumb_crossed": False,
            "index_extension": 0.0,
            "middle_motion": 0.0,
        }

    side = active_side.upper()
    thumb_key = f"{side}_HAND_THUMB_TIP"
    index_key = f"{side}_HAND_INDEX_TIP"
    mcp_key = f"{side}_HAND_INDEX_MCP"
    middle_key = f"{side}_HAND_MIDDLE_TIP"

    if any(key not in normalized_landmarks for key in (thumb_key, index_key, mcp_key, middle_key)):
        return {
            "event": False,
            "thumb_index_distance": 0.0,
            "thumb_index_delta": 0.0,
            "thumb_middle_distance": 0.0,
            "thumb_middle_delta": 0.0,
            "thumb_crossed": False,
            "index_extension": 0.0,
            "middle_motion": 0.0,
        }

    thumb = normalized_landmarks[thumb_key]
    index = normalized_landmarks[index_key]
    index_mcp = normalized_landmarks[mcp_key]
    middle = normalized_landmarks[middle_key]

    thumb_index_distance = float(np.linalg.norm(thumb - index))
    thumb_middle_distance = float(np.linalg.norm(thumb - middle))
    thumb_crossed = bool((thumb[0] - index[0]) > 0.0)
    index_extension = float(np.linalg.norm(index - index_mcp))

    thumb_index_delta = 0.0
    thumb_middle_delta = 0.0
    middle_motion = 0.0
    prev_thumb_middle_distance = thumb_middle_distance
    if previous_landmarks is not None and all(
        key in previous_landmarks for key in (thumb_key, index_key, middle_key)
    ):
        prev_thumb = previous_landmarks[thumb_key]
        prev_index = previous_landmarks[index_key]
        prev_middle = previous_landmarks[middle_key]
        prev_thumb_index_distance = float(np.linalg.norm(prev_thumb - prev_index))
        prev_thumb_middle_distance = float(np.linalg.norm(prev_thumb - prev_middle))
        thumb_index_delta = thumb_index_distance - prev_thumb_index_distance
        thumb_middle_delta = thumb_middle_distance - prev_thumb_middle_distance
        middle_motion = float(np.linalg.norm(middle - prev_middle))

    # Snap event should stay simple:
    # - gate handles "one-arm prep pose + thumb-middle prep contact"
    # - event handles the actual thumb-index touch/closure moment inside that gate
    clear_index_contact = thumb_index_distance < 0.12
    fast_index_closing = thumb_index_delta < -0.04 and thumb_index_distance < 0.18
    middle_releasing = thumb_middle_delta > 0.02 or middle_motion > 0.03

    event = bool(
        clear_index_contact
        or fast_index_closing
        or (clear_index_contact and middle_releasing)
    )

    return {
        "event": event,
        "thumb_index_distance": thumb_index_distance,
        "thumb_index_delta": thumb_index_delta,
        "thumb_middle_distance": thumb_middle_distance,
        "thumb_middle_delta": thumb_middle_delta,
        "thumb_crossed": thumb_crossed,
        "index_extension": index_extension,
        "middle_motion": middle_motion,
    }


def _hand_features(
    normalized_landmarks: dict[str, np.ndarray],
    side: str,
    previous_landmarks: dict[str, np.ndarray] | None,
) -> list[float]:
    thumb_key = f"{side}_HAND_THUMB_TIP"
    index_key = f"{side}_HAND_INDEX_TIP"
    mcp_key = f"{side}_HAND_INDEX_MCP"

    if any(key not in normalized_landmarks for key in (thumb_key, index_key, mcp_key)):
        return [0.0] * 11

    thumb = normalized_landmarks[thumb_key]
    index = normalized_landmarks[index_key]
    index_mcp = normalized_landmarks[mcp_key]

    pinch_distance = float(np.linalg.norm(thumb - index))
    thumb_index_dx = float(thumb[0] - index[0])
    thumb_index_dy = float(thumb[1] - index[1])
    index_extension = float(np.linalg.norm(index - index_mcp))
    thumb_crossed = 1.0 if thumb_index_dx > 0.0 else 0.0

    features = [
        float(thumb[0]),
        float(thumb[1]),
        float(index[0]),
        float(index[1]),
        pinch_distance,
        thumb_index_dx,
        thumb_index_dy,
        index_extension,
        thumb_crossed,
    ]

    if previous_landmarks is None or any(key not in previous_landmarks for key in (thumb_key, index_key)):
        features.extend([0.0, 0.0])
    else:
        prev_thumb = previous_landmarks[thumb_key]
        prev_index = previous_landmarks[index_key]
        prev_pinch = float(np.linalg.norm(prev_thumb - prev_index))
        features.extend(
            [
                pinch_distance - prev_pinch,
                float((thumb[0] - prev_thumb[0]) - (index[0] - prev_index[0])),
            ]
        )
    return features


def compute_frame_features(
    normalized_landmarks: dict[str, np.ndarray],
    previous_landmarks: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    lw = normalized_landmarks["LEFT_WRIST"]
    rw = normalized_landmarks["RIGHT_WRIST"]
    le = normalized_landmarks["LEFT_ELBOW"]
    re = normalized_landmarks["RIGHT_ELBOW"]
    ls = normalized_landmarks["LEFT_SHOULDER"]
    rs = normalized_landmarks["RIGHT_SHOULDER"]

    l_idx = normalized_landmarks.get("LEFT_INDEX", lw)
    r_idx = normalized_landmarks.get("RIGHT_INDEX", rw)
    l_thb = normalized_landmarks.get("LEFT_THUMB", lw)
    r_thb = normalized_landmarks.get("RIGHT_THUMB", rw)
    snap_gate = detect_snap_gate(normalized_landmarks)

    wrist_distance = np.linalg.norm(lw - rw)
    wrist_height_mean = float((lw[1] + rw[1]) * 0.5)
    left_wrist_height = float(lw[1] - ls[1])
    right_wrist_height = float(rw[1] - rs[1])
    left_elbow_height = float(le[1] - ls[1])
    right_elbow_height = float(re[1] - rs[1])
    l_pinch = np.linalg.norm(l_idx - l_thb)
    r_pinch = np.linalg.norm(r_idx - r_thb)

    features = [
        lw[0], lw[1],
        rw[0], rw[1],
        le[0], le[1],
        re[0], re[1],
        wrist_distance,
        wrist_height_mean,
        left_wrist_height,
        right_wrist_height,
        left_elbow_height,
        right_elbow_height,
        float(l_pinch),
        float(r_pinch),
        1.0 if snap_gate["left_gate"] else 0.0,
        1.0 if snap_gate["right_gate"] else 0.0,
    ]

    active_side = str(snap_gate["active_side"])
    if active_side == "left":
        features.extend(_hand_features(normalized_landmarks, "LEFT", previous_landmarks))
    elif active_side == "right":
        features.extend(_hand_features(normalized_landmarks, "RIGHT", previous_landmarks))
    else:
        features.extend([0.0] * 11)

    if previous_landmarks is None:
        features.extend([0.0] * 12)
    else:
        prev_lw = previous_landmarks["LEFT_WRIST"]
        prev_rw = previous_landmarks["RIGHT_WRIST"]
        prev_le = previous_landmarks["LEFT_ELBOW"]
        prev_re = previous_landmarks["RIGHT_ELBOW"]
        prev_l_idx = previous_landmarks.get("LEFT_INDEX", prev_lw)
        prev_r_idx = previous_landmarks.get("RIGHT_INDEX", prev_rw)
        features.extend(
            [
                *(lw - prev_lw),
                *(rw - prev_rw),
                *(le - prev_le),
                *(re - prev_re),
                *(l_idx - prev_l_idx),
                *(r_idx - prev_r_idx),
            ]
        )

    return np.array(features, dtype=np.float32)


def compute_snap_frame_features(
    normalized_landmarks: dict[str, np.ndarray],
    previous_landmarks: dict[str, np.ndarray] | None = None,
    active_side: str = "none",
) -> np.ndarray:
    if active_side not in {"left", "right"}:
        return np.zeros(18, dtype=np.float32)

    side = active_side.upper()
    thumb_key = f"{side}_HAND_THUMB_TIP"
    index_key = f"{side}_HAND_INDEX_TIP"
    middle_key = f"{side}_HAND_MIDDLE_TIP"
    mcp_key = f"{side}_HAND_INDEX_MCP"
    wrist_key = f"{side}_WRIST" if f"{side}_WRIST" in normalized_landmarks else f"{side}_HAND_WRIST"

    required = (wrist_key, thumb_key, index_key, middle_key, mcp_key)
    if any(key not in normalized_landmarks for key in required):
        return np.zeros(18, dtype=np.float32)

    wrist = normalized_landmarks[wrist_key]
    thumb = normalized_landmarks[thumb_key]
    index = normalized_landmarks[index_key]
    middle = normalized_landmarks[middle_key]
    index_mcp = normalized_landmarks[mcp_key]

    thumb_index_distance = float(np.linalg.norm(thumb - index))
    thumb_middle_distance = float(np.linalg.norm(thumb - middle))
    index_extension = float(np.linalg.norm(index - index_mcp))
    thumb_index_dx = float(thumb[0] - index[0])
    thumb_index_dy = float(thumb[1] - index[1])
    thumb_middle_dx = float(thumb[0] - middle[0])
    thumb_middle_dy = float(thumb[1] - middle[1])
    index_middle_distance = float(np.linalg.norm(index - middle))
    index_middle_dx = float(index[0] - middle[0])
    index_middle_dy = float(index[1] - middle[1])
    thumb_crossed = 1.0 if thumb_index_dx > 0.0 else 0.0

    thumb_index_delta = 0.0
    thumb_middle_delta = 0.0
    middle_motion = 0.0
    thumb_motion = 0.0
    index_motion = 0.0
    wrist_relative_thumb = float(np.linalg.norm(thumb - wrist))
    wrist_relative_index = float(np.linalg.norm(index - wrist))
    if previous_landmarks is not None and all(key in previous_landmarks for key in required):
        prev_thumb = previous_landmarks[thumb_key]
        prev_index = previous_landmarks[index_key]
        prev_middle = previous_landmarks[middle_key]
        prev_thumb_index_distance = float(np.linalg.norm(prev_thumb - prev_index))
        prev_thumb_middle_distance = float(np.linalg.norm(prev_thumb - prev_middle))
        thumb_index_delta = thumb_index_distance - prev_thumb_index_distance
        thumb_middle_delta = thumb_middle_distance - prev_thumb_middle_distance
        middle_motion = float(np.linalg.norm(middle - prev_middle))
        thumb_motion = float(np.linalg.norm(thumb - prev_thumb))
        index_motion = float(np.linalg.norm(index - prev_index))

    return np.array(
        [
            thumb_index_distance,
            thumb_middle_distance,
            thumb_index_delta,
            thumb_middle_delta,
            middle_motion,
            thumb_motion,
            index_motion,
            index_extension,
            thumb_index_dx,
            thumb_index_dy,
            thumb_middle_dx,
            thumb_middle_dy,
            index_middle_distance,
            index_middle_dx,
            index_middle_dy,
            thumb_crossed,
            wrist_relative_thumb,
            wrist_relative_index,
        ],
        dtype=np.float32,
    )


@dataclass
class SlidingWindowBuffer:
    window_size: int = 10
    items: deque[np.ndarray] = field(default_factory=deque)

    def add(self, item: np.ndarray) -> None:
        self.items.append(item)
        while len(self.items) > self.window_size:
            self.items.popleft()

    def clear(self) -> None:
        self.items.clear()

    def is_ready(self) -> bool:
        return len(self.items) >= self.window_size

    def to_feature_vector(self) -> np.ndarray:
        if not self.is_ready():
            raise ValueError("Sliding window is not ready")
        return np.concatenate(list(self.items), axis=0).astype(np.float32)
