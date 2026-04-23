from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


@dataclass(slots=True)
class ConductFeatures:
    right_hand_visible: bool = False
    left_hand_visible: bool = False
    right_index_raised: bool = False
    right_fist_closed: bool = False
    right_fist_score: float = 0.0
    right_index_x: float = 0.0
    right_index_y: float = 0.0
    right_palm_ref: float = 0.0
    right_index_extension: float = 0.0
    right_middle_extension: float = 0.0
    right_ring_extension: float = 0.0
    right_pinky_extension: float = 0.0
    left_hand_open_valid: bool = False
    left_hand_open: float = 0.0
    left_fist_closed: bool = False
    left_fist_score: float = 0.0
    left_palm_ref: float = 0.0


def _hand_open_score(normalized_landmarks: dict[str, np.ndarray], side: str) -> tuple[bool, float, float]:
    required = (
        f"{side}_HAND_WRIST",
        f"{side}_HAND_THUMB_TIP",
        f"{side}_HAND_INDEX_MCP",
        f"{side}_HAND_INDEX_TIP",
        f"{side}_HAND_MIDDLE_TIP",
        f"{side}_HAND_RING_TIP",
        f"{side}_HAND_PINKY_MCP",
        f"{side}_HAND_PINKY_TIP",
    )
    if any(key not in normalized_landmarks for key in required):
        return (False, 0.0, 0.0)

    wrist = normalized_landmarks[f"{side}_HAND_WRIST"]
    thumb_tip = normalized_landmarks[f"{side}_HAND_THUMB_TIP"]
    index_mcp = normalized_landmarks[f"{side}_HAND_INDEX_MCP"]
    index_tip = normalized_landmarks[f"{side}_HAND_INDEX_TIP"]
    middle_tip = normalized_landmarks[f"{side}_HAND_MIDDLE_TIP"]
    ring_tip = normalized_landmarks[f"{side}_HAND_RING_TIP"]
    pinky_mcp = normalized_landmarks[f"{side}_HAND_PINKY_MCP"]
    pinky_tip = normalized_landmarks[f"{side}_HAND_PINKY_TIP"]

    palm_ref = max(_distance(index_mcp, pinky_mcp), 1e-5)
    spread_thumb_pinky = _distance(thumb_tip, pinky_tip) / palm_ref
    spread_fingers = _avg(
        [
            _distance(index_tip, wrist),
            _distance(middle_tip, wrist),
            _distance(ring_tip, wrist),
            _distance(pinky_tip, wrist),
        ]
    ) / palm_ref
    hand_open = 0.5 * spread_thumb_pinky + 0.5 * spread_fingers
    return (True, hand_open, palm_ref)


def _index_raise_score(normalized_landmarks: dict[str, np.ndarray], side: str) -> tuple[bool, float, float, float, float, float]:
    required = (
        f"{side}_HAND_INDEX_MCP",
        f"{side}_HAND_INDEX_TIP",
        f"{side}_HAND_MIDDLE_MCP",
        f"{side}_HAND_MIDDLE_TIP",
        f"{side}_HAND_RING_MCP",
        f"{side}_HAND_RING_TIP",
        f"{side}_HAND_PINKY_MCP",
        f"{side}_HAND_PINKY_TIP",
    )
    if any(key not in normalized_landmarks for key in required):
        return (False, 0.0, 0.0, 0.0, 0.0, 0.0)

    index_mcp = normalized_landmarks[f"{side}_HAND_INDEX_MCP"]
    index_tip = normalized_landmarks[f"{side}_HAND_INDEX_TIP"]
    middle_mcp = normalized_landmarks[f"{side}_HAND_MIDDLE_MCP"]
    middle_tip = normalized_landmarks[f"{side}_HAND_MIDDLE_TIP"]
    ring_mcp = normalized_landmarks[f"{side}_HAND_RING_MCP"]
    ring_tip = normalized_landmarks[f"{side}_HAND_RING_TIP"]
    pinky_mcp = normalized_landmarks[f"{side}_HAND_PINKY_MCP"]
    pinky_tip = normalized_landmarks[f"{side}_HAND_PINKY_TIP"]

    palm_ref = max(_distance(index_mcp, pinky_mcp), 1e-5)
    index_extension = _distance(index_tip, index_mcp) / palm_ref
    middle_extension = _distance(middle_tip, middle_mcp) / palm_ref
    ring_extension = _distance(ring_tip, ring_mcp) / palm_ref
    pinky_extension = _distance(pinky_tip, pinky_mcp) / palm_ref
    index_vertical = (index_mcp[1] - index_tip[1]) / palm_ref

    raised = bool(
        index_extension > 0.9
        and index_vertical > 0.15
        and index_extension > middle_extension * 1.05
        and index_extension > ring_extension * 1.08
    )
    return (
        raised,
        palm_ref,
        index_extension,
        middle_extension,
        ring_extension,
        pinky_extension,
    )


def _fist_closed_score(normalized_landmarks: dict[str, np.ndarray], side: str) -> tuple[bool, float, float]:
    required = (
        f"{side}_HAND_INDEX_MCP",
        f"{side}_HAND_INDEX_TIP",
        f"{side}_HAND_MIDDLE_MCP",
        f"{side}_HAND_MIDDLE_TIP",
        f"{side}_HAND_RING_MCP",
        f"{side}_HAND_RING_TIP",
        f"{side}_HAND_PINKY_MCP",
        f"{side}_HAND_PINKY_TIP",
        f"{side}_HAND_THUMB_TIP",
        f"{side}_HAND_WRIST",
    )
    if any(key not in normalized_landmarks for key in required):
        return (False, 0.0, 0.0)

    wrist = normalized_landmarks[f"{side}_HAND_WRIST"]
    thumb_tip = normalized_landmarks[f"{side}_HAND_THUMB_TIP"]
    index_mcp = normalized_landmarks[f"{side}_HAND_INDEX_MCP"]
    index_tip = normalized_landmarks[f"{side}_HAND_INDEX_TIP"]
    middle_mcp = normalized_landmarks[f"{side}_HAND_MIDDLE_MCP"]
    middle_tip = normalized_landmarks[f"{side}_HAND_MIDDLE_TIP"]
    ring_mcp = normalized_landmarks[f"{side}_HAND_RING_MCP"]
    ring_tip = normalized_landmarks[f"{side}_HAND_RING_TIP"]
    pinky_mcp = normalized_landmarks[f"{side}_HAND_PINKY_MCP"]
    pinky_tip = normalized_landmarks[f"{side}_HAND_PINKY_TIP"]

    palm_ref = max(_distance(index_mcp, pinky_mcp), 1e-5)
    index_extension = _distance(index_tip, index_mcp) / palm_ref
    middle_extension = _distance(middle_tip, middle_mcp) / palm_ref
    ring_extension = _distance(ring_tip, ring_mcp) / palm_ref
    pinky_extension = _distance(pinky_tip, pinky_mcp) / palm_ref
    thumb_to_wrist = _distance(thumb_tip, wrist) / palm_ref

    extensions = [index_extension, middle_extension, ring_extension, pinky_extension]
    folded_count = sum(1 for value in extensions if value < 0.86)
    compact_count = sum(1 for value in extensions if value < 1.02)
    avg_extension = _avg(extensions)
    extension_score = 1.0 - min(max((avg_extension - 0.58) / 0.62, 0.0), 1.0)
    folded_score = folded_count / 4.0
    thumb_score = 1.0 if thumb_to_wrist < 1.85 else 0.0
    fist_score = 0.55 * folded_score + 0.35 * extension_score + 0.10 * thumb_score

    closed = bool(
        (folded_count >= 3 and compact_count >= 4)
        or fist_score >= 0.62
    )
    return (closed, fist_score, palm_ref)


def compute_conduct_features(normalized_landmarks: dict[str, np.ndarray]) -> ConductFeatures:
    features = ConductFeatures()
    # The camera preview is mirrored before MediaPipe runs, so handedness is visually reversed.
    # For vol2 we remap semantic hands to match what the user sees on screen.
    control_side = "LEFT"   # user's right hand
    support_side = "RIGHT"  # user's left hand

    if f"{control_side}_HAND_INDEX_TIP" in normalized_landmarks:
        features.right_hand_visible = True
        right_index_tip = normalized_landmarks[f"{control_side}_HAND_INDEX_TIP"]
        features.right_index_x = float(right_index_tip[0])
        features.right_index_y = float(right_index_tip[1])

    (
        features.right_index_raised,
        features.right_palm_ref,
        features.right_index_extension,
        features.right_middle_extension,
        features.right_ring_extension,
        features.right_pinky_extension,
    ) = _index_raise_score(normalized_landmarks, control_side)
    features.right_fist_closed, features.right_fist_score, _ = _fist_closed_score(normalized_landmarks, control_side)

    (
        features.left_hand_open_valid,
        features.left_hand_open,
        features.left_palm_ref,
    ) = _hand_open_score(normalized_landmarks, support_side)
    features.left_fist_closed, features.left_fist_score, _ = _fist_closed_score(normalized_landmarks, support_side)
    features.left_hand_visible = bool(features.left_hand_open_valid)

    return features
