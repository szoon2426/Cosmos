from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterable

import cv2
import numpy as np


@dataclass(slots=True)
class RescueCandidate:
    expected_side: str
    center: tuple[float, float]
    source: str


@dataclass(slots=True)
class RoiRescueConfig:
    enabled: bool = True
    size_fraction: float = 0.28
    scale: float = 2.5
    min_accept_score: float = 0.25
    mismatch_min_score: float = 0.55
    max_candidate_distance: float = 0.16
    duplicate_distance: float = 0.09


@dataclass(slots=True)
class RoiRescueStats:
    candidates: int = 0
    attempted: int = 0
    added: int = 0


@dataclass(slots=True)
class CropRect:
    x: int
    y: int
    width: int
    height: int


class HandResultAdapter:
    def __init__(self, hand_landmarks: list, handedness: list, hand_world_landmarks: list | None = None) -> None:
        self.hand_landmarks = hand_landmarks
        self.handedness = handedness
        self.hand_world_landmarks = hand_world_landmarks or [[] for _ in hand_landmarks]


def hand_count(results) -> int:
    return len(getattr(results, "hand_landmarks", None) or [])


def category_side(category) -> str:
    return str(getattr(category, "display_name", "") or getattr(category, "category_name", "")).upper()


def category_score(category) -> float:
    try:
        return float(getattr(category, "score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def first_handedness(results, idx: int):
    handedness = getattr(results, "handedness", None) or []
    if idx >= len(handedness) or not handedness[idx]:
        return None
    return handedness[idx][0]


def landmark_center(landmarks) -> tuple[float, float]:
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    return ((wrist.x + middle_mcp.x) * 0.5, (wrist.y + middle_mcp.y) * 0.5)


def normalized_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def crop_rect_from_center(
    center: tuple[float, float],
    frame_shape: tuple[int, int, int] | tuple[int, int],
    size_fraction: float,
) -> CropRect:
    frame_h, frame_w = frame_shape[:2]
    side = max(32, int(min(frame_w, frame_h) * max(0.05, size_fraction)))
    cx = int(max(0.0, min(1.0, center[0])) * frame_w)
    cy = int(max(0.0, min(1.0, center[1])) * frame_h)
    x0 = max(0, min(frame_w - side, cx - side // 2))
    y0 = max(0, min(frame_h - side, cy - side // 2))
    width = min(side, frame_w - x0)
    height = min(side, frame_h - y0)
    return CropRect(x=x0, y=y0, width=width, height=height)


def map_crop_landmarks_to_frame(
    landmarks: Iterable,
    crop_rect: CropRect,
    frame_size: tuple[int, int],
) -> list:
    frame_w, frame_h = frame_size
    z_scale = crop_rect.width / max(frame_w, 1)
    mapped = []
    for lm in landmarks:
        mapped.append(
            SimpleNamespace(
                x=(crop_rect.x + float(lm.x) * crop_rect.width) / max(frame_w, 1),
                y=(crop_rect.y + float(lm.y) * crop_rect.height) / max(frame_h, 1),
                z=float(getattr(lm, "z", 0.0)) * z_scale,
                visibility=getattr(lm, "visibility", None),
                presence=getattr(lm, "presence", None),
            )
        )
    return mapped


def clone_category(category, *, side: str | None = None):
    display_name = side or getattr(category, "display_name", "") or getattr(category, "category_name", "")
    return SimpleNamespace(
        index=getattr(category, "index", 0),
        score=category_score(category),
        display_name=display_name,
        category_name=getattr(category, "category_name", display_name),
    )


def merge_hand_results(base_results, additions: list[tuple[list, list]]) -> HandResultAdapter:
    hands = list(getattr(base_results, "hand_landmarks", None) or [])
    handedness = list(getattr(base_results, "handedness", None) or [])
    world = list(getattr(base_results, "hand_world_landmarks", None) or [])
    while len(world) < len(hands):
        world.append([])
    for landmarks, handed in additions:
        hands.append(landmarks)
        handedness.append(handed)
        world.append([])
    return HandResultAdapter(hands, handedness, world)


def _existing_sides_and_centers(results) -> tuple[set[str], list[tuple[float, float]]]:
    sides: set[str] = set()
    centers: list[tuple[float, float]] = []
    for idx, landmarks in enumerate(getattr(results, "hand_landmarks", None) or []):
        centers.append(landmark_center(landmarks))
        handed = first_handedness(results, idx)
        if handed is not None:
            side = category_side(handed)
            if side in {"LEFT", "RIGHT"}:
                sides.add(side)
    return sides, centers


def rescue_hand_results(
    *,
    image_detector,
    frame_rgb: np.ndarray,
    base_results,
    candidates: list[RescueCandidate],
    config: RoiRescueConfig,
) -> tuple[object, RoiRescueStats]:
    stats = RoiRescueStats(candidates=len(candidates))
    if not config.enabled or hand_count(base_results) >= 2 or not candidates:
        return base_results, stats

    existing_sides, existing_centers = _existing_sides_and_centers(base_results)
    additions: list[tuple[list, list]] = []
    frame_h, frame_w = frame_rgb.shape[:2]

    for candidate in candidates:
        expected_side = candidate.expected_side.upper()
        if expected_side in existing_sides:
            continue
        if any(normalized_distance(candidate.center, center) <= config.duplicate_distance for center in existing_centers):
            continue

        rect = crop_rect_from_center(candidate.center, frame_rgb.shape, config.size_fraction)
        crop = frame_rgb[rect.y : rect.y + rect.height, rect.x : rect.x + rect.width]
        if crop.size == 0:
            continue

        stats.attempted += 1
        scale = max(1.0, float(config.scale))
        scaled = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        crop_results = image_detector(scaled)
        selected = None

        for idx, crop_landmarks in enumerate(getattr(crop_results, "hand_landmarks", None) or []):
            handed = first_handedness(crop_results, idx)
            if handed is None:
                continue
            side = category_side(handed)
            score = category_score(handed)
            mapped = map_crop_landmarks_to_frame(crop_landmarks, rect, (frame_w, frame_h))
            center = landmark_center(mapped)
            distance = normalized_distance(center, candidate.center)
            if distance > config.max_candidate_distance or score < config.min_accept_score:
                continue
            if side != expected_side and score < config.mismatch_min_score:
                continue
            rank = (1 if side == expected_side else 0, score, -distance)
            if selected is None or rank > selected[0]:
                selected = (rank, mapped, [clone_category(handed)], center, side)

        if selected is None:
            continue
        _, mapped, handedness, center, side = selected
        if any(normalized_distance(center, existing) <= config.duplicate_distance for existing in existing_centers):
            continue
        additions.append((mapped, handedness))
        existing_centers.append(center)
        if side in {"LEFT", "RIGHT"}:
            existing_sides.add(side)
        stats.added += 1

    if not additions:
        return base_results, stats
    return merge_hand_results(base_results, additions), stats
