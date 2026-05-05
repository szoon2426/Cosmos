from __future__ import annotations

import math
from dataclasses import dataclass


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def remap_clamped(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    if in_max == in_min:
        return out_min
    t = clamp((value - in_min) / (in_max - in_min), 0.0, 1.0)
    return out_min + (out_max - out_min) * t


def smoothing_factor(dt: float, cutoff: float) -> float:
    r = 2.0 * math.pi * cutoff * dt
    return r / (r + 1.0)


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class LowPassFilter:
    def __init__(self) -> None:
        self.initialized = False
        self.prev = 0.0

    def apply(self, value: float, alpha: float) -> float:
        if not self.initialized:
            self.initialized = True
            self.prev = value
            return value
        self.prev = alpha * value + (1.0 - alpha) * self.prev
        return self.prev


class OneEuroFilter:
    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_value: float | None = None
        self.last_time: float | None = None

    def apply(self, value: float, now: float) -> float:
        if self.last_time is None or self.last_value is None:
            self.last_time = now
            self.last_value = value
            return self.x_filter.apply(value, 1.0)

        dt = max(now - self.last_time, 1e-3)
        dx = (value - self.last_value) / dt
        dx_hat = self.dx_filter.apply(dx, smoothing_factor(dt, self.d_cutoff))
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        filtered = self.x_filter.apply(value, smoothing_factor(dt, cutoff))
        self.last_time = now
        self.last_value = filtered
        return filtered


@dataclass(slots=True)
class FinalHandFeatures:
    visible: bool = False
    side: str = ""
    x: float = 0.5
    y: float = 0.5
    z: float = 0.0
    open_strength: float = 0.0
    grab_strength: float = 0.0
    grab_active: bool = False
    speed: float = 0.0
    palm_radius: float = 0.0


class FinalHandTracker:
    """Track one dominant hand as a soft 3D grab/open controller.

    We flip the frame before MediaPipe inference, so MediaPipe's handedness
    labels are mirrored from the visitor's point of view.
    """

    def __init__(self, control_hand: str = "RIGHT") -> None:
        control_hand = control_hand.upper()
        self.control_hand = control_hand if control_hand in ("LEFT", "RIGHT") else "RIGHT"
        self.expected_mp_side = "LEFT" if self.control_hand == "RIGHT" else "RIGHT"

        self.x_filter = OneEuroFilter(min_cutoff=0.45, beta=0.08)
        self.y_filter = OneEuroFilter(min_cutoff=0.45, beta=0.08)
        self.z_filter = OneEuroFilter(min_cutoff=0.55, beta=0.10)
        self.open_filter = OneEuroFilter(min_cutoff=0.90, beta=0.04)
        self.grab_filter = OneEuroFilter(min_cutoff=0.90, beta=0.04)
        self.radius_filter = OneEuroFilter(min_cutoff=0.80, beta=0.04)

        self.last_filtered_xyz: tuple[float, float, float] | None = None
        self.last_speed_time: float | None = None

    def update(self, hand_results, now: float) -> FinalHandFeatures:
        hand = self._select_hand(hand_results)
        if hand is None:
            return FinalHandFeatures()

        side, landmarks = hand
        points = {idx: (landmarks[idx].x, landmarks[idx].y) for idx in (0, 4, 5, 8, 9, 12, 13, 16, 17, 20)}

        center = self._compute_center(points)
        ref_scale = max(
            0.03,
            0.5 * (distance(points[5], points[17]) + distance(points[0], points[9])),
        )
        open_strength = self._compute_open_strength(points, center, ref_scale)
        grab_strength = self._compute_grab_strength(points, center, ref_scale)
        palm_radius = clamp(remap_clamped(ref_scale, 0.04, 0.18, 0.0, 1.0), 0.0, 1.0)
        pseudo_depth = clamp(remap_clamped(ref_scale, 0.05, 0.18, -1.0, 1.0), -1.0, 1.0)

        filtered_x = self.x_filter.apply(center[0], now)
        filtered_y = self.y_filter.apply(center[1], now)
        filtered_z = self.z_filter.apply(pseudo_depth, now)
        filtered_open = self.open_filter.apply(open_strength, now)
        filtered_grab = self.grab_filter.apply(grab_strength, now)
        filtered_radius = self.radius_filter.apply(palm_radius, now)

        speed = 0.0
        if self.last_filtered_xyz is not None and self.last_speed_time is not None:
            dt = max(now - self.last_speed_time, 1e-3)
            dx = filtered_x - self.last_filtered_xyz[0]
            dy = filtered_y - self.last_filtered_xyz[1]
            dz = filtered_z - self.last_filtered_xyz[2]
            raw_speed = math.sqrt(dx * dx + dy * dy + dz * dz) / dt
            speed = clamp(remap_clamped(raw_speed, 0.0, 1.8, 0.0, 1.0), 0.0, 1.0)

        self.last_filtered_xyz = (filtered_x, filtered_y, filtered_z)
        self.last_speed_time = now

        return FinalHandFeatures(
            visible=True,
            side=side,
            x=clamp(filtered_x, 0.0, 1.0),
            y=clamp(filtered_y, 0.0, 1.0),
            z=filtered_z,
            open_strength=filtered_open,
            grab_strength=filtered_grab,
            grab_active=filtered_grab >= 0.60 and filtered_open <= 0.55,
            speed=speed,
            palm_radius=filtered_radius,
        )

    def _select_hand(self, hand_results):
        if not getattr(hand_results, "hand_landmarks", None):
            return None

        candidates: list[tuple[int, float]] = []
        for idx, _ in enumerate(hand_results.hand_landmarks):
            if idx >= len(hand_results.handedness):
                continue
            handed = hand_results.handedness[idx][0]
            side = handed.display_name.upper()
            score = float(getattr(handed, "score", 0.0))
            if side in ("LEFT", "RIGHT"):
                side_bonus = 1.0 if side == self.expected_mp_side else 0.0
                candidates.append((idx, side_bonus + score))

        if not candidates:
            return None

        best_idx = max(candidates, key=lambda item: item[1])[0]
        side = hand_results.handedness[best_idx][0].display_name.upper()
        return side, hand_results.hand_landmarks[best_idx]

    def _compute_center(self, points: dict[int, tuple[float, float]]) -> tuple[float, float]:
        anchors = [points[0], points[5], points[9], points[13], points[17]]
        return (
            sum(point[0] for point in anchors) / len(anchors),
            sum(point[1] for point in anchors) / len(anchors),
        )

    def _compute_open_strength(
        self,
        points: dict[int, tuple[float, float]],
        center: tuple[float, float],
        ref_scale: float,
    ) -> float:
        tip_dists = [distance(points[idx], center) / ref_scale for idx in (8, 12, 16, 20)]
        thumb_dist = distance(points[4], center) / ref_scale
        span = distance(points[8], points[20]) / ref_scale

        tip_term = remap_clamped(sum(tip_dists) / len(tip_dists), 0.70, 1.70, 0.0, 1.0)
        span_term = remap_clamped(span, 1.00, 2.30, 0.0, 1.0)
        thumb_term = remap_clamped(thumb_dist, 0.65, 1.70, 0.0, 1.0)
        return clamp(0.55 * tip_term + 0.30 * span_term + 0.15 * thumb_term, 0.0, 1.0)

    def _compute_grab_strength(
        self,
        points: dict[int, tuple[float, float]],
        center: tuple[float, float],
        ref_scale: float,
    ) -> float:
        tip_dists = [distance(points[idx], center) / ref_scale for idx in (8, 12, 16, 20)]
        thumb_dist = distance(points[4], center) / ref_scale
        span = distance(points[8], points[20]) / ref_scale

        compact_term = 1.0 - remap_clamped(sum(tip_dists) / len(tip_dists), 0.70, 1.50, 0.0, 1.0)
        span_term = 1.0 - remap_clamped(span, 0.95, 2.05, 0.0, 1.0)
        thumb_term = 1.0 - remap_clamped(thumb_dist, 0.60, 1.40, 0.0, 1.0)
        return clamp(0.55 * compact_term + 0.25 * span_term + 0.20 * thumb_term, 0.0, 1.0)
