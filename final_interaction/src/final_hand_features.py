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
    hand_visible: bool = False
    side: str = ""
    x: float = 0.5
    y: float = 0.5
    z: float = 0.0
    open_strength: float = 0.0
    grab_strength: float = 0.0
    grab_active: bool = False
    speed: float = 0.0
    palm_radius: float = 0.0
    pose_fallback: bool = False
    fallback_age: float = 0.0
    handedness_score: float = 0.0
    index_ratio: float = 0.0
    middle_ratio: float = 0.0
    ring_ratio: float = 0.0
    pinky_ratio: float = 0.0


class FinalHandTracker:
    """Track one dominant hand as a soft 3D grab/open controller.

    We flip the frame before MediaPipe inference, so MediaPipe's handedness
    labels are mirrored from the visitor's point of view.
    """

    def __init__(self, control_hand: str = "RIGHT", *, grace_seconds: float = 0.22) -> None:
        control_hand = control_hand.upper()
        self.control_hand = control_hand if control_hand in ("LEFT", "RIGHT") else "RIGHT"
        self.expected_mp_side = "LEFT" if self.control_hand == "RIGHT" else "RIGHT"

        # Keep still-hand jitter low, but release the filter quickly on fast movement.
        self.x_filter = OneEuroFilter(min_cutoff=0.85, beta=0.45)
        self.y_filter = OneEuroFilter(min_cutoff=0.85, beta=0.45)
        self.z_filter = OneEuroFilter(min_cutoff=0.95, beta=0.50)
        self.open_filter = OneEuroFilter(min_cutoff=1.65, beta=0.18)
        self.grab_filter = OneEuroFilter(min_cutoff=1.10, beta=0.08)
        self.radius_filter = OneEuroFilter(min_cutoff=0.95, beta=0.06)
        self.grace_seconds = max(0.0, grace_seconds)
        self.grab_hold_seconds = 0.18

        self.last_filtered_xyz: tuple[float, float, float] | None = None
        self.last_speed_time: float | None = None
        self.last_features: FinalHandFeatures | None = None
        self.last_seen_time: float | None = None
        self.last_grab_time: float | None = None

    def update(self, hand_results, pose_landmarks: dict[str, tuple[float, float]] | None, now: float) -> FinalHandFeatures:
        pose_support = self._compute_pose_support(pose_landmarks)
        hand = self._select_hand(hand_results)
        if hand is None:
            if (
                self.last_features is not None
                and self.last_seen_time is not None
                and now - self.last_seen_time <= self.grace_seconds
            ):
                grace_t = clamp((now - self.last_seen_time) / self.grace_seconds, 0.0, 1.0)
                fallback_xy = self._pose_center_from_support(pose_support)
                x = self.last_features.x
                y = self.last_features.y
                if fallback_xy is not None:
                    x = clamp(self.x_filter.apply(fallback_xy[0], now), 0.0, 1.0)
                    y = clamp(self.y_filter.apply(fallback_xy[1], now), 0.0, 1.0)
                # Keep the last stable pose for a brief moment, while easing out motion.
                return FinalHandFeatures(
                    visible=True,
                    hand_visible=False,
                    side=self.last_features.side,
                    x=x,
                    y=y,
                    z=self.last_features.z,
                    open_strength=self.last_features.open_strength * (1.0 - 0.35 * grace_t),
                    grab_strength=self.last_features.grab_strength * (1.0 - 0.55 * grace_t),
                    grab_active=self.last_features.grab_active and grace_t <= 0.45,
                    speed=self.last_features.speed * (1.0 - grace_t),
                    palm_radius=self.last_features.palm_radius,
                    pose_fallback=fallback_xy is not None,
                    fallback_age=now - self.last_seen_time,
                    handedness_score=self.last_features.handedness_score,
                    index_ratio=self.last_features.index_ratio,
                    middle_ratio=self.last_features.middle_ratio,
                    ring_ratio=self.last_features.ring_ratio,
                    pinky_ratio=self.last_features.pinky_ratio,
                )

            self.last_features = None
            return FinalHandFeatures()

        side, landmarks, handedness_score = hand
        points = {idx: (landmarks[idx].x, landmarks[idx].y) for idx in (0, 4, 5, 8, 9, 12, 13, 16, 17, 20)}

        center = self._compute_center(points)
        palm_ref = self._compute_palm_ref(points)
        extension_ratios = self._compute_extension_ratios(points)
        open_strength = self._compute_open_strength(points, extension_ratios)
        grab_strength = self._compute_grab_strength(points, extension_ratios)
        palm_radius = clamp(remap_clamped(palm_ref, 0.035, 0.12, 0.0, 1.0), 0.0, 1.0)
        pseudo_depth = clamp(remap_clamped(palm_ref, 0.04, 0.14, -1.0, 1.0), -1.0, 1.0)

        filtered_x = self.x_filter.apply(center[0], now)
        filtered_y = self.y_filter.apply(center[1], now)
        filtered_z = self.z_filter.apply(pseudo_depth, now)
        filtered_open = self.open_filter.apply(open_strength, now)
        filtered_grab = self.grab_filter.apply(grab_strength, now)
        filtered_radius = self.radius_filter.apply(palm_radius, now)
        pose_fallback = False

        # Open hand is much easier to observe than a closed fist.
        # When the hand is clearly not open, use that as a strong grab prior.
        inverse_open_grab = 1.0 - remap_clamped(filtered_open, 0.16, 0.58, 0.0, 1.0)
        filtered_grab = max(filtered_grab, inverse_open_grab * 0.82)

        if pose_support is not None and filtered_open <= 0.34 and filtered_grab < 0.62:
            pose_boost = remap_clamped(pose_support[2], 0.0, 1.0, 0.06, 0.22)
            close_bonus = remap_clamped(0.34 - filtered_open, 0.0, 0.34, 0.0, 0.14)
            filtered_grab = clamp(max(filtered_grab, filtered_grab + pose_boost + close_bonus), 0.0, 1.0)
            pose_fallback = True

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

        grab_active = filtered_grab >= 0.85 and filtered_open <= 0.42
        if grab_active:
            self.last_grab_time = now
        elif (
            self.last_grab_time is not None
            and now - self.last_grab_time <= self.grab_hold_seconds
            and filtered_open <= 0.46
        ):
            grab_active = True
        else:
            self.last_grab_time = None

        if filtered_open >= 0.66:
            grab_active = False
            filtered_grab = min(filtered_grab, 0.42)
            self.last_grab_time = None

        features = FinalHandFeatures(
            visible=True,
            hand_visible=True,
            side=side,
            x=clamp(filtered_x, 0.0, 1.0),
            y=clamp(filtered_y, 0.0, 1.0),
            z=filtered_z,
            open_strength=filtered_open,
            grab_strength=filtered_grab,
            grab_active=grab_active,
            speed=speed,
            palm_radius=filtered_radius,
            pose_fallback=pose_fallback,
            fallback_age=0.0,
            handedness_score=handedness_score,
            index_ratio=extension_ratios[0],
            middle_ratio=extension_ratios[1],
            ring_ratio=extension_ratios[2],
            pinky_ratio=extension_ratios[3],
        )
        self.last_features = features
        self.last_seen_time = now
        return features

    def _select_hand(self, hand_results):
        if not getattr(hand_results, "hand_landmarks", None):
            return None

        def handed_side(handed) -> str:
            return str(getattr(handed, "display_name", "") or getattr(handed, "category_name", "")).upper()

        expected_candidates: list[tuple[int, float]] = []
        fallback_candidates: list[tuple[int, float, float]] = []
        for idx, _ in enumerate(hand_results.hand_landmarks):
            if idx >= len(hand_results.handedness):
                continue
            handed = hand_results.handedness[idx][0]
            side = handed_side(handed)
            score = float(getattr(handed, "score", 0.0))
            if side not in ("LEFT", "RIGHT"):
                continue

            if side == self.expected_mp_side:
                expected_candidates.append((idx, score))
                continue

            center = self._landmark_center(hand_results.hand_landmarks[idx])
            distance_to_last = 999.0
            if self.last_features is not None:
                dx = center[0] - self.last_features.x
                dy = center[1] - self.last_features.y
                distance_to_last = math.sqrt(dx * dx + dy * dy)
            fallback_candidates.append((idx, score, distance_to_last))

        if expected_candidates:
            best_idx = max(expected_candidates, key=lambda item: item[1])[0]
            handed = hand_results.handedness[best_idx][0]
            side = handed_side(handed)
            score = float(getattr(handed, "score", 0.0))
            return side, hand_results.hand_landmarks[best_idx], score

        if not fallback_candidates:
            return None

        # If handedness flips for the same physical hand, allow a very local fallback.
        nearest_idx, _, nearest_distance = min(fallback_candidates, key=lambda item: item[2])
        if self.last_features is not None and nearest_distance <= 0.08:
            handed = hand_results.handedness[nearest_idx][0]
            side = handed_side(handed)
            score = float(getattr(handed, "score", 0.0))
            return side, hand_results.hand_landmarks[nearest_idx], score

        return None

    def rescue_center(self, pose_landmarks: dict[str, tuple[float, float]] | None) -> tuple[float, float] | None:
        if self.last_features is not None:
            return self.last_features.x, self.last_features.y
        return self._pose_center_from_support(self._compute_pose_support(pose_landmarks))

    def _landmark_center(self, landmarks) -> tuple[float, float]:
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        return ((wrist.x + middle_mcp.x) * 0.5, (wrist.y + middle_mcp.y) * 0.5)

    def _compute_center(self, points: dict[int, tuple[float, float]]) -> tuple[float, float]:
        anchors = [points[0], points[5], points[9], points[13], points[17]]
        return (
            sum(point[0] for point in anchors) / len(anchors),
            sum(point[1] for point in anchors) / len(anchors),
        )


    def _compute_pose_support(
        self,
        pose_landmarks: dict[str, tuple[float, float]] | None,
    ) -> tuple[float, float, float] | None:
        if not pose_landmarks:
            return None

        side_prefix = "LEFT" if self.control_hand == "RIGHT" else "RIGHT"
        wrist_key = f"{side_prefix}_WRIST"
        elbow_key = f"{side_prefix}_ELBOW"
        shoulder_key = f"{side_prefix}_SHOULDER"
        if wrist_key not in pose_landmarks or elbow_key not in pose_landmarks or shoulder_key not in pose_landmarks:
            return None

        wrist = pose_landmarks[wrist_key]
        elbow = pose_landmarks[elbow_key]
        shoulder = pose_landmarks[shoulder_key]
        upper = max(distance(shoulder, elbow), 1e-4)
        lower = distance(elbow, wrist)
        arm_extension = clamp(remap_clamped(lower / upper, 0.55, 1.65, 0.0, 1.0), 0.0, 1.0)
        return wrist[0], wrist[1], arm_extension

    def _pose_center_from_support(
        self,
        pose_support: tuple[float, float, float] | None,
    ) -> tuple[float, float] | None:
        if pose_support is None:
            return None
        return pose_support[0], pose_support[1]

    def _compute_palm_ref(self, points: dict[int, tuple[float, float]]) -> float:
        wrist = points[0]
        refs = [
            distance(wrist, points[5]),
            distance(wrist, points[9]),
            distance(wrist, points[13]),
            distance(wrist, points[17]),
        ]
        return max(0.02, sum(refs) / len(refs))

    def _compute_extension_ratios(self, points: dict[int, tuple[float, float]]) -> list[float]:
        wrist = points[0]
        finger_pairs = ((5, 8), (9, 12), (13, 16), (17, 20))
        ratios: list[float] = []
        for mcp_idx, tip_idx in finger_pairs:
            wrist_to_mcp = max(distance(wrist, points[mcp_idx]), 1e-4)
            mcp_to_tip = distance(points[mcp_idx], points[tip_idx])
            ratios.append(mcp_to_tip / wrist_to_mcp)
        return ratios

    def _compute_open_strength(
        self,
        points: dict[int, tuple[float, float]],
        extension_ratios: list[float],
    ) -> float:
        ratio_terms = [remap_clamped(ratio, 0.45, 0.95, 0.0, 1.0) for ratio in extension_ratios]
        mean_term = sum(ratio_terms) / len(ratio_terms)
        min_term = min(ratio_terms)
        return clamp(0.82 * mean_term + 0.18 * min_term, 0.0, 1.0)

    def _compute_grab_strength(
        self,
        points: dict[int, tuple[float, float]],
        extension_ratios: list[float],
    ) -> float:
        compact_terms = [1.0 - remap_clamped(ratio, 0.28, 0.62, 0.0, 1.0) for ratio in extension_ratios]
        mean_term = sum(compact_terms) / len(compact_terms)
        max_term = max(compact_terms)
        return clamp(0.72 * mean_term + 0.28 * max_term, 0.0, 1.0)


def actual_both_open(
    right_features: FinalHandFeatures,
    left_features: FinalHandFeatures,
    *,
    min_each: float = 0.62,
    min_average: float = 0.72,
) -> bool:
    return (
        right_features.hand_visible
        and left_features.hand_visible
        and right_features.open_strength >= min_each
        and left_features.open_strength >= min_each
        and (right_features.open_strength + left_features.open_strength) * 0.5 >= min_average
    )


def actual_both_grab(right_features: FinalHandFeatures, left_features: FinalHandFeatures) -> bool:
    return (
        right_features.hand_visible
        and left_features.hand_visible
        and right_features.grab_active
        and left_features.grab_active
    )
