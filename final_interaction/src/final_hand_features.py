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
    pose_fallback: bool = False


class FinalHandTracker:
    """Track one dominant hand as a soft 3D grab/open controller.

    We flip the frame before MediaPipe inference, so MediaPipe's handedness
    labels are mirrored from the visitor's point of view.
    """

    def __init__(self, control_hand: str = "RIGHT") -> None:
        control_hand = control_hand.upper()
        self.control_hand = control_hand if control_hand in ("LEFT", "RIGHT") else "RIGHT"
        self.expected_mp_side = "LEFT" if self.control_hand == "RIGHT" else "RIGHT"

        # Keep still-hand jitter low, but release the filter quickly on fast movement.
        self.x_filter = OneEuroFilter(min_cutoff=0.85, beta=0.45)
        self.y_filter = OneEuroFilter(min_cutoff=0.85, beta=0.45)
        self.z_filter = OneEuroFilter(min_cutoff=0.95, beta=0.50)
        self.open_filter = OneEuroFilter(min_cutoff=1.10, beta=0.08)
        self.grab_filter = OneEuroFilter(min_cutoff=1.10, beta=0.08)
        self.radius_filter = OneEuroFilter(min_cutoff=0.95, beta=0.06)
        self.grace_seconds = 0.22
        self.grab_hold_seconds = 0.28

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
                    side=self.last_features.side,
                    x=x,
                    y=y,
                    z=self.last_features.z,
                    open_strength=self.last_features.open_strength,
                    grab_strength=self.last_features.grab_strength,
                    grab_active=self.last_features.grab_active,
                    speed=self.last_features.speed * (1.0 - grace_t),
                    palm_radius=self.last_features.palm_radius,
                    pose_fallback=fallback_xy is not None,
                )

            self.last_features = None
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
        pose_fallback = False

        # Open hand is much easier to observe than a closed fist.
        # When the hand is clearly not open, use that as a strong grab prior.
        inverse_open_grab = 1.0 - remap_clamped(filtered_open, 0.18, 0.70, 0.0, 1.0)
        filtered_grab = max(filtered_grab, inverse_open_grab * 0.92)

        if pose_support is not None and filtered_open <= 0.38 and filtered_grab < 0.60:
            pose_boost = remap_clamped(pose_support[2], 0.0, 1.0, 0.08, 0.26)
            close_bonus = remap_clamped(0.38 - filtered_open, 0.0, 0.38, 0.0, 0.18)
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

        grab_active = filtered_grab >= 0.58 and filtered_open <= 0.52
        if grab_active:
            self.last_grab_time = now
        elif (
            self.last_grab_time is not None
            and now - self.last_grab_time <= self.grab_hold_seconds
            and filtered_open <= 0.56
        ):
            grab_active = True
        else:
            self.last_grab_time = None

        if filtered_open >= 0.64:
            grab_active = False
            filtered_grab = min(filtered_grab, 0.42)
            self.last_grab_time = None

        features = FinalHandFeatures(
            visible=True,
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
        )
        self.last_features = features
        self.last_seen_time = now
        return features

    def _select_hand(self, hand_results):
        if not getattr(hand_results, "hand_landmarks", None):
            return None

        expected_candidates: list[tuple[int, float]] = []
        fallback_candidates: list[tuple[int, float, float]] = []
        for idx, _ in enumerate(hand_results.hand_landmarks):
            if idx >= len(hand_results.handedness):
                continue
            handed = hand_results.handedness[idx][0]
            side = handed.display_name.upper()
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
            side = hand_results.handedness[best_idx][0].display_name.upper()
            return side, hand_results.hand_landmarks[best_idx]

        if not fallback_candidates:
            return None

        # If handedness flips for the same physical hand, allow a very local fallback.
        nearest_idx, _, nearest_distance = min(fallback_candidates, key=lambda item: item[2])
        if self.last_features is not None and nearest_distance <= 0.08:
            side = hand_results.handedness[nearest_idx][0].display_name.upper()
            return side, hand_results.hand_landmarks[nearest_idx]

        return None

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
