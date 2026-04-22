from __future__ import annotations

import argparse
import math
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    from feature_engineering import (
        SlidingWindowBuffer,
        compute_frame_features,
        compute_snap_frame_features,
        detect_snap_event,
        detect_snap_gate,
        normalize_landmarks,
    )
    from eeg_profile_loader import BaseVADState, load_eeg_profile, resolve_base_vad
    from hand_extractor import HandExtractor
    from model_io import load_model
    from pose_extractor import PoseExtractor
    from vad_mapper import compute_unreal_payload
    from ue_bridge import UEBridge
    from pd_bridge import PDBridge
else:
    from .feature_engineering import (
        SlidingWindowBuffer,
        compute_frame_features,
        compute_snap_frame_features,
        detect_snap_event,
        detect_snap_gate,
        normalize_landmarks,
    )
    from .eeg_profile_loader import BaseVADState, load_eeg_profile, resolve_base_vad
    from .hand_extractor import HandExtractor
    from .model_io import load_model
    from .pose_extractor import PoseExtractor
    from .vad_mapper import compute_unreal_payload
    from .ue_bridge import UEBridge
    from .pd_bridge import PDBridge


SKELETON_PAIRS = [
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"),
    ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP", "RIGHT_HIP"),
]

JOINT_COLORS = {
    "LEFT_WRIST": (0, 0, 255),
    "RIGHT_WRIST": (0, 0, 255),
    "LEFT_ELBOW": (0, 165, 255),
    "RIGHT_ELBOW": (0, 165, 255),
    "LEFT_SHOULDER": (255, 255, 0),
    "RIGHT_SHOULDER": (255, 255, 0),
    "LEFT_HAND_INDEX_TIP": (255, 0, 255),
    "RIGHT_HAND_INDEX_TIP": (255, 0, 255),
    "LEFT_HAND_THUMB_TIP": (0, 255, 255),
    "RIGHT_HAND_THUMB_TIP": (0, 255, 255),
}

ARMING_SECONDS = {
    "rise": 1.0,
    "open": 1.0,
    "prayer": 1.0,
    "breath": 1.0,
    "snap": 0.2,
}

RELEASE_FRAMES = {
    "rise": 8,
    "open": 8,
    "prayer": 10,
    "breath": 10,
    "snap": 4,
}

COOLDOWN_FRAMES = {
    "rise": 12,
    "open": 12,
    "prayer": 14,
    "breath": 14,
    "snap": 12,
}

RIVAL_LABEL_FRAMES = {
    "rise": 3,
    "open": 3,
    "prayer": 4,
    "breath": 4,
    "snap": 2,
}
ARMING_GRACE_FRAMES = {
    "rise": 4,
    "open": 3,
}

ACTIVE_RIVAL_SECONDS = 1.0
ARMING_STABILITY_LIMITS = {
    "open": (0.45, 0.45, 0.45),
    "rise": (0.0, 0.26),
    "prayer": (0.18, 0.24, 0.24),
    "breath": (0.22, 0.28, 0.28),
}

OPEN_DISTANCE_SCALE = 1.2
RISE_HEIGHT_SCALE = 1.0
PRAYER_FULL_FRAMES = 45
BREATH_FULL_FRAMES = 45
SNAP_PULSE_FRAMES = 4
DECAY_DELTA_SCALE = 8.0
ONE_EURO_SIGNAL_MIN_CUTOFF = 1.2
ONE_EURO_SIGNAL_BETA = 0.18
ONE_EURO_SIGNAL_D_CUTOFF = 1.0
RISE_RELEASE_GRACE_FRAMES = 30
READY_OFF_RECOVERY_DELAY_SECONDS = 5.0
READY_OFF_RECOVERY_DURATION_SECONDS = 25.0
PRAYER_RECOVERY_ALPHA_MIN = 0.03
PRAYER_RECOVERY_ALPHA_MAX = 0.10
BREATH_RECOVERY_ALPHA_MIN = 0.02
BREATH_RECOVERY_ALPHA_MAX = 0.08
ACTIVE_TARGET_ALPHA = 0.22
BASE_MEMORY_LEARN_RATE = 0.02


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def allowed_prediction_labels(ready_gate: bool) -> set[str]:
    if not ready_gate:
        return {"snap", "uncertain"}
    return {"snap", "rise", "open", "prayer", "breath", "uncertain"}


def masked_prediction_from_probs(
    model: object,
    probs: np.ndarray,
    ready_gate: bool,
    threshold: float = 0.5,
) -> tuple[str, float]:
    classes = [str(item) for item in getattr(model, "classes_", [])]
    if not classes or len(classes) != len(probs):
        max_prob = float(np.max(probs)) if len(probs) else 0.0
        return ("uncertain", max_prob)

    allowed = allowed_prediction_labels(ready_gate)
    masked = np.zeros_like(probs, dtype=np.float32)
    for idx, label in enumerate(classes):
        if label in allowed:
            masked[idx] = float(probs[idx])

    total = float(masked.sum())
    if total <= 0.0:
        return ("uncertain", 0.0)

    masked /= total
    best_idx = int(np.argmax(masked))
    best_prob = float(masked[best_idx])
    best_label = classes[best_idx]
    if best_prob < threshold:
        return ("uncertain", best_prob)
    return (best_label, best_prob)


def smoothing_factor(dt: float, cutoff: float) -> float:
    if cutoff <= 0.0:
        return 1.0
    r = 2.0 * math.pi * cutoff * dt
    return r / (r + 1.0)


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


def compute_pd_continuous_values(
    controller: "InteractionController",
    signals: "RuntimeSignals",
    raw_payload,
) -> tuple[dict[str, float], float]:
    current_flower_value = (
        clamp((float(raw_payload.density) - 0.4) / 0.6, 0.0, 1.0)
        if controller.ready_gate
        else 0.0
    )
    current_decay_value = clamp(float(raw_payload.decay) / 1.2, 0.0, 1.0) if controller.ready_gate else 0.0
    previous_decay_value = getattr(controller, "pd_decay_prev", 0.0)

    decay_delta = clamp((current_decay_value - previous_decay_value) * DECAY_DELTA_SCALE, 0.0, 1.0)
    recover_delta = clamp((previous_decay_value - current_decay_value) * DECAY_DELTA_SCALE, 0.0, 1.0)

    controller.pd_decay_prev = current_decay_value

    fountain_value = clamp(((float(raw_payload.min_speed) + float(raw_payload.max_speed)) * 0.5) / 600.0, 0.0, 1.0)

    values = {
        "READY_MODE": 1.0 if controller.ready_gate else 0.0,
        "W": clamp(float(raw_payload.a), -1.0, 1.0),
        "F": clamp(fountain_value * 2.0 - 1.0, -1.0, 1.0),
        "FLOWER": current_flower_value,
        "DECAY": decay_delta,
        "RECOVER": recover_delta,
    }
    return values, current_decay_value


def combine_base_and_delta_vad(
    base_vad: tuple[float, float, float],
    interaction_vad: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        clamp(base_vad[0] + interaction_vad[0], -1.0, 1.0),
        clamp(base_vad[1] + interaction_vad[1], -1.0, 1.0),
        clamp(base_vad[2] + interaction_vad[2], -1.0, 1.0),
    )


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def blend_triplet(
    current: tuple[float, float, float],
    target: tuple[float, float, float],
    alpha: float,
) -> tuple[float, float, float]:
    alpha = clamp(alpha, 0.0, 1.0)
    return (
        lerp(current[0], target[0], alpha),
        lerp(current[1], target[1], alpha),
        lerp(current[2], target[2], alpha),
    )


@dataclass
class WorldVADState:
    v: float = 0.0
    a: float = 0.0
    d: float = 0.0
    initialized: bool = False

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.v, self.a, self.d)

    def set_from_tuple(self, values: tuple[float, float, float]) -> None:
        self.v = clamp(values[0], -1.0, 1.0)
        self.a = clamp(values[1], -1.0, 1.0)
        self.d = clamp(values[2], -1.0, 1.0)
        self.initialized = True


@dataclass
class BaseMemoryState:
    dv: float = 0.0
    da: float = 0.0
    dd: float = 0.0

    def apply_to(self, base_vad: tuple[float, float, float]) -> tuple[float, float, float]:
        return (
            clamp(base_vad[0] + self.dv, -1.0, 1.0),
            clamp(base_vad[1] + self.da, -1.0, 1.0),
            clamp(base_vad[2] + self.dd, -1.0, 1.0),
        )

    def learn_from(
        self,
        base_vad: tuple[float, float, float],
        observed_vad: tuple[float, float, float],
        alpha: float = BASE_MEMORY_LEARN_RATE,
    ) -> None:
        target_dv = observed_vad[0] - base_vad[0]
        target_da = observed_vad[1] - base_vad[1]
        target_dd = observed_vad[2] - base_vad[2]
        self.dv = clamp(lerp(self.dv, target_dv, alpha), -0.25, 0.25)
        self.da = clamp(lerp(self.da, target_da, alpha), -0.25, 0.25)
        self.dd = clamp(lerp(self.dd, target_dd, alpha), -0.25, 0.25)


def _midpoint(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def compute_tracking_metrics(landmarks: dict[str, tuple[float, float]] | None) -> dict[str, object]:
    if not landmarks:
        return {
            "torso_present": False,
            "head_count": 0,
            "front_face_count": 0,
            "shoulder_center": None,
            "torso_height": None,
        }

    has_shoulders = "LEFT_SHOULDER" in landmarks and "RIGHT_SHOULDER" in landmarks
    has_hips = "LEFT_HIP" in landmarks and "RIGHT_HIP" in landmarks
    torso_present = has_shoulders and has_hips

    shoulder_center = None
    torso_height = None
    if torso_present:
        shoulder_center = _midpoint(landmarks["LEFT_SHOULDER"], landmarks["RIGHT_SHOULDER"])
        hip_center = _midpoint(landmarks["LEFT_HIP"], landmarks["RIGHT_HIP"])
        torso_height = max(abs(hip_center[1] - shoulder_center[1]), 1e-5)

    head_count = sum(
        1 for name in ("NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR") if name in landmarks
    )
    front_face_count = sum(
        1 for name in ("LEFT_EYE", "RIGHT_EYE", "MOUTH_LEFT", "MOUTH_RIGHT") if name in landmarks
    )

    return {
        "torso_present": torso_present,
        "head_count": head_count,
        "front_face_count": front_face_count,
        "shoulder_center": shoulder_center,
        "torso_height": torso_height,
    }


@dataclass
class RuntimeSignals:
    open_ready: bool = False
    open_distance: float = 0.0
    wrist_center_x: float = 0.0
    wrist_center_y: float = 0.0
    open_release: bool = False
    rise_side: str = "none"
    left_rise_height: float = 0.0
    right_rise_height: float = 0.0
    left_rise_out: bool = False
    right_rise_out: bool = False
    prayer_gate: bool = False
    breath_gate: bool = False
    snap_gate_side: str = "none"

    def rise_height_for_side(self, side: str) -> float:
        if side == "left":
            return self.left_rise_height
        if side == "right":
            return self.right_rise_height
        return 0.0

    def rise_out_for_side(self, side: str) -> bool:
        if side == "left":
            return self.left_rise_out
        if side == "right":
            return self.right_rise_out
        return False


@dataclass
class RuntimeSignalFilter:
    open_distance: OneEuroFilter = field(
        default_factory=lambda: OneEuroFilter(
            min_cutoff=ONE_EURO_SIGNAL_MIN_CUTOFF,
            beta=ONE_EURO_SIGNAL_BETA,
            d_cutoff=ONE_EURO_SIGNAL_D_CUTOFF,
        )
    )
    wrist_center_x: OneEuroFilter = field(
        default_factory=lambda: OneEuroFilter(
            min_cutoff=ONE_EURO_SIGNAL_MIN_CUTOFF,
            beta=ONE_EURO_SIGNAL_BETA,
            d_cutoff=ONE_EURO_SIGNAL_D_CUTOFF,
        )
    )
    wrist_center_y: OneEuroFilter = field(
        default_factory=lambda: OneEuroFilter(
            min_cutoff=ONE_EURO_SIGNAL_MIN_CUTOFF,
            beta=ONE_EURO_SIGNAL_BETA,
            d_cutoff=ONE_EURO_SIGNAL_D_CUTOFF,
        )
    )
    left_rise_height: OneEuroFilter = field(
        default_factory=lambda: OneEuroFilter(
            min_cutoff=ONE_EURO_SIGNAL_MIN_CUTOFF,
            beta=ONE_EURO_SIGNAL_BETA,
            d_cutoff=ONE_EURO_SIGNAL_D_CUTOFF,
        )
    )
    right_rise_height: OneEuroFilter = field(
        default_factory=lambda: OneEuroFilter(
            min_cutoff=ONE_EURO_SIGNAL_MIN_CUTOFF,
            beta=ONE_EURO_SIGNAL_BETA,
            d_cutoff=ONE_EURO_SIGNAL_D_CUTOFF,
        )
    )

    def apply(self, signals: RuntimeSignals, now: float) -> RuntimeSignals:
        filtered_open_distance = self.open_distance.apply(signals.open_distance, now)
        filtered_wrist_center_x = self.wrist_center_x.apply(signals.wrist_center_x, now)
        filtered_wrist_center_y = self.wrist_center_y.apply(signals.wrist_center_y, now)
        filtered_left_rise_height = self.left_rise_height.apply(signals.left_rise_height, now)
        filtered_right_rise_height = self.right_rise_height.apply(signals.right_rise_height, now)

        return RuntimeSignals(
            open_ready=signals.open_ready,
            open_distance=filtered_open_distance,
            wrist_center_x=filtered_wrist_center_x,
            wrist_center_y=filtered_wrist_center_y,
            open_release=signals.open_release,
            rise_side=signals.rise_side,
            left_rise_height=filtered_left_rise_height,
            right_rise_height=filtered_right_rise_height,
            left_rise_out=signals.left_rise_out,
            right_rise_out=signals.right_rise_out,
            prayer_gate=signals.prayer_gate,
            breath_gate=signals.breath_gate,
            snap_gate_side=signals.snap_gate_side,
        )


@dataclass
class SnapGateSmoother:
    open_frames: int = 3
    close_frames: int = 4
    state: str = "none"
    candidate_side: str = "none"
    candidate_count: int = 0
    missing_count: int = 0

    def update(self, raw_side: str) -> str:
        raw_side = raw_side if raw_side in {"left", "right"} else "none"

        if self.state == "none":
            if raw_side == "none":
                self.candidate_side = "none"
                self.candidate_count = 0
                return "none"
            if raw_side == self.candidate_side:
                self.candidate_count += 1
            else:
                self.candidate_side = raw_side
                self.candidate_count = 1
            if self.candidate_count >= self.open_frames:
                self.state = self.candidate_side
                self.missing_count = 0
            return self.state

        if raw_side == self.state:
            self.missing_count = 0
            self.candidate_side = self.state
            self.candidate_count = 0
            return self.state

        if raw_side == "none":
            self.missing_count += 1
            if self.missing_count >= self.close_frames:
                self.state = "none"
                self.candidate_side = "none"
                self.candidate_count = 0
                self.missing_count = 0
            return self.state

        self.missing_count += 1
        if raw_side == self.candidate_side:
            self.candidate_count += 1
        else:
            self.candidate_side = raw_side
            self.candidate_count = 1

        if self.candidate_count >= self.open_frames:
            self.state = raw_side
            self.missing_count = 0
            self.candidate_count = 0
        elif self.missing_count >= self.close_frames:
            self.state = "none"
            self.candidate_side = "none"
            self.candidate_count = 0
            self.missing_count = 0

        return self.state


@dataclass
class RiseSideSmoother:
    open_frames: int = 2
    close_frames: int = 5
    state: str = "none"
    candidate_side: str = "none"
    candidate_count: int = 0
    missing_count: int = 0

    def update(self, raw_side: str) -> str:
        raw_side = raw_side if raw_side in {"left", "right"} else "none"

        if self.state == "none":
            if raw_side == "none":
                self.candidate_side = "none"
                self.candidate_count = 0
                return "none"
            if raw_side == self.candidate_side:
                self.candidate_count += 1
            else:
                self.candidate_side = raw_side
                self.candidate_count = 1
            if self.candidate_count >= self.open_frames:
                self.state = self.candidate_side
                self.missing_count = 0
            return self.state

        if raw_side == self.state:
            self.missing_count = 0
            self.candidate_side = self.state
            self.candidate_count = 0
            return self.state

        if raw_side == "none":
            self.missing_count += 1
            if self.missing_count >= self.close_frames:
                self.state = "none"
                self.candidate_side = "none"
                self.candidate_count = 0
                self.missing_count = 0
            return self.state

        self.missing_count += 1
        if raw_side == self.candidate_side:
            self.candidate_count += 1
        else:
            self.candidate_side = raw_side
            self.candidate_count = 1

        if self.candidate_count >= self.open_frames:
            self.state = raw_side
            self.missing_count = 0
            self.candidate_count = 0
        elif self.missing_count >= self.close_frames:
            self.state = "none"
            self.candidate_side = "none"
            self.candidate_count = 0
            self.missing_count = 0

        return self.state


def build_runtime_signals(normalized_landmarks: dict[str, np.ndarray]) -> RuntimeSignals:
    ls = normalized_landmarks["LEFT_SHOULDER"]
    rs = normalized_landmarks["RIGHT_SHOULDER"]
    le = normalized_landmarks["LEFT_ELBOW"]
    re = normalized_landmarks["RIGHT_ELBOW"]
    lw = normalized_landmarks["LEFT_WRIST"]
    rw = normalized_landmarks["RIGHT_WRIST"]
    nose = normalized_landmarks.get("NOSE", np.array([0.0, -0.5], dtype=np.float32))

    wrist_distance = float(np.linalg.norm(lw - rw))
    wrist_center = (lw + rw) * 0.5
    open_ready = bool(
        abs(float(lw[1] - rw[1])) < 0.6
        and lw[1] < 1.4
        and rw[1] < 1.4
    )
    # Open should release only when both hands are clearly lowered,
    # not when they move inward in front of the chest.
    open_release = bool(lw[1] > 1.25 and rw[1] > 1.25)

    left_rise_height = float(-lw[1])
    right_rise_height = float(-rw[1])
    rise_side_margin = 0.06
    if left_rise_height > right_rise_height + rise_side_margin:
        rise_side = "left"
    elif right_rise_height > left_rise_height + rise_side_margin:
        rise_side = "right"
    else:
        rise_side = "none"

    # Rise should only release when the active hand clearly exits sideways,
    # not because the wrist jitters a little while moving up/down.
    left_rise_out = bool(lw[0] < ls[0] - 1.15 or lw[0] > rs[0] + 1.15)
    right_rise_out = bool(rw[0] > rs[0] + 1.15 or rw[0] < ls[0] - 1.15)

    prayer_gate = bool(wrist_distance < 0.55 and abs(float(lw[1] - rw[1])) < 0.45)

    elbows_outward = bool(le[0] < lw[0] - 0.05 and re[0] > rw[0] + 0.05)
    wrists_low = bool(lw[1] > 0.15 and rw[1] > 0.15)
    head_up = bool(nose[1] < -0.55)
    breath_gate = bool(wrists_low and elbows_outward and head_up)

    snap_gate = detect_snap_gate(normalized_landmarks)

    return RuntimeSignals(
        open_ready=open_ready,
        open_distance=wrist_distance,
        wrist_center_x=float(wrist_center[0]),
        wrist_center_y=float(wrist_center[1]),
        open_release=open_release,
        rise_side=rise_side,
        left_rise_height=left_rise_height,
        right_rise_height=right_rise_height,
        left_rise_out=left_rise_out,
        right_rise_out=right_rise_out,
        prayer_gate=prayer_gate,
        breath_gate=breath_gate,
        snap_gate_side=str(snap_gate["active_side"]),
    )


@dataclass
class InteractionController:
    ready_gate: bool = False
    state: str = "idle"
    candidate: str = "none"
    active_label: str = "none"
    arming_count: int = 0
    arming_started_at: float = 0.0
    missing_count: int = 0
    cooldown_count: int = 0
    rival_candidate: str = "none"
    rival_count: int = 0
    rival_started_at: float = 0.0
    active_frames: int = 0
    open_base_distance: float = 0.0
    rise_base_height: float = 0.0
    rise_side: str = "none"
    pd_decay_prev: float = 0.0
    arming_reference: tuple[float, ...] = field(default_factory=tuple)
    arming_grace_count: int = 0

    def _eligible_label(
        self,
        prediction: str,
        signals: RuntimeSignals,
        face_present: bool,
        snap_event_detected: bool,
    ) -> str:
        if not face_present:
            return "none"
        if snap_event_detected:
            return "snap" if signals.snap_gate_side != "none" else "none"
        if prediction in {"waiting", "none"}:
            return "none"
        if not self.ready_gate:
            return "none"
        if prediction == "rise":
            return "rise" if signals.rise_side != "none" else "none"
        if prediction == "prayer":
            return "prayer" if signals.prayer_gate else "none"
        if prediction == "breath":
            return "breath" if signals.breath_gate else "none"
        if prediction == "open":
            return "open" if signals.open_ready else "none"
        if prediction == "uncertain":
            if self.state == "arming" and self.candidate == "open" and signals.open_ready:
                return "open"
            if self.state == "arming" and self.candidate == "rise" and signals.rise_side != "none":
                return "rise"
            if self.state == "arming" and self.candidate == "prayer" and signals.prayer_gate:
                return "prayer"
            if self.state == "arming" and self.candidate == "breath" and signals.breath_gate:
                return "breath"

            open_fallback_ok = (
                signals.open_ready
                and not signals.prayer_gate
                and not signals.breath_gate
                and signals.rise_side == "none"
            )
            if open_fallback_ok:
                return "open"
            return "none"
        return "none"

    def _stability_signature(self, label: str, signals: RuntimeSignals) -> tuple[float, ...]:
        if label == "open":
            return (signals.open_distance, signals.wrist_center_x, signals.wrist_center_y)
        if label == "rise":
            side_code = 1.0 if signals.rise_side == "left" else 2.0 if signals.rise_side == "right" else 0.0
            return (side_code, signals.rise_height_for_side(signals.rise_side))
        if label == "prayer":
            return (signals.open_distance, signals.wrist_center_x, signals.wrist_center_y)
        if label == "breath":
            return (signals.open_distance, signals.wrist_center_x, signals.wrist_center_y)
        return ()

    def _pose_stable_for_candidate(self, label: str, signals: RuntimeSignals) -> bool:
        if label == "snap":
            return True
        reference = self.arming_reference
        current = self._stability_signature(label, signals)
        limits = ARMING_STABILITY_LIMITS.get(label)
        if not reference or not current or limits is None or len(reference) != len(current):
            self.arming_reference = current
            return True
        return all(abs(now - start) <= limit for now, start, limit in zip(current, reference, limits))

    def _set_candidate(self, label: str, signals: RuntimeSignals, now: float) -> None:
        self.candidate = label
        self.arming_count = 1
        self.arming_started_at = now
        self.arming_reference = self._stability_signature(label, signals)
        self.arming_grace_count = 0

    def _activate(self, label: str, signals: RuntimeSignals) -> list[str]:
        self.state = "active"
        self.active_label = label
        self.active_frames = 0
        self.missing_count = 0
        self.rival_candidate = "none"
        self.rival_count = 0
        self.rival_started_at = 0.0

        if label == "open":
            self.open_base_distance = signals.open_distance
        elif label == "rise":
            self.rise_side = signals.rise_side
            self.rise_base_height = signals.rise_height_for_side(self.rise_side)

        self.candidate = "none"
        self.arming_count = 0
        self.arming_started_at = 0.0
        self.arming_reference = ()
        return [f"{label}_started"]

    def _toggle_ready_gate(self, release_active_label: str | None = None) -> list[str]:
        events: list[str] = []
        if release_active_label not in (None, "none"):
            events.append(f"{release_active_label}_ended")

        self.ready_gate = not self.ready_gate
        self.state = "cooldown"
        self.cooldown_count = COOLDOWN_FRAMES.get("snap", 12)
        self.candidate = "none"
        self.active_label = "none"
        self.arming_count = 0
        self.arming_started_at = 0.0
        self.missing_count = 0
        self.rival_candidate = "none"
        self.rival_count = 0
        self.rival_started_at = 0.0
        self.active_frames = 0
        self.open_base_distance = 0.0
        self.rise_base_height = 0.0
        self.rise_side = "none"
        self.pd_decay_prev = 0.0
        self.arming_reference = ()
        self.arming_grace_count = 0
        gate_event = "ready_enabled" if self.ready_gate else "ready_disabled"
        events.extend(["snap_started", gate_event, "snap_ended"])
        return events

    def _release(self, reason_label: str) -> list[str]:
        self.state = "cooldown"
        self.cooldown_count = COOLDOWN_FRAMES.get(reason_label, 8)
        self.active_label = "none"
        self.candidate = "none"
        self.arming_count = 0
        self.arming_started_at = 0.0
        self.missing_count = 0
        self.rival_candidate = "none"
        self.rival_count = 0
        self.rival_started_at = 0.0
        self.active_frames = 0
        self.rise_side = "none"
        self.arming_reference = ()
        self.arming_grace_count = 0
        return [f"{reason_label}_ended"]

    def update(
        self,
        prediction: str,
        face_present: bool,
        signals: RuntimeSignals,
        snap_event_detected: bool = False,
        now: float | None = None,
    ) -> list[str]:
        if now is None:
            now = time.time()
        events: list[str] = []
        eligible = self._eligible_label(prediction, signals, face_present, snap_event_detected)

        if self.state == "idle":
            if eligible != "none":
                self.state = "arming"
                self._set_candidate(eligible, signals, now)
            return events

        if self.state == "arming":
            if eligible == "none":
                grace_limit = ARMING_GRACE_FRAMES.get(self.candidate, 0)
                if grace_limit > 0 and self.arming_grace_count < grace_limit:
                    self.arming_grace_count += 1
                    return events
                self.state = "idle"
                self.candidate = "none"
                self.arming_count = 0
                self.arming_started_at = 0.0
                self.arming_reference = ()
                self.arming_grace_count = 0
                return events
            self.arming_grace_count = 0

            if eligible == self.candidate:
                if self._pose_stable_for_candidate(eligible, signals):
                    self.arming_count += 1
                else:
                    self._set_candidate(eligible, signals, now)
                    return events
            else:
                self._set_candidate(eligible, signals, now)

            hold_time = now - self.arming_started_at
            if hold_time >= ARMING_SECONDS.get(self.candidate, 0.5):
                if self.candidate == "snap":
                    return self._toggle_ready_gate()
                return self._activate(self.candidate, signals)
            return events

        if self.state == "active":
            self.active_frames += 1

            if eligible == "snap":
                return self._toggle_ready_gate(release_active_label=self.active_label)

            release_condition = False

            if not face_present:
                release_condition = True
            elif self.active_label == "open":
                release_condition = signals.open_release
            elif self.active_label == "rise":
                release_condition = (
                    self.active_frames > RISE_RELEASE_GRACE_FRAMES
                    and signals.rise_out_for_side(self.rise_side)
                )
            elif self.active_label == "prayer":
                release_condition = not signals.prayer_gate
            elif self.active_label == "breath":
                release_condition = not signals.breath_gate

            if release_condition:
                self.missing_count += 1
                self.rival_candidate = "none"
                self.rival_count = 0
                self.rival_started_at = 0.0
            elif self.active_label in {"open", "rise"}:
                self.missing_count = 0
                self.rival_candidate = "none"
                self.rival_count = 0
                self.rival_started_at = 0.0
            elif prediction == self.active_label or prediction == "uncertain":
                self.missing_count = 0
                self.rival_candidate = "none"
                self.rival_count = 0
                self.rival_started_at = 0.0
            elif eligible != "none" and eligible != self.active_label:
                if eligible == self.rival_candidate:
                    self.rival_count += 1
                else:
                    self.rival_candidate = eligible
                    self.rival_count = 1
                    self.rival_started_at = now

                rival_hold_time = now - self.rival_started_at
                if (
                    rival_hold_time >= ACTIVE_RIVAL_SECONDS
                    and self.rival_count >= RIVAL_LABEL_FRAMES.get(self.active_label, 3)
                ):
                    self.missing_count += 1
                else:
                    self.missing_count = 0
            else:
                self.missing_count += 1
                self.rival_candidate = "none"
                self.rival_count = 0
                self.rival_started_at = 0.0

            if self.missing_count >= RELEASE_FRAMES.get(self.active_label, 4):
                return self._release(self.active_label)
            return events

        if self.state == "cooldown":
            self.cooldown_count -= 1
            if self.cooldown_count <= 0:
                self.state = "idle"
                self.cooldown_count = 0
            return events

        return events

    def current_vad(self, signals: RuntimeSignals) -> tuple[float, float, float]:
        if self.state != "active":
            return (0.0, 0.0, 0.0)

        if self.active_label == "open":
            open_amount = clamp((signals.open_distance - self.open_base_distance) / OPEN_DISTANCE_SCALE, -1.0, 1.0)
            return (0.7 * open_amount, 0.1 * open_amount, 0.7 * open_amount)

        if self.active_label == "rise":
            current_height = signals.rise_height_for_side(self.rise_side)
            rise_amount = clamp((current_height - self.rise_base_height) / RISE_HEIGHT_SCALE, -1.0, 1.0)
            return (0.2 * rise_amount, 0.8 * rise_amount, 0.2 * rise_amount)

        if self.active_label == "prayer":
            progress = clamp(self.active_frames / PRAYER_FULL_FRAMES, 0.0, 1.0)
            return (0.2 * progress, -0.2 * progress, 0.2 * progress)

        if self.active_label == "breath":
            progress = clamp(self.active_frames / BREATH_FULL_FRAMES, 0.0, 1.0)
            return (0.0, -0.6 * progress, 0.0)

        return (0.0, 0.0, 0.0)

    def current_delta_vad(self, signals: RuntimeSignals) -> tuple[float, float, float]:
        if self.state != "active":
            return (0.0, 0.0, 0.0)
        if self.active_label in {"prayer", "breath"}:
            return (0.0, 0.0, 0.0)
        return self.current_vad(signals)

    def current_recovery_progress(self) -> float:
        if self.state != "active":
            return 0.0
        if self.active_label == "prayer":
            return clamp(self.active_frames / PRAYER_FULL_FRAMES, 0.0, 1.0)
        if self.active_label == "breath":
            return clamp(self.active_frames / BREATH_FULL_FRAMES, 0.0, 1.0)
        return 0.0

    def vad_source_description(self, signals: RuntimeSignals) -> str:
        if self.state == "arming":
            hold_time = max(0.0, time.time() - self.arming_started_at)
            needed = ARMING_SECONDS.get(self.candidate, 0.5)
            return f"arming {self.candidate} {hold_time:.1f}/{needed:.1f}s"
        if self.state != "active":
            return self.state
        if self.active_label == "open":
            amount = clamp((signals.open_distance - self.open_base_distance) / OPEN_DISTANCE_SCALE, -1.0, 1.0)
            return f"active open amount={amount:+.2f}"
        if self.active_label == "rise":
            current_height = signals.rise_height_for_side(self.rise_side)
            amount = clamp((current_height - self.rise_base_height) / RISE_HEIGHT_SCALE, -1.0, 1.0)
            return f"active rise side={self.rise_side} amount={amount:+.2f}"
        if self.active_label == "prayer":
            progress = clamp(self.active_frames / PRAYER_FULL_FRAMES, 0.0, 1.0)
            return f"active prayer progress={progress:.2f}"
        if self.active_label == "breath":
            progress = clamp(self.active_frames / BREATH_FULL_FRAMES, 0.0, 1.0)
            return f"active breath progress={progress:.2f}"
        return f"active {self.active_label}"


def draw_skeleton(frame: np.ndarray, landmarks: dict[str, tuple[float, float]]) -> None:
    h, w = frame.shape[:2]
    for a, b in SKELETON_PAIRS:
        if a in landmarks and b in landmarks:
            ax, ay = int(landmarks[a][0] * w), int(landmarks[a][1] * h)
            bx, by = int(landmarks[b][0] * w), int(landmarks[b][1] * h)
            cv2.line(frame, (ax, ay), (bx, by), (0, 255, 0), 2)
    for name, (x, y) in landmarks.items():
        px, py = int(x * w), int(y * h)
        color = JOINT_COLORS.get(name, (200, 200, 200))
        cv2.circle(frame, (px, py), 5, color, -1)


def draw_vad_meter(frame: np.ndarray, label: str, value: float, x: int, y: int, color: tuple[int, int, int]) -> None:
    width = 260
    height = 16
    half = width // 2
    value = clamp(value, -1.0, 1.0)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 40, 40), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (160, 160, 160), 1)
    cv2.line(frame, (x + half, y - 2), (x + half, y + height + 2), (220, 220, 220), 1)
    if value >= 0:
        end_x = x + half + int(value * half)
        cv2.rectangle(frame, (x + half, y), (end_x, y + height), color, -1)
    else:
        start_x = x + half + int(value * half)
        cv2.rectangle(frame, (start_x, y), (x + half, y + height), color, -1)
    cv2.putText(
        frame,
        f"{label} {value:+.2f}",
        (x, y - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime kNN inference for interaction3")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", default="interaction3/models/knn_model_v3_uncertain.joblib")
    parser.add_argument("--snap-model", default="interaction3/models/snap_model_v1.joblib")
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--snap-window-size", type=int, default=8)
    parser.add_argument("--vote-size", type=int, default=5)
    parser.add_argument("--send", action="store_true", help="Enable sending data to UE and PD bridges")
    parser.add_argument("--eeg-result", help="Path to EEG result.json or a job directory that contains result.json")
    parser.add_argument("--eeg-jobs-root", default="eeg-emotions/artifacts/jobs")
    parser.add_argument("--eeg-min-valid-window-ratio", type=float, default=0.70)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model = load_model(model_path)
    snap_model_path = Path(args.snap_model)
    snap_model = load_model(snap_model_path) if snap_model_path.exists() else None
    signal_filter = RuntimeSignalFilter()
    snap_gate_smoother = SnapGateSmoother()
    rise_side_smoother = RiseSideSmoother()
    cap = cv2.VideoCapture(args.camera)

    ue = UEBridge(enabled=args.send)
    pd = PDBridge(enabled=args.send)
    ue.start()
    pd.start()

    if not cap.isOpened():
        print(f"[ERROR] camera {args.camera} could not be opened")
        ue.stop()
        pd.stop()
        sys.exit(1)

    extractor = PoseExtractor()
    hand_extractor = HandExtractor()
    window = SlidingWindowBuffer(window_size=args.window_size)
    snap_window = SlidingWindowBuffer(window_size=args.snap_window_size)
    previous = None
    recent_predictions: deque[str] = deque(maxlen=args.vote_size)
    controller = InteractionController()
    face_history: deque[bool] = deque(maxlen=5)
    tracking_locked = False
    tracking_state = "searching"
    lock_anchor_center: tuple[float, float] | None = None
    lock_anchor_torso_height: float | None = None
    lost_counter = 0
    back_counter = 0
    body_mismatch_counter = 0
    last_snap_time = 0.0
    last_ready_gate_for_prediction = False
    eeg_base_state = BaseVADState()
    base_memory_state = BaseMemoryState()
    world_vad_state = WorldVADState()
    ready_off_started_at: float | None = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            frame_now = time.time()

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = extractor.process(frame_rgb)
            hand_results = hand_extractor.process(frame_rgb)
            landmarks = extractor.extract_upper_body(pose_results)
            face_valid = extractor.detect_face_presence(landmarks)
            tracking_metrics = compute_tracking_metrics(landmarks)
            face_history.append(face_valid)
            face_present = False

            if not tracking_locked:
                face_present = sum(face_history) >= 3 and bool(tracking_metrics["torso_present"])
                if face_present:
                    tracking_locked = True
                    tracking_state = "locked"
                    lost_counter = 0
                    back_counter = 0
                    body_mismatch_counter = 0
                    lock_anchor_center = tracking_metrics["shoulder_center"]  # type: ignore[assignment]
                    lock_anchor_torso_height = float(tracking_metrics["torso_height"])  # type: ignore[arg-type]
            else:
                torso_present = bool(tracking_metrics["torso_present"])
                if not torso_present:
                    lost_counter += 1
                else:
                    lost_counter = max(0, lost_counter - 1)

                head_count = int(tracking_metrics["head_count"])
                front_face_count = int(tracking_metrics["front_face_count"])
                if torso_present and front_face_count == 0 and head_count >= 1 and not face_valid:
                    back_counter += 1
                else:
                    back_counter = max(0, back_counter - 1)

                if torso_present and lock_anchor_center is not None and lock_anchor_torso_height is not None:
                    current_center = tracking_metrics["shoulder_center"]
                    current_torso_height = float(tracking_metrics["torso_height"])  # type: ignore[arg-type]
                    if current_center is not None:
                        center_dx = current_center[0] - lock_anchor_center[0]
                        center_dy = current_center[1] - lock_anchor_center[1]
                        center_dist = (center_dx * center_dx + center_dy * center_dy) ** 0.5
                        torso_ratio = current_torso_height / max(lock_anchor_torso_height, 1e-5)
                        if center_dist > 1.8 or torso_ratio < 0.35 or torso_ratio > 2.8:
                            body_mismatch_counter += 1
                        else:
                            body_mismatch_counter = max(0, body_mismatch_counter - 1)

                        if face_valid:
                            lock_anchor_center = (
                                lock_anchor_center[0] * 0.9 + current_center[0] * 0.1,
                                lock_anchor_center[1] * 0.9 + current_center[1] * 0.1,
                            )
                            lock_anchor_torso_height = lock_anchor_torso_height * 0.9 + current_torso_height * 0.1
                else:
                    body_mismatch_counter += 1

                if lost_counter >= 8 or back_counter >= 15 or body_mismatch_counter >= 12:
                    tracking_locked = False
                    tracking_state = "searching"
                    face_present = False
                    face_history.clear()
                    window.clear()
                    snap_window.clear()
                    previous = None
                    recent_predictions.clear()
                    lock_anchor_center = None
                    lock_anchor_torso_height = None
                    lost_counter = 0
                    back_counter = 0
                    body_mismatch_counter = 0
                    controller = InteractionController()
                    snap_gate_smoother = SnapGateSmoother()
                    rise_side_smoother = RiseSideSmoother()
                    last_ready_gate_for_prediction = False
                else:
                    face_present = True
                    if lost_counter > 0 or back_counter > 0 or body_mismatch_counter > 0:
                        tracking_state = "lost_pending"
                    else:
                        tracking_state = "locked"

            normalized = None
            signals = RuntimeSignals()
            raw_snap_gate_side = "none"
            raw_rise_side = "none"
            if landmarks is not None and face_present:
                landmarks.update(hand_extractor.extract_keypoints(hand_results))
                draw_skeleton(frame, landmarks)
                normalized = normalize_landmarks(landmarks)
                if normalized is not None:
                    signals = signal_filter.apply(build_runtime_signals(normalized), frame_now)
                    raw_rise_side = signals.rise_side
                    signals.rise_side = rise_side_smoother.update(raw_rise_side)
                    raw_snap_gate_side = signals.snap_gate_side
                    signals.snap_gate_side = snap_gate_smoother.update(raw_snap_gate_side)
                    frame_features = compute_frame_features(normalized, previous)
                    window.add(frame_features)
                    snap_features = compute_snap_frame_features(normalized, previous, signals.snap_gate_side)
                    snap_window.add(snap_features)
            else:
                window.clear()
                snap_window.clear()
                previous = None

            prediction = "waiting"
            payload = None
            interaction_events: list[str] = []
            snap_event_detected = False

            snap_event = {
                "event": False,
                "raw_event": False,
                "thumb_index_distance": 0.0,
                "thumb_index_delta": 0.0,
                "thumb_middle_distance": 0.0,
                "thumb_middle_delta": 0.0,
                "thumb_crossed": False,
                "index_extension": 0.0,
                "middle_motion": 0.0,
                "ml_prediction": "waiting",
                "ml_confidence": 0.0,
            }
            if normalized is not None:
                raw_snap_event = detect_snap_event(normalized, previous, signals.snap_gate_side)
                snap_event.update(
                    {
                        "raw_event": bool(raw_snap_event["event"]),
                        "thumb_index_distance": float(raw_snap_event["thumb_index_distance"]),
                        "thumb_middle_distance": float(raw_snap_event["thumb_middle_distance"]),
                        "thumb_index_delta": float(raw_snap_event["thumb_index_delta"]),
                        "thumb_middle_delta": float(raw_snap_event["thumb_middle_delta"]),
                        "middle_motion": float(raw_snap_event["middle_motion"]),
                        "index_extension": float(raw_snap_event["index_extension"]),
                        "thumb_crossed": bool(raw_snap_event["thumb_crossed"]),
                    }
                )

            if normalized is not None and snap_model is not None and snap_window.is_ready():
                snap_vector = snap_window.to_feature_vector().reshape(1, -1)
                snap_probs = snap_model.predict_proba(snap_vector)[0]
                snap_confidence = float(np.max(snap_probs))
                snap_prediction = str(snap_model.predict(snap_vector)[0])
                snap_event["ml_prediction"] = snap_prediction
                snap_event["ml_confidence"] = snap_confidence
                snap_event["event"] = bool(
                    snap_event["raw_event"]
                    or (snap_prediction == "snap" and snap_confidence >= 0.5)
                )
            else:
                snap_event["event"] = bool(snap_event["raw_event"])

            if face_present and window.is_ready():
                if controller.ready_gate != last_ready_gate_for_prediction:
                    recent_predictions.clear()
                    last_ready_gate_for_prediction = controller.ready_gate

                feature_vector = window.to_feature_vector().reshape(1, -1)
                expected_dim = getattr(model, "n_features_in_", None)
                if expected_dim is None and hasattr(model, "named_steps") and "knn" in model.named_steps:
                    expected_dim = getattr(model.named_steps["knn"], "n_features_in_", None)
                if expected_dim is not None and feature_vector.shape[1] != expected_dim:
                    raise ValueError(
                        f"Model expects {expected_dim} features, but current extractor produced {feature_vector.shape[1]}. "
                        "Please recollect data and retrain the model for the new snap-gate pipeline."
                    )

                probs = model.predict_proba(feature_vector)[0]
                pred_label, max_prob = masked_prediction_from_probs(
                    model,
                    probs,
                    ready_gate=controller.ready_gate,
                    threshold=0.5,
                )

                if max_prob < 0.5:
                    voted_prediction = "uncertain"
                else:
                    recent_predictions.append(pred_label)
                    voted_prediction = Counter(recent_predictions).most_common(1)[0][0]

                prediction = voted_prediction
                snap_event_detected = bool(snap_event["event"])
                interaction_events = controller.update(voted_prediction, face_present, signals, snap_event_detected, now=frame_now)
            else:
                interaction_events = controller.update("none", face_present, signals, False, now=frame_now)

            eeg_profile = load_eeg_profile(
                result_path=args.eeg_result,
                jobs_root=args.eeg_jobs_root,
                min_valid_window_ratio=args.eeg_min_valid_window_ratio,
            )
            eeg_base_state = resolve_base_vad(eeg_profile, previous=eeg_base_state)

            eeg_base_vad = (eeg_base_state.v, eeg_base_state.a, eeg_base_state.d)
            base_vad = base_memory_state.apply_to(eeg_base_vad)
            interaction_delta_vad = controller.current_delta_vad(signals)
            recovery_progress = controller.current_recovery_progress()

            if not world_vad_state.initialized:
                world_vad_state.set_from_tuple(base_vad)

            if controller.ready_gate:
                ready_off_started_at = None
            elif ready_off_started_at is None:
                ready_off_started_at = frame_now

            if controller.state == "active" and controller.active_label in {"open", "rise"}:
                active_target = combine_base_and_delta_vad(base_vad, interaction_delta_vad)
                world_vad_state.set_from_tuple(
                    blend_triplet(world_vad_state.as_tuple(), active_target, ACTIVE_TARGET_ALPHA)
                )
            elif controller.state == "active" and controller.active_label == "prayer":
                alpha = lerp(PRAYER_RECOVERY_ALPHA_MIN, PRAYER_RECOVERY_ALPHA_MAX, recovery_progress)
                world_vad_state.set_from_tuple(
                    blend_triplet(world_vad_state.as_tuple(), base_vad, alpha)
                )
            elif controller.state == "active" and controller.active_label == "breath":
                alpha = lerp(BREATH_RECOVERY_ALPHA_MIN, BREATH_RECOVERY_ALPHA_MAX, recovery_progress)
                world_vad_state.set_from_tuple(
                    blend_triplet(world_vad_state.as_tuple(), base_vad, alpha)
                )
            elif (
                not controller.ready_gate
                and ready_off_started_at is not None
                and frame_now - ready_off_started_at >= READY_OFF_RECOVERY_DELAY_SECONDS
            ):
                recovery_alpha = 1.0 / max(READY_OFF_RECOVERY_DURATION_SECONDS * 30.0, 1.0)
                world_vad_state.set_from_tuple(
                    blend_triplet(world_vad_state.as_tuple(), base_vad, recovery_alpha)
                )

            final_vad = world_vad_state.as_tuple()

            raw_payload = compute_unreal_payload(*final_vad)
            payload = raw_payload.as_dict()
            pd_values, _ = compute_pd_continuous_values(controller, signals, raw_payload)
            vad_source = controller.vad_source_description(signals)

            if "snap_started" in interaction_events:
                now = time.time()
                if now - last_snap_time > 1.0:
                    last_snap_time = now
                    gate_state = "READY ON" if controller.ready_gate else "READY OFF"
                    print(f"[{now:.2f}] ML SNAP DETECTED! -> {gate_state}")

            if "ready_disabled" in interaction_events:
                base_memory_state.learn_from(eeg_base_vad, final_vad)

            if args.send:
                if "ready_enabled" in interaction_events:
                    pd.send_trigger("READY_ON")
                if "ready_disabled" in interaction_events:
                    pd.send_trigger("READY_OFF")

                pd.send(raw_payload)
                for symbol, value in pd_values.items():
                    pd.send_value(symbol, value)

            if args.send and controller.state == "active":
                ue.send(raw_payload)

            cv2.putText(frame, f"prediction={prediction}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"WORLD TARGET VAD  V={payload['V']:+.2f} A={payload['A']:+.2f} D={payload['D']:+.2f}", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"EEG RAW BASE  V={eeg_base_vad[0]:+.2f} A={eeg_base_vad[1]:+.2f} D={eeg_base_vad[2]:+.2f} usable={eeg_base_state.usable} reason={eeg_base_state.reason}", (20, 462),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 200, 120), 2, cv2.LINE_AA)
            cv2.putText(frame, f"ADAPTIVE BASE  V={base_vad[0]:+.2f} A={base_vad[1]:+.2f} D={base_vad[2]:+.2f} mem=({base_memory_state.dv:+.3f},{base_memory_state.da:+.3f},{base_memory_state.dd:+.3f})", (20, 488),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 220, 160), 2, cv2.LINE_AA)
            cv2.putText(frame, f"INTERACTION DELTA  V={interaction_delta_vad[0]:+.2f} A={interaction_delta_vad[1]:+.2f} D={interaction_delta_vad[2]:+.2f} rec={recovery_progress:.2f}", (20, 514),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"vad_source={vad_source}", (20, 430),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"UE flower={payload['flower']:.2f} tree={payload['tree']:.2f} decay={payload['decay']:.2f} fountain={payload['min_speed']:.0f}-{payload['max_speed']:.0f}", (20, 540),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2, cv2.LINE_AA)
            cv2.putText(frame, f"PD ready={pd_values['READY_MODE']:.0f} flower={pd_values['FLOWER']:.2f} decay={pd_values['DECAY']:.2f} recover={pd_values['RECOVER']:.2f}", (20, 566),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2, cv2.LINE_AA)
            meter_x = max(20, frame.shape[1] - 310)
            draw_vad_meter(frame, "V", payload["V"], meter_x, 42, (80, 220, 255))
            draw_vad_meter(frame, "A", payload["A"], meter_x, 88, (80, 255, 120))
            draw_vad_meter(frame, "D", payload["D"], meter_x, 134, (255, 180, 80))
            cv2.putText(frame, f"ready={controller.ready_gate} interaction={controller.state} active={controller.active_label} candidate={controller.candidate}", (20, 108),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            arming_elapsed = max(0.0, time.time() - controller.arming_started_at) if controller.state == "arming" else 0.0
            cv2.putText(frame, f"arming={controller.arming_count} arming_elapsed={arming_elapsed:.2f}s missing={controller.missing_count} rival={controller.rival_candidate}:{controller.rival_count}", (20, 141),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"face_valid={face_valid}", (20, 174), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (120, 255, 120) if face_valid else (0, 120, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"track={tracking_state} face_present={face_present} ({sum(face_history)}/{len(face_history)})", (20, 207),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 255, 120) if face_present else (0, 120, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"lost={lost_counter} back={back_counter} drift={body_mismatch_counter}", (20, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"front_face={tracking_metrics['front_face_count']} head={tracking_metrics['head_count']}", (20, 273),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"open_ready={signals.open_ready} open_release={signals.open_release} rise_raw={raw_rise_side} rise_side={signals.rise_side} prayer={signals.prayer_gate} breath={signals.breath_gate} snap_gate={signals.snap_gate_side}",
                        (20, 306), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2, cv2.LINE_AA)
            if normalized is not None:
                snap_debug = detect_snap_gate(normalized)
                cv2.putText(
                    frame,
                    f"snap_event={snap_event_detected} raw={snap_event['raw_event']} ml={snap_event['ml_prediction']} conf={snap_event['ml_confidence']:.2f} "
                    f"final_snap={snap_event_detected and signals.snap_gate_side != 'none'} "
                    f"L={snap_debug['left_gate']} R={snap_debug['right_gate']} "
                    f"Lup={snap_debug['left_above']} Rup={snap_debug['right_above']} "
                    f"Lclear={snap_debug['left_clear_side']} Rclear={snap_debug['right_clear_side']} "
                    f"Lprep={snap_debug['left_prep_contact']} Rprep={snap_debug['right_prep_contact']}",
                    (20, 339),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 220, 120),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"ready={controller.ready_gate} snap_gate_raw={raw_snap_gate_side} snap_gate_side={signals.snap_gate_side} "
                    f"Ltm={snap_debug['left_thumb_middle_distance']:.3f} Rtm={snap_debug['right_thumb_middle_distance']:.3f}",
                    (20, 368),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (180, 220, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"ti={snap_event['thumb_index_distance']:.3f} tid={snap_event['thumb_index_delta']:.3f} "
                    f"tm={snap_event['thumb_middle_distance']:.3f} tmd={snap_event['thumb_middle_delta']:.3f} "
                    f"midmv={snap_event['middle_motion']:.3f} crossed={snap_event['thumb_crossed']} "
                    f"ext={snap_event['index_extension']:.3f}",
                    (20, 397),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (180, 220, 255),
                    2,
                    cv2.LINE_AA,
                )
            if time.time() - last_snap_time < 0.5:
                cv2.putText(frame, "SNAP FIRED!", (240, 100), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)

            cv2.imshow("interaction3 inference", frame)

            if normalized is not None and face_present:
                previous = normalized

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty("interaction3 inference", cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        ue.stop()
        pd.stop()
        cap.release()
        extractor.close()
        hand_extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
