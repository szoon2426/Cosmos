from __future__ import annotations

from dataclasses import dataclass

if __package__ in (None, ""):
    from conduct_features import ConductFeatures
    from conduct_state import ConductSession
else:
    from .conduct_features import ConductFeatures
    from .conduct_state import ConductSession


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def remap_clamped(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    if in_max == in_min:
        return out_min
    t = (value - in_min) / (in_max - in_min)
    t = clamp(t, 0.0, 1.0)
    return out_min + (out_max - out_min) * t


def blend(current: float, target: float, alpha: float) -> float:
    return current + (target - current) * clamp(alpha, 0.0, 1.0)


@dataclass(slots=True)
class ConductDebug:
    index_dx: float = 0.0
    index_dy: float = 0.0
    left_delta_open: float = 0.0
    d_velocity: float = 0.0
    v_target: float = 0.0
    a_target: float = 0.0
    d_target: float = 0.0


def compute_conduct_vad(
    base_vad: tuple[float, float, float],
    current_world_vad: tuple[float, float, float],
    session: ConductSession,
    features: ConductFeatures,
    dt: float,
) -> tuple[tuple[float, float, float], ConductDebug]:
    debug = ConductDebug()

    if not session.is_active():
        debug.v_target = base_vad[0]
        debug.a_target = base_vad[1]
        debug.d_target = base_vad[2]
        return current_world_vad, debug

    debug.index_dx = features.right_index_x - session.base_right_index_x
    debug.index_dy = features.right_index_y - session.base_right_index_y

    v_delta = remap_clamped(debug.index_dx, -0.9, 0.9, -1.0, 1.0) * 0.9
    a_delta = remap_clamped(-debug.index_dy, -0.9, 0.9, -1.0, 1.0) * 0.9
    debug.v_target = clamp(base_vad[0] + v_delta, -1.0, 1.0)
    debug.a_target = clamp(base_vad[1] + a_delta, -1.0, 1.0)

    if features.left_hand_open_valid:
        debug.left_delta_open = features.left_hand_open - session.base_left_hand_open
    else:
        debug.left_delta_open = 0.0

    if features.left_fist_closed or features.left_fist_score >= 0.45:
        debug.d_velocity = -remap_clamped(features.left_fist_score, 0.45, 0.75, 0.35, 1.0)
    else:
        abs_delta = abs(debug.left_delta_open)
        if abs_delta < 0.05:
            debug.d_velocity = 0.0
        else:
            speed = remap_clamped(abs_delta, 0.05, 0.25, 0.0, 1.0)
            debug.d_velocity = speed if debug.left_delta_open > 0.0 else -speed

    debug.d_target = clamp(current_world_vad[2] + debug.d_velocity * 0.75 * max(dt, 0.0), -1.0, 1.0)

    next_vad = (
        blend(current_world_vad[0], debug.v_target, 0.18),
        blend(current_world_vad[1], debug.a_target, 0.18),
        debug.d_target,
    )
    return next_vad, debug
