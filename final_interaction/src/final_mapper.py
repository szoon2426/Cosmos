from __future__ import annotations

from dataclasses import dataclass


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def remap(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    if in_max == in_min:
        return out_min
    t = (value - in_min) / (in_max - in_min)
    return out_min + (out_max - out_min) * t


@dataclass(slots=True)
class FinalInteractionPayload:
    interaction_active: float
    pointer_x: float
    pointer_y: float
    target_v: float
    target_a: float
    target_d: float
    decay: float
    density: float
    min_speed: float
    max_speed: float
    grab_active: float
    switch_to_camera: float

    def as_preset_properties(self) -> dict[str, float]:
        return {
            "Interaction Active": self.interaction_active,
            "Pointer X": self.pointer_x,
            "Pointer Y": self.pointer_y,
            "Target V": self.target_v,
            "Target A": self.target_a,
            "Target D": self.target_d,
            "Density": self.density,
            "Decay Amount": self.decay,
            "Min Speed": self.min_speed,
            "Max Speed": self.max_speed,
            "Grab Active": self.grab_active,
            "Switch To Camera": self.switch_to_camera,
        }


def compute_final_payload(
    *,
    interaction_active: bool,
    pointer_x: float,
    pointer_y: float,
    target_v: float,
    target_a: float,
    target_d: float,
    grab_active: bool,
    open_strength: float,
    grab_strength: float,
    switch_to_camera: bool,
) -> FinalInteractionPayload:
    target_v = clamp(target_v, -1.0, 1.0)
    target_a = clamp(target_a, -1.0, 1.0)
    target_d = clamp(target_d, -1.0, 1.0)

    decay = clamp(0.5 - 0.3 * target_v + 0.3 * target_a, 0.0, 1.2)
    density = clamp(remap(0.5 * target_v + 0.5 * target_d, -1.0, 1.0, 0.4, 1.0), 0.4, 1.0)
    min_speed = clamp(remap(target_a, -1.0, 1.0, 0.0, 580.0), 0.0, 600.0)
    max_speed = clamp(min_speed + 20.0, 0.0, 600.0)

    return FinalInteractionPayload(
        interaction_active=1.0 if interaction_active else 0.0,
        pointer_x=clamp(pointer_x, -1.0, 1.5),
        pointer_y=clamp(pointer_y, -1.0, 1.5),
        target_v=target_v,
        target_a=target_a,
        target_d=target_d,
        decay=decay,
        density=density,
        min_speed=min_speed,
        max_speed=max_speed,
        grab_active=1.0 if grab_active else 0.0,
        switch_to_camera=1.0 if switch_to_camera else 0.0,
    )
