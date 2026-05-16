from __future__ import annotations

from dataclasses import dataclass


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


@dataclass(slots=True)
class FinalInteractionPayload:
    interaction_active: float
    pointer_x: float
    pointer_y: float
    target_v: float
    target_a: float
    target_d: float
    grab_active: float
    open_strength: float
    grab_strength: float
    switch_to_camera: float

    def as_preset_properties(self) -> dict[str, float]:
        return {
            "Interaction Active": self.interaction_active,
            "Pointer X": self.pointer_x,
            "Pointer Y": self.pointer_y,
            "Target V": self.target_v,
            "Target A": self.target_a,
            "Target D": self.target_d,
            "Grab Active": self.grab_active,
            "Open Strength": self.open_strength,
            "Grab Strength": self.grab_strength,
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
    return FinalInteractionPayload(
        interaction_active=1.0 if interaction_active else 0.0,
        pointer_x=clamp(pointer_x, -1.0, 1.5),
        pointer_y=clamp(pointer_y, -1.0, 1.5),
        target_v=clamp(target_v, -1.0, 1.0),
        target_a=clamp(target_a, -1.0, 1.0),
        target_d=clamp(target_d, -1.0, 1.0),
        grab_active=1.0 if grab_active else 0.0,
        open_strength=clamp(open_strength, 0.0, 1.0),
        grab_strength=clamp(grab_strength, 0.0, 1.0),
        switch_to_camera=1.0 if switch_to_camera else 0.0,
    )
