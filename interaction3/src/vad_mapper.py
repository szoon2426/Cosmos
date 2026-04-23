from __future__ import annotations

from dataclasses import dataclass


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def remap(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    if in_max == in_min:
        return out_min
    t = (value - in_min) / (in_max - in_min)
    return out_min + (out_max - out_min) * t


@dataclass
class UnrealPayload:
    v: float
    a: float
    d: float
    decay: float
    density: float
    flower: float
    tree: float
    min_speed: float
    max_speed: float
    pointer_active: float = 0.0
    pointer_x: float = 0.5
    pointer_y: float = 0.5
    pointer_speed: float = 0.0
    pointer_strength: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "V": self.v,
            "A": self.a,
            "D": self.d,
            "decay": self.decay,
            "density": self.density,
            "flower": self.flower,
            "tree": self.tree,
            "min_speed": self.min_speed,
            "max_speed": self.max_speed,
            "pointer_active": self.pointer_active,
            "pointer_x": self.pointer_x,
            "pointer_y": self.pointer_y,
            "pointer_speed": self.pointer_speed,
            "pointer_strength": self.pointer_strength,
        }

    def as_preset_properties(self) -> dict[str, float]:
        return {
            "Target V": self.v,
            "Target A": self.a,
            "Target D": self.d,
            "Flower Scale": self.flower,
            "Tree": self.tree,
            "Decay Amount": self.decay,
            "Min Speed": self.min_speed,
            "Max Speed": self.max_speed,
            "Pointer Active": self.pointer_active,
            "Pointer X": self.pointer_x,
            "Pointer Y": self.pointer_y,
            "Pointer Speed": self.pointer_speed,
            "Pointer Strength": self.pointer_strength,
        }


def compute_unreal_payload(
    v: float,
    a: float,
    d: float,
    pointer_active: float = 0.0,
    pointer_x: float = 0.5,
    pointer_y: float = 0.5,
    pointer_speed: float = 0.0,
    pointer_strength: float = 0.0,
) -> UnrealPayload:
    v = clamp(v, -1.0, 1.0)
    a = clamp(a, -1.0, 1.0)
    d = clamp(d, -1.0, 1.0)
    pointer_active = clamp(pointer_active, 0.0, 1.0)
    pointer_x = clamp(pointer_x, 0.0, 1.0)
    pointer_y = clamp(pointer_y, 0.0, 1.0)
    pointer_speed = clamp(pointer_speed, 0.0, 1.0)
    pointer_strength = clamp(pointer_strength, 0.0, 1.0)

    decay = clamp(0.5 - 0.3 * v + 0.3 * a, 0.0, 1.2)
    density = clamp(remap(0.5 * v + 0.5 * d, -1.0, 1.0, 0.4, 1.0), 0.4, 1.0)
    flower = clamp(remap(density, 0.4, 1.0, 0.0, 1.5), 0.0, 1.5)
    tree = clamp(remap(density, 0.4, 1.0, 0.0, 1.0), 0.0, 1.0)
    min_speed = clamp(remap(a, -1.0, 1.0, 0.0, 580.0), 0.0, 600.0)
    max_speed = clamp(min_speed + 20.0, 0.0, 600.0)

    return UnrealPayload(
        v=v,
        a=a,
        d=d,
        decay=decay,
        density=density,
        flower=flower,
        tree=tree,
        min_speed=min_speed,
        max_speed=max_speed,
        pointer_active=pointer_active,
        pointer_x=pointer_x,
        pointer_y=pointer_y,
        pointer_speed=pointer_speed,
        pointer_strength=pointer_strength,
    )
