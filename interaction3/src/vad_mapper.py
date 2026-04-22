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
        }

    def as_preset_properties(self) -> dict[str, float]:
        return {
            "Target V": self.v,
            "Target A": self.a,
            "Target D": self.d,
            "Flower": self.flower,
            "Tree": self.tree,
            "Decay Amount": self.decay,
            "Min Speed": self.min_speed,
            "Max Speed": self.max_speed,
        }


def compute_unreal_payload(v: float, a: float, d: float) -> UnrealPayload:
    v = clamp(v, -1.0, 1.0)
    a = clamp(a, -1.0, 1.0)
    d = clamp(d, -1.0, 1.0)

    decay = clamp(0.5 - 0.3 * v + 0.3 * a, 0.0, 1.2)
    density = clamp(remap(0.5 * v + 0.5 * d, -1.0, 1.0, 0.4, 1.0), 0.4, 1.0)
    flower = density
    tree = density
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
    )
