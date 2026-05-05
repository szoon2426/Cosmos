from __future__ import annotations

from dataclasses import dataclass

if __package__ in (None, ""):
    from final_hand_features import FinalHandFeatures
else:
    from .final_hand_features import FinalHandFeatures


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(slots=True)
class FinalInteractionPayload:
    hand_active: float
    hand_x: float
    hand_y: float
    hand_z: float
    grab_active: float
    grab_strength: float
    open_strength: float
    hand_speed: float
    palm_radius: float
    distortion_strength: float

    def as_preset_properties(self) -> dict[str, float]:
        return {
            "Hand Active": self.hand_active,
            "Hand X": self.hand_x,
            "Hand Y": self.hand_y,
            "Hand Z": self.hand_z,
            "Grab Active": self.grab_active,
            "Grab Strength": self.grab_strength,
            "Open Strength": self.open_strength,
            "Hand Speed": self.hand_speed,
            "Palm Radius": self.palm_radius,
            "Distortion Strength": self.distortion_strength,
        }


def compute_final_payload(features: FinalHandFeatures) -> FinalInteractionPayload:
    if not features.visible:
        return FinalInteractionPayload(
            hand_active=0.0,
            hand_x=0.5,
            hand_y=0.5,
            hand_z=0.0,
            grab_active=0.0,
            grab_strength=0.0,
            open_strength=0.0,
            hand_speed=0.0,
            palm_radius=0.0,
            distortion_strength=0.0,
        )

    distortion = clamp(
        (0.20 + 0.80 * features.grab_strength) * (0.40 + 0.60 * max(features.speed, features.palm_radius)),
        0.0,
        1.0,
    )
    return FinalInteractionPayload(
        hand_active=1.0,
        hand_x=clamp(features.x, 0.0, 1.0),
        hand_y=clamp(features.y, 0.0, 1.0),
        hand_z=clamp(features.z, -1.0, 1.0),
        grab_active=1.0 if features.grab_active else 0.0,
        grab_strength=clamp(features.grab_strength, 0.0, 1.0),
        open_strength=clamp(features.open_strength, 0.0, 1.0),
        hand_speed=clamp(features.speed, 0.0, 1.0),
        palm_radius=clamp(features.palm_radius, 0.0, 1.0),
        distortion_strength=distortion,
    )
