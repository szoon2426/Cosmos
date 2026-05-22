from __future__ import annotations

from dataclasses import dataclass

try:
    from .final_hand_features import FinalHandFeatures
    from .final_mapper import FinalInteractionPayload
except ImportError:
    from final_hand_features import FinalHandFeatures
    from final_mapper import FinalInteractionPayload


@dataclass(slots=True)
class HudHandState:
    visible: bool
    hand_visible: bool
    x: float
    y: float
    z: float
    open_strength: float
    grab_strength: float
    grab_active: bool
    palm_radius: float
    pose_fallback: bool
    fallback_age: float
    handedness_score: float


@dataclass(slots=True)
class HudFrameState:
    timestamp: float
    frame_index: int
    mode: str
    interaction_active: bool
    world_active: bool
    world_id: str | None
    world_number: int | None
    both_visible: bool
    both_open: bool
    both_grab: bool
    pointer_x: float
    pointer_y: float
    target_v: float
    target_a: float
    target_d: float
    open_strength: float
    grab_strength: float
    switch_to_camera: bool
    left: HudHandState
    right: HudHandState
    left_pointer_active: bool
    right_pointer_active: bool
    left_pointer_y: float
    left_pointer_z: float
    right_pointer_y: float
    right_pointer_z: float
    luma_mean: float
    luma_std: float
    preprocess_applied: bool
    hand_count: int
    roi_rescue_count: int
    camera_props: str


def hud_hand_state(features: FinalHandFeatures) -> HudHandState:
    return HudHandState(
        visible=features.visible,
        hand_visible=features.hand_visible,
        x=features.x,
        y=features.y,
        z=features.z,
        open_strength=features.open_strength,
        grab_strength=features.grab_strength,
        grab_active=features.grab_active,
        palm_radius=features.palm_radius,
        pose_fallback=features.pose_fallback,
        fallback_age=features.fallback_age,
        handedness_score=features.handedness_score,
    )


def hud_frame_state(
    *,
    timestamp: float,
    frame_index: int,
    mode: str,
    world_active: bool,
    world_id: str | None,
    world_number: int | None,
    both_visible: bool,
    both_open: bool,
    both_grab: bool,
    left_features: FinalHandFeatures,
    right_features: FinalHandFeatures,
    left_pointer_active: bool,
    right_pointer_active: bool,
    payload: FinalInteractionPayload,
    luma_mean: float = 0.0,
    luma_std: float = 0.0,
    preprocess_applied: bool = False,
    hand_count: int = 0,
    roi_rescue_count: int = 0,
    camera_props: str = "",
) -> HudFrameState:
    return HudFrameState(
        timestamp=timestamp,
        frame_index=frame_index,
        mode=mode,
        interaction_active=payload.interaction_active > 0.5,
        world_active=world_active,
        world_id=world_id,
        world_number=world_number,
        both_visible=both_visible,
        both_open=both_open,
        both_grab=both_grab,
        pointer_x=payload.pointer_x,
        pointer_y=payload.pointer_y,
        target_v=payload.target_v,
        target_a=payload.target_a,
        target_d=payload.target_d,
        open_strength=payload.open_strength,
        grab_strength=payload.grab_strength,
        switch_to_camera=payload.switch_to_camera > 0.5,
        left=hud_hand_state(left_features),
        right=hud_hand_state(right_features),
        left_pointer_active=left_pointer_active,
        right_pointer_active=right_pointer_active,
        left_pointer_y=payload.l_y_location,
        left_pointer_z=payload.l_z_location,
        right_pointer_y=payload.r_y_location,
        right_pointer_z=payload.r_z_location,
        luma_mean=luma_mean,
        luma_std=luma_std,
        preprocess_applied=preprocess_applied,
        hand_count=hand_count,
        roi_rescue_count=roi_rescue_count,
        camera_props=camera_props,
    )
