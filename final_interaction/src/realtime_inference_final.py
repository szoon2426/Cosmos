from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    from camera_preprocess import FramePreprocessor, PreprocessConfig
    from final_hand_features import FinalHandFeatures, FinalHandTracker, actual_both_grab, actual_both_open
    from final_hud_qt import FinalHudController
    from final_hud_state import hud_frame_state
    from final_mapper import compute_final_payload
    from final_pd_bridge import FinalPDBridge
    from final_preview_overlay import draw_preview_overlay
    from final_ue_bridge import FinalUEBridge
    from hand_extractor import HandExtractor
    from hand_roi_rescue import RescueCandidate, RoiRescueConfig, hand_count, rescue_hand_results
    from pose_extractor import PoseExtractor
else:
    from .camera_preprocess import FramePreprocessor, PreprocessConfig
    from .final_hand_features import FinalHandFeatures, FinalHandTracker, actual_both_grab, actual_both_open
    from .final_hud_qt import FinalHudController
    from .final_hud_state import hud_frame_state
    from .final_mapper import compute_final_payload
    from .final_pd_bridge import FinalPDBridge
    from .final_preview_overlay import draw_preview_overlay
    from .final_ue_bridge import FinalUEBridge
    from .hand_extractor import HandExtractor
    from .hand_roi_rescue import RescueCandidate, RoiRescueConfig, hand_count, rescue_hand_results
    from .pose_extractor import PoseExtractor


POINTER_BASE_Y = 43.0
POINTER_WORLD_SPACE_Y = 1900.0
POINTER_LEFT_ANCHOR_OFFSET_Y = -27.0
POINTER_RIGHT_ANCHOR_OFFSET_Y = 41.0
POINTER_ANCHOR_Z = 176.0
POINTER_JOYSTICK_RADIUS = 40.0
POINTER_HAND_DELTA_RANGE = 0.18
POINTER_JOYSTICK_ACTIVE_ALPHA = 0.035
POINTER_JOYSTICK_RETURN_ALPHA = 0.28
GRAB_RECOVERY_SECONDS = 5.0
LEFT_SOLO_GRAB_HOLD_SECONDS = 5.0
LEFT_SOLO_GRAB_SWIPE_DELTA_X = 0.22
LEFT_SOLO_GRAB_SWIPE_VELOCITY_X = 1.2
VAD_DEADZONE_XY = 0.045
VAD_DEADZONE_Z = 0.018
VAD_RESPONSE_ALPHA = 0.35
VAD_GAIN_V = 2.25
VAD_GAIN_A = 2.05
VAD_GAIN_D = 3.8
VAD_MEMORY_BLEND_ALPHA = 0.15
VAD_RELEASE_HOLD_SECONDS = 5.0
VAD_RECOVERY_ALPHA = 1.0 / (25.0 * 30.0)
TRACKING_PROFILES = {
    "normal": {
        "hand_detection": 0.5,
        "hand_presence": 0.5,
        "tracking": 0.5,
        "hand_grace": 0.22,
        "lost_timeout": 0.35,
    },
    "lowlight": {
        "hand_detection": 0.35,
        "hand_presence": 0.35,
        "tracking": 0.45,
        "hand_grace": 0.8,
        "lost_timeout": 0.8,
    },
}
CAMERA_BACKENDS = {
    "auto": None,
    "any": cv2.CAP_ANY,
    "dshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
    "msmf": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
    "v4l2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY),
    "avfoundation": getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY),
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def remap_clamped(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    if in_max == in_min:
        return out_min
    t = clamp((value - in_min) / (in_max - in_min), 0.0, 1.0)
    return out_min + (out_max - out_min) * t


def signed_deadzone(value: float, deadzone: float) -> float:
    if abs(value) <= deadzone:
        return 0.0
    if value > 0:
        return value - deadzone
    return value + deadzone


def vad_to_dict(vad: tuple[float, float, float]) -> dict[str, float]:
    return {
        "valence": round(clamp(vad[0], -1.0, 1.0), 4),
        "arousal": round(clamp(vad[1], -1.0, 1.0), 4),
        "dominance": round(clamp(vad[2], -1.0, 1.0), 4),
    }


def dict_to_vad(data: dict[str, object]) -> tuple[float, float, float] | None:
    try:
        return (
            float(data["valence"]),
            float(data["arousal"]),
            float(data["dominance"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def blend_vad(
    base_vad: tuple[float, float, float],
    memory_vad: tuple[float, float, float],
    alpha: float = VAD_MEMORY_BLEND_ALPHA,
) -> tuple[float, float, float]:
    return tuple(
        clamp(lerp(base_vad[idx], memory_vad[idx], alpha), -1.0, 1.0)
        for idx in range(3)
    )


@dataclass(slots=True)
class BaseMemoryState:
    base_v: float
    base_a: float
    base_d: float
    dv: float = 0.0
    da: float = 0.0
    dd: float = 0.0

    def current_base(self) -> tuple[float, float, float]:
        return (
            clamp(self.base_v + self.dv, -1.0, 1.0),
            clamp(self.base_a + self.da, -1.0, 1.0),
            clamp(self.base_d + self.dd, -1.0, 1.0),
        )

    def set_base(self, base_v: float, base_a: float, base_d: float) -> None:
        self.base_v = clamp(base_v, -1.0, 1.0)
        self.base_a = clamp(base_a, -1.0, 1.0)
        self.base_d = clamp(base_d, -1.0, 1.0)
        self.dv = 0.0
        self.da = 0.0
        self.dd = 0.0

    def learn_from(self, observed_vad: tuple[float, float, float], alpha: float = 0.025) -> None:
        base_v, base_a, base_d = self.base_v, self.base_a, self.base_d
        self.dv = clamp(lerp(self.dv, observed_vad[0] - base_v, alpha), -0.30, 0.30)
        self.da = clamp(lerp(self.da, observed_vad[1] - base_a, alpha), -0.30, 0.30)
        self.dd = clamp(lerp(self.dd, observed_vad[2] - base_d, alpha), -0.30, 0.30)


@dataclass(slots=True)
class InteractionSession:
    active: bool = False
    mode: str = "idle"
    engaged_at: float | None = None
    open_anchor_x: float = 0.5
    open_anchor_y: float = 0.5
    open_anchor_z: float = 0.0
    open_span_x: float = 0.0
    open_span_y: float = 0.0
    anchor_v: float = 0.0
    anchor_a: float = 0.0
    anchor_d: float = 0.0
    grab_anchor_x: float = 0.5
    grab_anchor_y: float = 0.5
    grab_anchor_z: float = 0.0
    grab_span_x: float = 0.0
    grab_span_y: float = 0.0
    grab_anchor_v: float = 0.0
    grab_anchor_a: float = 0.0
    grab_anchor_d: float = 0.0
    left_pointer_anchor_x: float = 0.5
    left_pointer_anchor_y: float = 0.5
    right_pointer_anchor_x: float = 0.5
    right_pointer_anchor_y: float = 0.5
    left_pointer_locked: bool = False
    right_pointer_locked: bool = False
    grab_locked: bool = False
    lost_started_at: float | None = None
    released_at: float | None = None
    grab_recover_until: float | None = None
    thrust_ready: bool = False
    thrust_fired_at: float | None = None
    retreat_started_at: float | None = None
    last_z: float = 0.0
    left_solo_grab_started_at: float | None = None
    left_solo_grab_anchor_x: float = 0.5
    left_solo_grab_last_x: float = 0.5
    left_solo_grab_last_at: float | None = None
    left_solo_grab_delta_x: float = 0.0
    left_solo_grab_velocity_x: float = 0.0
    left_solo_grab_fired: bool = False


@dataclass(slots=True)
class PointerRuntimeState:
    world_number: int | None = None
    left_y: float = POINTER_BASE_Y + POINTER_LEFT_ANCHOR_OFFSET_Y
    left_z: float = POINTER_ANCHOR_Z
    right_y: float = POINTER_BASE_Y + POINTER_RIGHT_ANCHOR_OFFSET_Y
    right_z: float = POINTER_ANCHOR_Z

    def reset(self, world_number: int | None) -> None:
        self.world_number = world_number
        self.left_y, self.left_z = pointer_anchor_y(world_number, "left"), POINTER_ANCHOR_Z
        self.right_y, self.right_z = pointer_anchor_y(world_number, "right"), POINTER_ANCHOR_Z

    def update(
        self,
        *,
        world_number: int | None,
        left_target: tuple[float, float],
        right_target: tuple[float, float],
        left_active: bool,
        right_active: bool,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        if world_number != self.world_number:
            self.world_number = world_number
            self.left_y, self.left_z = pointer_anchor_y(world_number, "left"), POINTER_ANCHOR_Z
            self.right_y, self.right_z = pointer_anchor_y(world_number, "right"), POINTER_ANCHOR_Z

        left_alpha = POINTER_JOYSTICK_ACTIVE_ALPHA if left_active else POINTER_JOYSTICK_RETURN_ALPHA
        right_alpha = POINTER_JOYSTICK_ACTIVE_ALPHA if right_active else POINTER_JOYSTICK_RETURN_ALPHA
        self.left_y = lerp(self.left_y, left_target[0], left_alpha)
        self.left_z = lerp(self.left_z, left_target[1], left_alpha)
        self.right_y = lerp(self.right_y, right_target[0], right_alpha)
        self.right_z = lerp(self.right_z, right_target[1], right_alpha)
        return (self.left_y, self.left_z), (self.right_y, self.right_z)


@dataclass(slots=True)
class VADFootprintStore:
    footprint_dir: Path
    active_world_id: str | None = None
    base_vad: tuple[float, float, float] | None = None
    current_vad: tuple[float, float, float] | None = None
    runtime_base_vad: tuple[float, float, float] | None = None

    def load_world(self, world_id: str, base_vad: tuple[float, float, float]) -> tuple[float, float, float]:
        self.footprint_dir.mkdir(parents=True, exist_ok=True)
        self.active_world_id = world_id
        self.base_vad = base_vad

        path = self._path_for(world_id)
        footprints: list[object] = []
        memory_vad = base_vad
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            loaded_footprints = data.get("vad_footprints")
            if isinstance(loaded_footprints, list):
                footprints = loaded_footprints
            current = data.get("current_vad")
            if isinstance(current, dict):
                loaded = dict_to_vad(current)
                if loaded is not None:
                    memory_vad = loaded

        runtime_base_vad = blend_vad(base_vad, memory_vad)
        self.current_vad = memory_vad
        self.runtime_base_vad = runtime_base_vad
        if not path.exists():
            self._write(world_id, base_vad, base_vad, [])
        elif memory_vad == base_vad:
            self._write(world_id, base_vad, base_vad, footprints)
        return runtime_base_vad

    def clear_active_world(self) -> None:
        self.active_world_id = None
        self.base_vad = None
        self.current_vad = None
        self.runtime_base_vad = None

    def record_interaction(
        self,
        vad: tuple[float, float, float],
        reason: str,
    ) -> bool:
        if self.active_world_id is None or self.base_vad is None:
            return False
        self.commit(tuple(clamp(value, -1.0, 1.0) for value in vad), reason)
        return True

    def commit(self, vad: tuple[float, float, float], reason: str) -> None:
        if self.active_world_id is None or self.base_vad is None:
            return

        world_id = self.active_world_id
        path = self._path_for(world_id)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
        else:
            data = {}

        footprints = data.get("vad_footprints")
        if not isinstance(footprints, list):
            footprints = []

        entry = vad_to_dict(vad)
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        entry["reason"] = reason
        footprints.append(entry)

        self.current_vad = vad
        self._write(world_id, self.base_vad, vad, footprints)
        print(
            "[final_interaction] vad footprint -> "
            f"{world_id} ({vad[0]:.3f}, {vad[1]:.3f}, {vad[2]:.3f}) reason={reason}"
        )

    def _path_for(self, world_id: str) -> Path:
        return self.footprint_dir / f"{world_id}.json"

    def _write(
        self,
        world_id: str,
        base_vad: tuple[float, float, float],
        current_vad: tuple[float, float, float],
        footprints: list[object],
    ) -> None:
        data = {
            "world_id": world_id,
            "base_vad": vad_to_dict(base_vad),
            "current_vad": vad_to_dict(current_vad),
            "vad_footprints": footprints,
        }
        self._path_for(world_id).write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


@dataclass(slots=True)
class WorldVADState:
    world_json_dir: Path
    footprint_store: VADFootprintStore
    poll_interval_sec: float = 0.1
    current_world_id: str | None = None
    current_world_number: int | None = None
    next_poll_at: float = 0.0
    poll_future: Future[str | None] | None = None
    poll_executor: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(max_workers=1))

    def poll_unreal(self, ue: FinalUEBridge, memory: BaseMemoryState, now: float) -> bool:
        changed = False

        if self.poll_future is not None and self.poll_future.done():
            try:
                world_id = self.poll_future.result()
            except Exception:
                world_id = None
            self.poll_future = None
            changed = self._apply_world_id(world_id, memory)

        if self.poll_future is not None or now < self.next_poll_at:
            return changed

        self.next_poll_at = now + self.poll_interval_sec
        self.poll_future = self.poll_executor.submit(ue.get_current_world_id)
        return changed

    def close(self) -> None:
        self.poll_executor.shutdown(wait=False)

    def _apply_world_id(self, world_id: str | None, memory: BaseMemoryState) -> bool:
        if world_id == self.current_world_id:
            return False

        if is_space_world_id(world_id):
            self.footprint_store.clear_active_world()
            self.current_world_id = world_id
            self.current_world_number = None
            print(f"[final_interaction] current space -> {world_id}")
            return True
        if not world_id:
            return False

        world_data = self._load_world_data(world_id)
        if world_data is None:
            print(f"[final_interaction] world json not found or invalid -> {world_id}")
            return False

        vad, world_number = world_data
        current_vad = self.footprint_store.load_world(world_id, vad)
        memory.set_base(*current_vad)
        self.current_world_id = world_id
        self.current_world_number = world_number
        print(
            "[final_interaction] current world -> "
            f"{world_id} current_vad=({current_vad[0]:.3f}, {current_vad[1]:.3f}, {current_vad[2]:.3f})"
        )
        return True

    def is_world_active(self) -> bool:
        return self.current_world_id is not None and self.current_world_number is not None

    def _load_world_data(self, world_id: str) -> tuple[tuple[float, float, float], int] | None:
        path = self.world_json_dir / f"{world_id}.json"
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            return None

        base_vad = data.get("base_vad")
        if not isinstance(base_vad, dict):
            return None

        try:
            vad = (
                float(base_vad["valence"]),
                float(base_vad["arousal"]),
                float(base_vad["dominance"]),
            )
            world_number = int(data["world_number"])
            return vad, world_number
        except (KeyError, TypeError, ValueError):
            return None



def midpoint(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def is_space_world_id(world_id: str | None) -> bool:
    if not world_id:
        return False
    return world_id.strip().lower() in {"galaxy", "space", "none", "null"}


def pd_mode_from_world_id(world_id: str | None) -> str | None:
    if not world_id:
        return None
    return "space" if is_space_world_id(world_id) else "world"


def pointer_y_from_world_number(world_number: int | None) -> float:
    if world_number is None:
        return POINTER_BASE_Y
    return POINTER_BASE_Y + world_number * POINTER_WORLD_SPACE_Y


def pointer_anchor_y(world_number: int | None, side: str) -> float:
    center_y = pointer_y_from_world_number(world_number)
    if side == "left":
        return center_y + POINTER_LEFT_ANCHOR_OFFSET_Y
    return center_y + POINTER_RIGHT_ANCHOR_OFFSET_Y


def pointer_offset_from_hand_delta(delta: float, output_range: float) -> float:
    scaled = (delta / POINTER_HAND_DELTA_RANGE) * output_range
    return clamp(scaled, -output_range, output_range)


def clamp_offset_to_circle(offset_y: float, offset_z: float, radius: float) -> tuple[float, float]:
    length = math.hypot(offset_y, offset_z)
    if length <= radius or length == 0.0:
        return offset_y, offset_z
    scale = radius / length
    return offset_y * scale, offset_z * scale


def joystick_pointer_location(
    *,
    world_number: int | None,
    side: str,
    hand_x: float,
    hand_y: float,
    anchor_x: float,
    anchor_y: float,
    active: bool,
) -> tuple[float, float]:
    anchor_world_y = pointer_anchor_y(world_number, side)
    if not active:
        return anchor_world_y, POINTER_ANCHOR_Z

    offset_y = pointer_offset_from_hand_delta(hand_x - anchor_x, POINTER_JOYSTICK_RADIUS)
    offset_z = pointer_offset_from_hand_delta(anchor_y - hand_y, POINTER_JOYSTICK_RADIUS)
    offset_y, offset_z = clamp_offset_to_circle(offset_y, offset_z, POINTER_JOYSTICK_RADIUS)
    return anchor_world_y + offset_y, POINTER_ANCHOR_Z + offset_z


def reset_left_solo_galaxy_gesture(session: InteractionSession) -> None:
    session.left_solo_grab_started_at = None
    session.left_solo_grab_anchor_x = 0.5
    session.left_solo_grab_last_x = 0.5
    session.left_solo_grab_last_at = None
    session.left_solo_grab_delta_x = 0.0
    session.left_solo_grab_velocity_x = 0.0
    session.left_solo_grab_fired = False


def update_left_solo_galaxy_gesture(
    session: InteractionSession,
    *,
    pointer_world_active: bool,
    left_features: FinalHandFeatures,
    right_features: FinalHandFeatures,
    world_vad: tuple[float, float, float],
    world_base_vad: tuple[float, float, float],
    now: float,
) -> tuple[tuple[float, float, float], bool]:
    def left_grab_or_arm_fallback() -> bool:
        real_left_grab = left_features.hand_visible and left_features.grab_active
        arm_fallback_continuation = (
            session.left_solo_grab_started_at is not None
            and left_features.visible
            and left_features.pose_fallback
        )
        return real_left_grab or arm_fallback_continuation

    right_actual_grab = right_features.hand_visible and right_features.grab_active
    left_only_grab = (
        pointer_world_active
        and left_grab_or_arm_fallback()
        and not right_actual_grab
    )
    if not left_only_grab:
        reset_left_solo_galaxy_gesture(session)
        return world_vad, False

    if session.left_solo_grab_started_at is None:
        session.left_solo_grab_started_at = now
        session.left_solo_grab_anchor_x = left_features.x
        session.left_solo_grab_last_x = left_features.x
        session.left_solo_grab_last_at = now
        session.left_solo_grab_delta_x = 0.0
        session.left_solo_grab_velocity_x = 0.0
        session.left_solo_grab_fired = False
        return world_vad, False

    previous_x = session.left_solo_grab_last_x
    previous_at = session.left_solo_grab_last_at if session.left_solo_grab_last_at is not None else now
    x_velocity = (left_features.x - previous_x) / max(now - previous_at, 1e-3)
    delta_from_anchor = left_features.x - session.left_solo_grab_anchor_x
    session.left_solo_grab_delta_x = delta_from_anchor
    session.left_solo_grab_velocity_x = x_velocity
    held = now - session.left_solo_grab_started_at >= LEFT_SOLO_GRAB_HOLD_SECONDS

    switch_to_camera = (
        held
        and not session.left_solo_grab_fired
        and delta_from_anchor >= LEFT_SOLO_GRAB_SWIPE_DELTA_X
        and x_velocity >= LEFT_SOLO_GRAB_SWIPE_VELOCITY_X
    )
    if switch_to_camera:
        session.left_solo_grab_fired = True

    session.left_solo_grab_last_x = left_features.x
    session.left_solo_grab_last_at = now

    if held:
        return world_base_vad, switch_to_camera
    return world_vad, False


def left_solo_galaxy_debug_state(session: InteractionSession, now: float) -> dict[str, float | bool]:
    active = session.left_solo_grab_started_at is not None
    elapsed = 0.0
    if session.left_solo_grab_started_at is not None:
        elapsed = max(0.0, now - session.left_solo_grab_started_at)
    hold_progress = clamp(elapsed / LEFT_SOLO_GRAB_HOLD_SECONDS, 0.0, 1.0)
    restore_active = active and hold_progress >= 1.0
    return {
        "left_solo_grab_active": active,
        "left_solo_grab_elapsed": elapsed,
        "left_solo_grab_hold_progress": hold_progress,
        "left_solo_vad_restore_active": restore_active,
        "left_solo_swipe_delta_x": session.left_solo_grab_delta_x if active else 0.0,
        "left_solo_swipe_velocity_x": session.left_solo_grab_velocity_x if active else 0.0,
        "left_solo_world_move_ready": restore_active and not session.left_solo_grab_fired,
        "left_solo_world_move_fired": session.left_solo_grab_fired,
    }


def reset_interaction_session(session: InteractionSession) -> None:
    session.active = False
    session.mode = "idle"
    session.engaged_at = None
    session.grab_locked = False
    session.lost_started_at = None
    session.thrust_ready = False
    session.retreat_started_at = None
    session.released_at = None
    session.grab_recover_until = None
    session.left_pointer_locked = False
    session.right_pointer_locked = False
    reset_left_solo_galaxy_gesture(session)


def end_interaction(
    session: InteractionSession,
    footprint_store: VADFootprintStore,
    world_vad: tuple[float, float, float],
    now: float,
    *,
    allow_grab_recovery: bool = False,
) -> None:
    should_open_recovery = allow_grab_recovery and session.active and session.mode == "open"
    session.active = False
    session.mode = "idle"
    session.engaged_at = None
    session.grab_locked = False
    session.lost_started_at = None
    session.thrust_ready = False
    session.retreat_started_at = None
    session.released_at = now
    session.grab_recover_until = now + GRAB_RECOVERY_SECONDS if should_open_recovery else None
    footprint_store.record_interaction(world_vad, "interaction_settled")


def recover_world_vad_after_release(
    session: InteractionSession,
    memory: BaseMemoryState,
    world_vad: tuple[float, float, float],
    now: float,
) -> tuple[float, float, float]:
    if session.active or session.released_at is None:
        return world_vad

    if now - session.released_at < VAD_RELEASE_HOLD_SECONDS:
        return world_vad

    base_vad = memory.current_base()
    return tuple(
        lerp(world_vad[idx], base_vad[idx], VAD_RECOVERY_ALPHA)
        for idx in range(3)
    )


def activate_grab_session(
    session: InteractionSession,
    *,
    pointer_xy: tuple[float, float],
    avg_z: float,
    span_x: float,
    span_y: float,
    world_vad: tuple[float, float, float],
    now: float,
) -> None:
    session.active = True
    session.mode = "grab"
    session.engaged_at = None
    session.grab_locked = True
    session.grab_anchor_x = pointer_xy[0]
    session.grab_anchor_y = pointer_xy[1]
    session.grab_anchor_z = avg_z
    session.grab_span_x = span_x
    session.grab_span_y = span_y
    session.grab_anchor_v, session.grab_anchor_a, session.grab_anchor_d = world_vad
    session.released_at = None
    session.grab_recover_until = None
    session.retreat_started_at = None
    session.lost_started_at = None
    session.thrust_ready = False


def can_recover_grab(session: InteractionSession, now: float) -> bool:
    return session.grab_recover_until is not None and now <= session.grab_recover_until


def open_video_capture(camera_index: int, backend_name: str):
    backend = CAMERA_BACKENDS.get(backend_name)
    if backend is None:
        return cv2.VideoCapture(camera_index)
    return cv2.VideoCapture(camera_index, backend)


def set_camera_prop(cap, prop_id: int, label: str, value: float | None) -> None:
    if value is None:
        return
    ok = cap.set(prop_id, value)
    actual = cap.get(prop_id)
    if not ok:
        print(f"[final_interaction] camera prop unsupported or rejected -> {label} requested={value} actual={actual}")


def camera_props_label(cap) -> str:
    props = [
        ("w", cv2.CAP_PROP_FRAME_WIDTH),
        ("h", cv2.CAP_PROP_FRAME_HEIGHT),
        ("fps", cv2.CAP_PROP_FPS),
        ("exp", cv2.CAP_PROP_EXPOSURE),
        ("gain", cv2.CAP_PROP_GAIN),
        ("bri", cv2.CAP_PROP_BRIGHTNESS),
        ("con", cv2.CAP_PROP_CONTRAST),
        ("gam", cv2.CAP_PROP_GAMMA),
        ("focus", cv2.CAP_PROP_FOCUS),
    ]
    parts = []
    for label, prop_id in props:
        value = cap.get(prop_id)
        if value == -1:
            parts.append(f"{label}=-")
        elif label in {"w", "h", "fps"}:
            parts.append(f"{label}={value:.0f}")
        else:
            parts.append(f"{label}={value:.2f}")
    return "cam " + " ".join(parts)


def build_rescue_candidates(
    right_tracker: FinalHandTracker,
    left_tracker: FinalHandTracker,
    pose_landmarks: dict[str, tuple[float, float]] | None,
) -> list[RescueCandidate]:
    candidates: list[RescueCandidate] = []
    for tracker in (right_tracker, left_tracker):
        center = tracker.rescue_center(pose_landmarks)
        if center is None:
            continue
        candidates.append(
            RescueCandidate(
                expected_side=tracker.expected_mp_side,
                center=(clamp(center[0], 0.0, 1.0), clamp(center[1], 0.0, 1.0)),
                source=tracker.control_hand,
            )
        )
    return candidates


class TrackingLogWriter:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.last_write_at = 0.0
        self.file = None
        self.csv_writer = None
        self.csv_fields = [
            "timestamp",
            "frame_index",
            "mode",
            "interaction_active",
            "luma_mean",
            "luma_std",
            "preprocess_applied",
            "hand_count",
            "roi_rescue_count",
            "both_visible",
            "both_open",
            "both_grab",
            "left_visible",
            "left_hand_visible",
            "left_open",
            "left_grab",
            "left_palm_radius",
            "left_fallback_age",
            "left_handedness_score",
            "right_visible",
            "right_hand_visible",
            "right_open",
            "right_grab",
            "right_palm_radius",
            "right_fallback_age",
            "right_handedness_score",
        ]
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w", encoding="utf-8", newline="")
        if path.suffix.lower() == ".csv":
            self.csv_writer = csv.DictWriter(self.file, fieldnames=self.csv_fields)
            self.csv_writer.writeheader()

    def write(self, state) -> None:
        if self.file is None:
            return
        if state.timestamp - self.last_write_at < 0.1:
            return
        self.last_write_at = state.timestamp
        row = {
            "timestamp": f"{state.timestamp:.3f}",
            "frame_index": state.frame_index,
            "mode": state.mode,
            "interaction_active": int(state.interaction_active),
            "luma_mean": f"{state.luma_mean:.3f}",
            "luma_std": f"{state.luma_std:.3f}",
            "preprocess_applied": int(state.preprocess_applied),
            "hand_count": state.hand_count,
            "roi_rescue_count": state.roi_rescue_count,
            "both_visible": int(state.both_visible),
            "both_open": int(state.both_open),
            "both_grab": int(state.both_grab),
            "left_visible": int(state.left.visible),
            "left_hand_visible": int(state.left.hand_visible),
            "left_open": f"{state.left.open_strength:.3f}",
            "left_grab": f"{state.left.grab_strength:.3f}",
            "left_palm_radius": f"{state.left.palm_radius:.3f}",
            "left_fallback_age": f"{state.left.fallback_age:.3f}",
            "left_handedness_score": f"{state.left.handedness_score:.3f}",
            "right_visible": int(state.right.visible),
            "right_hand_visible": int(state.right.hand_visible),
            "right_open": f"{state.right.open_strength:.3f}",
            "right_grab": f"{state.right.grab_strength:.3f}",
            "right_palm_radius": f"{state.right.palm_radius:.3f}",
            "right_fallback_age": f"{state.right.fallback_age:.3f}",
            "right_handedness_score": f"{state.right.handedness_score:.3f}",
        }
        if self.csv_writer is not None:
            self.csv_writer.writerow(row)
        else:
            self.file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.file.flush()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Cosmos final grab/open interaction runtime")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--camera-backend",
        choices=tuple(CAMERA_BACKENDS.keys()),
        default="auto",
        help="OpenCV camera backend. On Windows, dshow or msmf can expose different camera controls.",
    )
    parser.add_argument("--send", action="store_true", help="Enable sending data to Unreal")
    parser.add_argument("--send-pd", action="store_true", help="Enable sending data to Pure Data")
    parser.add_argument(
        "--control-hand",
        choices=("left", "right"),
        default="right",
        help="Visitor-facing control hand. Because the frame is mirrored, MediaPipe handedness is compensated internally.",
    )
    parser.add_argument("--base-v", type=float, default=0.0)
    parser.add_argument("--base-a", type=float, default=0.0)
    parser.add_argument("--base-d", type=float, default=0.0)
    parser.add_argument(
        "--world-json-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "world_spawn_json",
        help="Directory containing world_XXXX.json files with base_vad.",
    )
    parser.add_argument(
        "--world-poll-sec",
        type=float,
        default=0.1,
        help="How often to read the current world id from Unreal.",
    )
    parser.add_argument(
        "--vad-footprint-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "vad_footprint",
        help="Directory where per-world VAD footprint JSON files are stored.",
    )
    parser.add_argument("--camera-width", type=int, default=1280, help="Requested webcam capture width.")
    parser.add_argument("--camera-height", type=int, default=720, help="Requested webcam capture height.")
    parser.add_argument("--camera-fps", type=int, default=30, help="Requested webcam FPS.")
    parser.add_argument("--camera-exposure", type=float, default=None, help="Optional camera exposure value.")
    parser.add_argument("--camera-gain", type=float, default=None, help="Optional camera gain value.")
    parser.add_argument("--camera-brightness", type=float, default=None, help="Optional camera brightness value.")
    parser.add_argument("--camera-contrast", type=float, default=None, help="Optional camera contrast value.")
    parser.add_argument("--camera-gamma", type=float, default=None, help="Optional camera gamma value.")
    parser.add_argument("--camera-focus", type=float, default=None, help="Optional camera focus value.")
    parser.add_argument(
        "--preprocess",
        choices=("off", "auto", "lowlight"),
        default="auto",
        help="Apply inference-only low-light enhancement. Preview remains the original camera frame.",
    )
    parser.add_argument("--preprocess-alpha", type=float, default=1.12, help="Low-light contrast gain.")
    parser.add_argument("--preprocess-beta", type=int, default=8, help="Low-light brightness bias.")
    parser.add_argument("--preprocess-clahe-clip", type=float, default=2.0, help="CLAHE clip limit.")
    parser.add_argument("--preprocess-clahe-grid", type=int, default=8, help="CLAHE tile grid size.")
    parser.add_argument(
        "--tracking-profile",
        choices=tuple(TRACKING_PROFILES.keys()),
        default="normal",
        help="normal preserves current sensitivity; lowlight lowers thresholds and extends fallback timing.",
    )
    parser.add_argument(
        "--roi-rescue",
        choices=("off", "auto", "on"),
        default="auto",
        help="Run pose/last-position crop re-detection when full-frame hand detection is incomplete.",
    )
    parser.add_argument("--roi-rescue-size", type=float, default=0.28, help="Square crop size as a fraction of the shorter frame edge.")
    parser.add_argument("--roi-rescue-scale", type=float, default=2.5, help="Upscale factor for ROI rescue crops.")
    parser.add_argument("--tracking-log", type=Path, default=None, help="Optional .jsonl or .csv tracking diagnostics log path.")
    parser.add_argument(
        "--pose-every",
        type=int,
        default=3,
        help="Run pose detection every N frames. Use 1 for maximum accuracy, 0 to disable pose fallback.",
    )
    parser.add_argument(
        "--debug-overlay",
        action="store_true",
        help="Draw detailed debug labels on the camera preview. Slower, so keep off for exhibition.",
    )
    parser.add_argument("--preview-overlay", dest="preview_overlay", action="store_true", default=True)
    parser.add_argument("--no-preview-overlay", dest="preview_overlay", action="store_false")
    parser.add_argument("--hud", action="store_true", help="Show a transparent visual HUD window.")
    parser.add_argument("--hud-monitor", type=int, default=0, help="Monitor index for the HUD window.")
    parser.add_argument("--hud-opacity", type=float, default=1.0, help="Opacity scale for HUD visual elements.")
    parser.add_argument("--hud-scale", type=float, default=1.0, help="Scale factor for HUD visual elements.")
    parser.add_argument("--hud-click-through", dest="hud_click_through", action="store_true", default=True)
    parser.add_argument("--no-hud-click-through", dest="hud_click_through", action="store_false")
    parser.add_argument("--hud-topmost", dest="hud_topmost", action="store_true", default=True)
    parser.add_argument("--no-hud-topmost", dest="hud_topmost", action="store_false")
    parser.add_argument(
        "--perf-log-sec",
        type=float,
        default=0.0,
        help="Print loop FPS and stage timings every N seconds when greater than 0.",
    )
    args = parser.parse_args()

    cap = open_video_capture(args.camera, args.camera_backend)
    if not cap.isOpened():
        print(f"[ERROR] camera {args.camera} could not be opened")
        sys.exit(1)
    if args.camera_width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    if args.camera_height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    if args.camera_fps > 0:
        cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    set_camera_prop(cap, cv2.CAP_PROP_EXPOSURE, "exposure", args.camera_exposure)
    set_camera_prop(cap, cv2.CAP_PROP_GAIN, "gain", args.camera_gain)
    set_camera_prop(cap, cv2.CAP_PROP_BRIGHTNESS, "brightness", args.camera_brightness)
    set_camera_prop(cap, cv2.CAP_PROP_CONTRAST, "contrast", args.camera_contrast)
    set_camera_prop(cap, cv2.CAP_PROP_GAMMA, "gamma", args.camera_gamma)
    set_camera_prop(cap, cv2.CAP_PROP_FOCUS, "focus", args.camera_focus)
    camera_props = camera_props_label(cap)
    print(f"[final_interaction] camera backend={args.camera_backend} {camera_props}")

    profile = TRACKING_PROFILES[args.tracking_profile]
    roi_enabled = args.roi_rescue == "on" or (args.roi_rescue == "auto" and args.tracking_profile == "lowlight")
    preprocessor = FramePreprocessor(
        PreprocessConfig(
            mode=args.preprocess,
            alpha=args.preprocess_alpha,
            beta=args.preprocess_beta,
            clahe_clip_limit=args.preprocess_clahe_clip,
            clahe_tile_grid=args.preprocess_clahe_grid,
        )
    )
    roi_config = RoiRescueConfig(
        enabled=roi_enabled,
        size_fraction=args.roi_rescue_size,
        scale=args.roi_rescue_scale,
    )
    hand_extractor = HandExtractor(
        min_hand_detection_confidence=profile["hand_detection"],
        min_hand_presence_confidence=profile["hand_presence"],
        min_tracking_confidence=profile["tracking"],
        enable_image_mode=roi_enabled,
    )
    pose_extractor = PoseExtractor()
    right_tracker = FinalHandTracker(control_hand="RIGHT", grace_seconds=profile["hand_grace"])
    left_tracker = FinalHandTracker(control_hand="LEFT", grace_seconds=profile["hand_grace"])
    lost_timeout = profile["lost_timeout"]
    tracking_log = TrackingLogWriter(args.tracking_log)
    ue = FinalUEBridge(enabled=args.send)
    pd = FinalPDBridge(enabled=args.send_pd)
    ue.start()
    pd.start()
    memory = BaseMemoryState(base_v=args.base_v, base_a=args.base_a, base_d=args.base_d)
    footprint_store = VADFootprintStore(footprint_dir=args.vad_footprint_dir)
    world_vad_state = WorldVADState(
        world_json_dir=args.world_json_dir,
        footprint_store=footprint_store,
        poll_interval_sec=max(0.1, args.world_poll_sec),
    )
    session = InteractionSession()
    pointer_state = PointerRuntimeState()
    hud = FinalHudController(
        enabled=args.hud,
        monitor_index=args.hud_monitor,
        click_through=args.hud_click_through,
        topmost=args.hud_topmost,
        opacity=args.hud_opacity,
        scale=args.hud_scale,
    )
    hud.start()
    world_vad = memory.current_base()
    pd_mode: str | None = None
    frame_index = 0
    last_pose_landmarks: dict[str, tuple[float, float]] | None = None
    perf_last_at = time.perf_counter()
    perf_frames = 0
    perf_acc = {"capture": 0.0, "pose": 0.0, "hand": 0.0, "logic": 0.0, "draw": 0.0, "hud": 0.0}

    if args.send_pd:
        pd.send_value("READY_MODE", 0.0)
        initial_payload = compute_final_payload(
            interaction_active=False,
            pointer_x=-1.0,
            pointer_y=-1.0,
            target_v=world_vad[0],
            target_a=world_vad[1],
            target_d=world_vad[2],
            grab_active=False,
            open_strength=0.0,
            grab_strength=0.0,
            switch_to_camera=False,
        )
        pd.send_payload(initial_payload)

    try:
        while True:
            loop_started = time.perf_counter()
            ok, frame = cap.read()
            after_capture = time.perf_counter()
            if not ok:
                time.sleep(0.03)
                continue

            now = time.time()
            frame = cv2.flip(frame, 1)
            preprocess_result = preprocessor.process(frame)
            frame_rgb = cv2.cvtColor(preprocess_result.frame_bgr, cv2.COLOR_BGR2RGB)

            pose_started = time.perf_counter()
            if args.pose_every <= 0:
                pose_landmarks = None
            elif frame_index % args.pose_every == 0:
                pose_results = pose_extractor.process(frame_rgb)
                last_pose_landmarks = pose_extractor.extract_upper_body(pose_results)
                pose_landmarks = last_pose_landmarks
            else:
                pose_landmarks = last_pose_landmarks
            after_pose = time.perf_counter()

            hand_results = hand_extractor.process(frame_rgb)
            rescue_stats = None
            if roi_enabled and hand_count(hand_results) < 2:
                hand_results, rescue_stats = rescue_hand_results(
                    image_detector=hand_extractor.detect_image,
                    frame_rgb=frame_rgb,
                    base_results=hand_results,
                    candidates=build_rescue_candidates(right_tracker, left_tracker, pose_landmarks),
                    config=roi_config,
                )
            after_hand = time.perf_counter()
            right_features = right_tracker.update(hand_results, pose_landmarks, now)
            left_features = left_tracker.update(hand_results, pose_landmarks, now)
            switch_to_camera = False
            was_active = session.active
            world_changed = args.send and world_vad_state.poll_unreal(ue, memory, now)
            if world_changed:
                world_vad = memory.current_base()
                reset_interaction_session(session)
                pointer_state.reset(world_vad_state.current_world_number)
            if args.send_pd:
                desired_pd_mode = pd_mode_from_world_id(world_vad_state.current_world_id)
                if desired_pd_mode is not None and desired_pd_mode != pd_mode:
                    if desired_pd_mode == "space":
                        pd.set_space_mode()
                    else:
                        pd.set_world_mode()
                    pd_mode = desired_pd_mode
            base_vad = memory.current_base()
            pointer_world_active = world_vad_state.is_world_active()
            both_visible = right_features.visible and left_features.visible
            both_open = actual_both_open(right_features, left_features)
            both_grab = actual_both_grab(right_features, left_features)
            interaction_both_open = pointer_world_active and both_open
            interaction_both_grab = pointer_world_active and both_grab

            if both_visible:
                pointer_xy = midpoint(
                    (right_features.x, right_features.y),
                    (left_features.x, left_features.y),
                )
                avg_z = 0.5 * (right_features.z + left_features.z)
                avg_speed = 0.5 * (right_features.speed + left_features.speed)
                avg_open = 0.5 * (right_features.open_strength + left_features.open_strength)
                avg_grab = 0.5 * (right_features.grab_strength + left_features.grab_strength)
                span_x = abs(right_features.x - left_features.x)
                span_y = abs(right_features.y - left_features.y)
            else:
                pointer_xy = (-1.0, -1.0)
                avg_z = 0.0
                avg_speed = 0.0
                avg_open = 0.0
                avg_grab = 0.0
                span_x = 0.0
                span_y = 0.0

            if not session.active and session.released_at is None:
                world_vad = tuple(
                    lerp(world_vad[idx], base_vad[idx], 1.0 / (25.0 * 30.0))
                    for idx in range(3)
                )

            if not pointer_world_active:
                if session.active:
                    end_interaction(session, footprint_store, world_vad, now)
                session.engaged_at = None
                session.grab_locked = False
                session.grab_recover_until = None
                session.left_pointer_locked = False
                session.right_pointer_locked = False
                session.thrust_ready = False
                session.retreat_started_at = None
                session.lost_started_at = None
            elif not session.active:
                if can_recover_grab(session, now) and interaction_both_grab:
                    activate_grab_session(
                        session,
                        pointer_xy=pointer_xy,
                        avg_z=avg_z,
                        span_x=span_x,
                        span_y=span_y,
                        world_vad=world_vad,
                        now=now,
                    )
                elif session.grab_recover_until is not None and now > session.grab_recover_until:
                    session.grab_recover_until = None
                    session.engaged_at = None
                elif interaction_both_open:
                    if session.engaged_at is None:
                        session.engaged_at = now
                    elif now - session.engaged_at >= 0.08:
                        session.active = True
                        session.mode = "open"
                        session.open_anchor_x = pointer_xy[0]
                        session.open_anchor_y = pointer_xy[1]
                        session.open_anchor_z = avg_z
                        session.open_span_x = span_x
                        session.open_span_y = span_y
                        session.anchor_v, session.anchor_a, session.anchor_d = world_vad
                        session.released_at = None
                        session.grab_recover_until = None
                        session.retreat_started_at = None
                else:
                    session.engaged_at = None
            else:
                if both_visible:
                    session.lost_started_at = None
                    if interaction_both_grab:
                        if not session.grab_locked:
                            session.grab_locked = True
                            session.mode = "grab"
                            session.grab_anchor_x = pointer_xy[0]
                            session.grab_anchor_y = pointer_xy[1]
                            session.grab_anchor_z = avg_z
                            session.grab_span_x = span_x
                            session.grab_span_y = span_y
                            session.grab_anchor_v, session.grab_anchor_a, session.grab_anchor_d = world_vad

                        dx = signed_deadzone(span_x - session.grab_span_x, VAD_DEADZONE_XY)
                        dy = signed_deadzone(span_y - session.grab_span_y, VAD_DEADZONE_XY)
                        dz = signed_deadzone(avg_z - session.grab_anchor_z, VAD_DEADZONE_Z)
                        target_v = clamp(session.grab_anchor_v + dx * VAD_GAIN_V, -1.0, 1.0)
                        target_a = clamp(session.grab_anchor_a + dy * VAD_GAIN_A, -1.0, 1.0)
                        target_d = clamp(session.grab_anchor_d + dz * VAD_GAIN_D, -1.0, 1.0)

                        world_vad = (
                            lerp(world_vad[0], target_v, VAD_RESPONSE_ALPHA),
                            lerp(world_vad[1], target_a, VAD_RESPONSE_ALPHA),
                            lerp(world_vad[2], target_d, VAD_RESPONSE_ALPHA),
                        )
                        session.retreat_started_at = None
                    else:
                        session.grab_locked = False
                        session.mode = "open"
                        if interaction_both_open:
                            # Open state holds the current world state without forcing it back yet.
                            pass

                        # If the hand stays clearly pulled back while open, treat that as
                        # ending the interaction and let the pointer leave the screen.
                        if both_open and avg_z <= -0.28:
                            if session.retreat_started_at is None:
                                session.retreat_started_at = now
                            elif now - session.retreat_started_at >= 0.45:
                                end_interaction(session, footprint_store, world_vad, now)
                        else:
                            session.retreat_started_at = None
                else:
                    if session.lost_started_at is None:
                        session.lost_started_at = now
                    elif now - session.lost_started_at >= lost_timeout:
                        end_interaction(session, footprint_store, world_vad, now, allow_grab_recovery=True)

            if not session.active and session.released_at is not None:
                world_vad = recover_world_vad_after_release(session, memory, world_vad, now)

            raw_world_base_vad = footprint_store.base_vad if footprint_store.base_vad is not None else memory.current_base()
            world_vad, left_solo_switch_to_camera = update_left_solo_galaxy_gesture(
                session,
                pointer_world_active=pointer_world_active,
                left_features=left_features,
                right_features=right_features,
                world_vad=world_vad,
                world_base_vad=raw_world_base_vad,
                now=now,
            )
            switch_to_camera = switch_to_camera or left_solo_switch_to_camera

            if switch_to_camera:
                footprint_store.record_interaction(world_vad, "switch_to_galaxy")
                if args.send_pd and pd_mode != "space":
                    pd.set_space_mode()
                    pd_mode = "space"

            left_solo_debug = left_solo_galaxy_debug_state(session, now)
            session.last_z = avg_z

            pointer_x = pointer_xy[0] if session.active and both_visible else -1.0
            pointer_y = pointer_xy[1] if session.active and both_visible else -1.0
            left_pointer_active = pointer_world_active and left_features.hand_visible and left_features.grab_active
            right_pointer_active = pointer_world_active and right_features.hand_visible and right_features.grab_active

            if left_pointer_active and not session.left_pointer_locked:
                session.left_pointer_locked = True
                session.left_pointer_anchor_x = left_features.x
                session.left_pointer_anchor_y = left_features.y
            elif not left_pointer_active:
                session.left_pointer_locked = False

            if right_pointer_active and not session.right_pointer_locked:
                session.right_pointer_locked = True
                session.right_pointer_anchor_x = right_features.x
                session.right_pointer_anchor_y = right_features.y
            elif not right_pointer_active:
                session.right_pointer_locked = False

            left_pointer_target = joystick_pointer_location(
                world_number=world_vad_state.current_world_number,
                side="left",
                hand_x=left_features.x,
                hand_y=left_features.y,
                anchor_x=session.left_pointer_anchor_x,
                anchor_y=session.left_pointer_anchor_y,
                active=left_pointer_active,
            )
            right_pointer_target = joystick_pointer_location(
                world_number=world_vad_state.current_world_number,
                side="right",
                hand_x=right_features.x,
                hand_y=right_features.y,
                anchor_x=session.right_pointer_anchor_x,
                anchor_y=session.right_pointer_anchor_y,
                active=right_pointer_active,
            )
            (left_pointer_y, left_pointer_z), (right_pointer_y, right_pointer_z) = pointer_state.update(
                world_number=world_vad_state.current_world_number,
                left_target=left_pointer_target,
                right_target=right_pointer_target,
                left_active=left_pointer_active,
                right_active=right_pointer_active,
            )
            left_pointer_grip = left_features.grab_strength if left_pointer_active else 0.0
            right_pointer_grip = right_features.grab_strength if right_pointer_active else 0.0
            payload = compute_final_payload(
                interaction_active=session.active,
                pointer_x=pointer_x,
                pointer_y=pointer_y,
                target_v=world_vad[0],
                target_a=world_vad[1],
                target_d=world_vad[2],
                grab_active=interaction_both_grab,
                open_strength=avg_open,
                grab_strength=avg_grab,
                switch_to_camera=switch_to_camera,
                l_grip=left_pointer_grip,
                r_grip=right_pointer_grip,
                l_y_location=left_pointer_y,
                l_z_location=left_pointer_z,
                r_y_location=right_pointer_y,
                r_z_location=right_pointer_z,
            )
            hud_state = hud_frame_state(
                timestamp=now,
                frame_index=frame_index,
                mode=session.mode,
                world_active=pointer_world_active,
                world_id=world_vad_state.current_world_id,
                world_number=world_vad_state.current_world_number,
                both_visible=both_visible,
                both_open=both_open,
                both_grab=both_grab,
                left_features=left_features,
                right_features=right_features,
                left_pointer_active=left_pointer_active,
                right_pointer_active=right_pointer_active,
                payload=payload,
                luma_mean=preprocess_result.luma_mean,
                luma_std=preprocess_result.luma_std,
                preprocess_applied=preprocess_result.applied,
                hand_count=hand_count(hand_results),
                roi_rescue_count=rescue_stats.added if rescue_stats is not None else 0,
                camera_props=camera_props,
                **left_solo_debug,
            )
            tracking_log.write(hud_state)

            if args.send:
                ue.send(payload)

            if args.send_pd:
                if not was_active and session.active:
                    pd.trigger_ready_on()
                    pd.send_value("READY_MODE", 1.0)
                elif was_active and not session.active:
                    pd.trigger_ready_off()
                    pd.send_value("READY_MODE", 0.0)

                pd.send_payload(payload)

            after_logic = time.perf_counter()
            if args.preview_overlay or args.debug_overlay:
                draw_preview_overlay(
                    frame,
                    hand_results=hand_results,
                    pose_landmarks=pose_landmarks,
                    state=hud_state,
                    debug=args.debug_overlay,
                )
            after_draw = time.perf_counter()

            cv2.imshow("Cosmos Final 0505", frame)
            hud_started = time.perf_counter()
            hud.update(hud_state)
            hud.process_events()
            after_hud = time.perf_counter()
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            frame_index += 1
            if args.perf_log_sec > 0:
                perf_frames += 1
                perf_acc["capture"] += after_capture - loop_started
                perf_acc["pose"] += after_pose - pose_started
                perf_acc["hand"] += after_hand - after_pose
                perf_acc["logic"] += after_logic - after_hand
                perf_acc["draw"] += after_draw - after_logic
                perf_acc["hud"] += after_hud - hud_started
                perf_now = time.perf_counter()
                elapsed = perf_now - perf_last_at
                if elapsed >= args.perf_log_sec:
                    fps = perf_frames / max(elapsed, 1e-6)
                    print(
                        "[final_interaction] perf "
                        f"fps={fps:.1f} "
                        f"capture={perf_acc['capture'] / perf_frames * 1000:.1f}ms "
                        f"pose={perf_acc['pose'] / perf_frames * 1000:.1f}ms "
                        f"hand={perf_acc['hand'] / perf_frames * 1000:.1f}ms "
                        f"logic={perf_acc['logic'] / perf_frames * 1000:.1f}ms "
                        f"draw={perf_acc['draw'] / perf_frames * 1000:.1f}ms "
                        f"hud={perf_acc['hud'] / perf_frames * 1000:.1f}ms"
                    )
                    perf_last_at = perf_now
                    perf_frames = 0
                    perf_acc = {"capture": 0.0, "pose": 0.0, "hand": 0.0, "logic": 0.0, "draw": 0.0, "hud": 0.0}
    finally:
        hud.close()
        tracking_log.close()
        world_vad_state.close()
        cap.release()
        cv2.destroyAllWindows()
        hand_extractor.close()
        pose_extractor.close()
        world_vad_state.close()
        ue.stop()
        pd.stop()


if __name__ == "__main__":
    main()
