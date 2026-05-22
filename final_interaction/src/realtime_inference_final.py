from __future__ import annotations

import argparse
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
    from final_hand_features import FinalHandTracker
    from final_mapper import compute_final_payload
    from final_pd_bridge import FinalPDBridge
    from final_ue_bridge import FinalUEBridge
    from hand_extractor import HandExtractor
    from pose_extractor import PoseExtractor
else:
    from .final_hand_features import FinalHandTracker
    from .final_mapper import compute_final_payload
    from .final_pd_bridge import FinalPDBridge
    from .final_ue_bridge import FinalUEBridge
    from .hand_extractor import HandExtractor
    from .pose_extractor import PoseExtractor


HAND_LANDMARK_LABELS = [
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_MCP",
    "INDEX_PIP",
    "INDEX_DIP",
    "INDEX_TIP",
    "MIDDLE_MCP",
    "MIDDLE_PIP",
    "MIDDLE_DIP",
    "MIDDLE_TIP",
    "RING_MCP",
    "RING_PIP",
    "RING_DIP",
    "RING_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
]

POINTER_BASE_Y = 43.0
POINTER_WORLD_SPACE_Y = 1900.0
POINTER_LEFT_ANCHOR_OFFSET_Y = -27.0
POINTER_RIGHT_ANCHOR_OFFSET_Y = 41.0
POINTER_ANCHOR_Z = 176.0
POINTER_JOYSTICK_RADIUS = 40.0
POINTER_HAND_DELTA_RANGE = 0.18
POINTER_JOYSTICK_ACTIVE_ALPHA = 0.035
POINTER_JOYSTICK_RETURN_ALPHA = 0.28
VAD_DEADZONE_XY = 0.045
VAD_DEADZONE_Z = 0.018
VAD_RESPONSE_ALPHA = 0.35
VAD_GAIN_V = 2.25
VAD_GAIN_A = 2.05
VAD_GAIN_D = 3.8


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
    thrust_ready: bool = False
    thrust_fired_at: float | None = None
    retreat_started_at: float | None = None
    last_z: float = 0.0


@dataclass(slots=True)
class PointerRuntimeState:
    world_number: int | None = None
    left_y: float = POINTER_BASE_Y + POINTER_LEFT_ANCHOR_OFFSET_Y
    left_z: float = POINTER_ANCHOR_Z
    right_y: float = POINTER_BASE_Y + POINTER_RIGHT_ANCHOR_OFFSET_Y
    right_z: float = POINTER_ANCHOR_Z

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
    pending_vad: tuple[float, float, float] | None = None

    def load_world(self, world_id: str, base_vad: tuple[float, float, float]) -> tuple[float, float, float]:
        self.footprint_dir.mkdir(parents=True, exist_ok=True)
        self.active_world_id = world_id
        self.base_vad = base_vad
        self.pending_vad = None

        path = self._path_for(world_id)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            current = data.get("current_vad")
            if isinstance(current, dict):
                loaded = dict_to_vad(current)
                if loaded is not None:
                    self.current_vad = loaded
                    return loaded

        self.current_vad = base_vad
        self._write(world_id, base_vad, base_vad, [])
        return base_vad

    def remember_pending(self, vad: tuple[float, float, float]) -> None:
        if self.active_world_id is None:
            return
        self.pending_vad = tuple(clamp(value, -1.0, 1.0) for value in vad)

    def step_toward_pending(
        self,
        memory: BaseMemoryState,
        world_vad: tuple[float, float, float],
        alpha: float = 0.06,
        epsilon: float = 0.006,
    ) -> tuple[tuple[float, float, float], bool]:
        if self.pending_vad is None:
            return world_vad, False

        target = self.pending_vad
        next_vad = tuple(lerp(world_vad[idx], target[idx], alpha) for idx in range(3))
        if max(abs(next_vad[idx] - target[idx]) for idx in range(3)) <= epsilon:
            self.commit(target, "interaction_settled")
            memory.set_base(*target)
            return target, True
        return next_vad, False

    def commit_pending(self, memory: BaseMemoryState, reason: str) -> bool:
        if self.pending_vad is None:
            return False
        target = self.pending_vad
        self.commit(target, reason)
        memory.set_base(*target)
        return True

    def commit_current(
        self,
        vad: tuple[float, float, float],
        memory: BaseMemoryState,
        reason: str,
    ) -> bool:
        self.pending_vad = tuple(clamp(value, -1.0, 1.0) for value in vad)
        return self.commit_pending(memory, reason)

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
        self.pending_vad = None
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
            self.footprint_store.commit_pending(memory, "space_switch")
            self.current_world_id = world_id
            self.current_world_number = None
            print(f"[final_interaction] current space -> {world_id}")
            return False
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


def draw_hand_overlay(frame, hand_results) -> None:
    if not getattr(hand_results, "hand_landmarks", None):
        return

    h, w = frame.shape[:2]
    for hand_idx, hand in enumerate(hand_results.hand_landmarks):
        user_side = ""
        if hand_idx < len(hand_results.handedness):
            mp_side = hand_results.handedness[hand_idx][0].display_name.upper()
            if mp_side in ("LEFT", "RIGHT"):
                user_side = "RIGHT" if mp_side == "LEFT" else "LEFT"

        for landmark_idx, landmark in enumerate(hand):
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 3, (120, 170, 255), -1, cv2.LINE_AA)
            label = f"{landmark_idx}:{HAND_LANDMARK_LABELS[landmark_idx]}"
            label_x = min(w - 170, x + 6)
            label_y = max(14, y - 4)
            cv2.putText(
                frame,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                (40, 40, 40),
                1,
                cv2.LINE_AA,
            )

        if user_side:
            wrist = hand[0]
            wx = int(wrist.x * w)
            wy = int(wrist.y * h)
            cv2.putText(
                frame,
                f"HAND={user_side}",
                (min(w - 120, wx + 8), min(h - 10, wy + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )


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


def end_interaction(
    session: InteractionSession,
    footprint_store: VADFootprintStore,
    world_vad: tuple[float, float, float],
    now: float,
) -> None:
    session.active = False
    session.mode = "idle"
    session.engaged_at = None
    session.grab_locked = False
    session.lost_started_at = None
    session.thrust_ready = False
    session.retreat_started_at = None
    session.released_at = now
    footprint_store.remember_pending(world_vad)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cosmos final grab/open interaction runtime")
    parser.add_argument("--camera", type=int, default=0)
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
    parser.add_argument("--camera-width", type=int, default=640, help="Requested webcam capture width.")
    parser.add_argument("--camera-height", type=int, default=360, help="Requested webcam capture height.")
    parser.add_argument("--camera-fps", type=int, default=30, help="Requested webcam FPS.")
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
    parser.add_argument(
        "--perf-log-sec",
        type=float,
        default=0.0,
        help="Print loop FPS and stage timings every N seconds when greater than 0.",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
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

    hand_extractor = HandExtractor()
    pose_extractor = PoseExtractor()
    right_tracker = FinalHandTracker(control_hand="RIGHT")
    left_tracker = FinalHandTracker(control_hand="LEFT")
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
    world_vad = memory.current_base()
    pd_mode: str | None = None
    frame_index = 0
    last_pose_landmarks: dict[str, tuple[float, float]] | None = None
    perf_last_at = time.perf_counter()
    perf_frames = 0
    perf_acc = {"capture": 0.0, "pose": 0.0, "hand": 0.0, "logic": 0.0, "draw": 0.0}

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
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
            after_hand = time.perf_counter()
            right_features = right_tracker.update(hand_results, pose_landmarks, now)
            left_features = left_tracker.update(hand_results, pose_landmarks, now)
            switch_to_camera = False
            if args.send and world_vad_state.poll_unreal(ue, memory, now) and not session.active:
                world_vad = memory.current_base()
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
            both_open = (
                pointer_world_active
                and both_visible
                and right_features.open_strength >= 0.62
                and left_features.open_strength >= 0.62
                and (right_features.open_strength + left_features.open_strength) * 0.5 >= 0.72
            )
            both_grab = pointer_world_active and both_visible and right_features.grab_active and left_features.grab_active

            if both_visible:
                pointer_xy = midpoint(
                    (right_features.x, right_features.y),
                    (left_features.x, left_features.y),
                )
                avg_z = 0.5 * (right_features.z + left_features.z)
                avg_speed = 0.5 * (right_features.speed + left_features.speed)
                avg_open = 0.5 * (right_features.open_strength + left_features.open_strength)
                avg_grab = 0.5 * (right_features.grab_strength + left_features.grab_strength)
                avg_radius = 0.5 * (right_features.palm_radius + left_features.palm_radius)
                span_x = abs(right_features.x - left_features.x)
                span_y = abs(right_features.y - left_features.y)
            else:
                pointer_xy = (-1.0, -1.0)
                avg_z = 0.0
                avg_speed = 0.0
                avg_open = 0.0
                avg_grab = 0.0
                avg_radius = 0.0
                span_x = 0.0
                span_y = 0.0

            if not session.active:
                world_vad = tuple(
                    lerp(world_vad[idx], base_vad[idx], 1.0 / (25.0 * 30.0))
                    for idx in range(3)
                )

            was_active = session.active

            if not pointer_world_active:
                if session.active:
                    end_interaction(session, footprint_store, world_vad, now)
                session.engaged_at = None
                session.grab_locked = False
                session.left_pointer_locked = False
                session.right_pointer_locked = False
                session.thrust_ready = False
                session.retreat_started_at = None
                session.lost_started_at = None
            elif not session.active:
                if both_open:
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
                        session.retreat_started_at = None
                else:
                    session.engaged_at = None
            else:
                if both_visible:
                    session.lost_started_at = None
                    if both_grab:
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
                        if both_open:
                            # Open state holds the current world state without forcing it back yet.
                            pass

                        if both_open and avg_z <= -0.08:
                            session.thrust_ready = True

                        z_velocity = (avg_z - session.last_z) / max(1e-3, 1 / 30.0)
                        if (
                            session.thrust_ready
                            and both_open
                            and z_velocity >= 1.6
                            and avg_z >= 0.20
                        ):
                            switch_to_camera = True
                            session.thrust_ready = False
                            session.thrust_fired_at = now

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
                    elif now - session.lost_started_at >= 0.35:
                        end_interaction(session, footprint_store, world_vad, now)

            if not session.active and session.released_at is not None:
                if footprint_store.pending_vad is not None:
                    world_vad, _ = footprint_store.step_toward_pending(memory, world_vad)
                else:
                    elapsed = now - session.released_at
                    if elapsed >= 5.0:
                        base_vad = memory.current_base()
                        recovery_alpha = 1.0 / max(25.0 * 30.0, 1.0)
                        world_vad = tuple(
                            lerp(world_vad[idx], base_vad[idx], recovery_alpha)
                            for idx in range(3)
                        )

            if switch_to_camera:
                footprint_store.commit_current(world_vad, memory, "switch_to_galaxy")
                if args.send_pd and pd_mode != "space":
                    pd.set_space_mode()
                    pd_mode = "space"

            session.last_z = avg_z

            pointer_x = pointer_xy[0] if session.active and both_visible else -1.0
            pointer_y = pointer_xy[1] if session.active and both_visible else -1.0
            left_pointer_active = pointer_world_active and left_features.visible and left_features.grab_active
            right_pointer_active = pointer_world_active and right_features.visible and right_features.grab_active

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
                grab_active=both_grab,
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
            if args.debug_overlay:
                draw_hand_overlay(frame, hand_results)
                h, w = frame.shape[:2]
                cx = int(clamp(pointer_x, 0.0, 1.0) * w)
                cy = int(clamp(pointer_y, 0.0, 1.0) * h)
                if session.active and both_visible:
                    color = (70, 240, 160) if payload.grab_active > 0.5 else (255, 210, 120)
                    radius = max(10, int(18 + avg_radius * 28))
                    cv2.circle(frame, (cx, cy), radius, color, 2, cv2.LINE_AA)
                    cv2.circle(frame, (cx, cy), max(4, radius // 5), color, -1, cv2.LINE_AA)

                cv2.putText(
                    frame,
                    (
                        f"mode={session.mode} active={session.active} "
                        f"V={payload.target_v:.2f} A={payload.target_a:.2f} D={payload.target_d:.2f}"
                    ),
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (0, 245, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    (
                        f"world={pointer_world_active} visible={both_visible} both_open={both_open} "
                        f"RO={right_features.open_strength:.2f} LO={left_features.open_strength:.2f} z={avg_z:.2f}"
                    ),
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (220, 245, 220),
                    2,
                    cv2.LINE_AA,
                )
            after_draw = time.perf_counter()

            cv2.imshow("Cosmos Final 0505", frame)
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
                        f"draw={perf_acc['draw'] / perf_frames * 1000:.1f}ms"
                    )
                    perf_last_at = perf_now
                    perf_frames = 0
                    perf_acc = {"capture": 0.0, "pose": 0.0, "hand": 0.0, "logic": 0.0, "draw": 0.0}
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand_extractor.close()
        pose_extractor.close()
        world_vad_state.close()
        ue.stop()
        pd.stop()


if __name__ == "__main__":
    main()
