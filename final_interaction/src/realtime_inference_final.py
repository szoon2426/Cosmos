from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    from final_hand_features import FinalHandTracker
    from final_mapper import compute_final_payload
    from final_ue_bridge import FinalUEBridge
    from hand_extractor import HandExtractor
    from pose_extractor import PoseExtractor
else:
    from .final_hand_features import FinalHandTracker
    from .final_mapper import compute_final_payload
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


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def remap_clamped(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    if in_max == in_min:
        return out_min
    t = clamp((value - in_min) / (in_max - in_min), 0.0, 1.0)
    return out_min + (out_max - out_min) * t


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
    anchor_v: float = 0.0
    anchor_a: float = 0.0
    anchor_d: float = 0.0
    grab_anchor_x: float = 0.5
    grab_anchor_y: float = 0.5
    grab_anchor_z: float = 0.0
    grab_anchor_v: float = 0.0
    grab_anchor_a: float = 0.0
    grab_anchor_d: float = 0.0
    grab_locked: bool = False
    lost_started_at: float | None = None
    released_at: float | None = None
    thrust_ready: bool = False
    thrust_fired_at: float | None = None
    last_z: float = 0.0


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Cosmos final grab/open interaction runtime")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--send", action="store_true", help="Enable sending data to Unreal")
    parser.add_argument(
        "--control-hand",
        choices=("left", "right"),
        default="right",
        help="Visitor-facing control hand. Because the frame is mirrored, MediaPipe handedness is compensated internally.",
    )
    parser.add_argument("--base-v", type=float, default=0.0)
    parser.add_argument("--base-a", type=float, default=0.0)
    parser.add_argument("--base-d", type=float, default=0.0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] camera {args.camera} could not be opened")
        sys.exit(1)

    hand_extractor = HandExtractor()
    pose_extractor = PoseExtractor()
    tracker = FinalHandTracker(control_hand=args.control_hand.upper())
    ue = FinalUEBridge(enabled=args.send)
    ue.start()
    memory = BaseMemoryState(base_v=args.base_v, base_a=args.base_a, base_d=args.base_d)
    session = InteractionSession()
    world_vad = memory.current_base()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.03)
                continue

            now = time.time()
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose_extractor.process(frame_rgb)
            pose_landmarks = pose_extractor.extract_upper_body(pose_results)
            hand_results = hand_extractor.process(frame_rgb)
            features = tracker.update(hand_results, pose_landmarks, now)
            switch_to_camera = False
            base_vad = memory.current_base()

            if not session.active:
                world_vad = tuple(
                    lerp(world_vad[idx], base_vad[idx], 1.0 / (25.0 * 30.0))
                    for idx in range(3)
                )

            if not session.active:
                # Enter as soon as the hand is clearly open and present near the screen.
                # The previous z gate was too strict and blocked idle -> active even when
                # open/grab recognition itself was working well.
                if features.visible and features.open_strength >= 0.70 and features.z >= -0.05:
                    if session.engaged_at is None:
                        session.engaged_at = now
                    elif now - session.engaged_at >= 0.08:
                        session.active = True
                        session.mode = "open"
                        session.open_anchor_x = features.x
                        session.open_anchor_y = features.y
                        session.open_anchor_z = features.z
                        session.anchor_v, session.anchor_a, session.anchor_d = world_vad
                        session.released_at = None
                else:
                    session.engaged_at = None
            else:
                if features.visible:
                    session.lost_started_at = None
                    if features.grab_active:
                        if not session.grab_locked:
                            session.grab_locked = True
                            session.mode = "grab"
                            session.grab_anchor_x = features.x
                            session.grab_anchor_y = features.y
                            session.grab_anchor_z = features.z
                            session.grab_anchor_v, session.grab_anchor_a, session.grab_anchor_d = world_vad

                        dx = features.x - session.grab_anchor_x
                        dy = session.grab_anchor_y - features.y
                        dz = features.z - session.grab_anchor_z
                        target_v = clamp(session.grab_anchor_v + dx * 2.8, -1.0, 1.0)
                        target_a = clamp(session.grab_anchor_a + dy * 2.6, -1.0, 1.0)
                        target_d = session.grab_anchor_d
                        if dz < 0.0:
                            target_d = clamp(session.grab_anchor_d + dz * 2.2, -1.0, 1.0)

                        world_vad = (
                            lerp(world_vad[0], target_v, 0.35),
                            lerp(world_vad[1], target_a, 0.35),
                            lerp(world_vad[2], target_d, 0.35),
                        )
                    else:
                        session.grab_locked = False
                        session.mode = "open"
                        if features.open_strength >= 0.70:
                            # Open state holds the current world state without forcing it back yet.
                            pass

                        if features.z <= -0.08 and features.open_strength >= 0.70:
                            session.thrust_ready = True

                        z_velocity = (features.z - session.last_z) / max(1e-3, 1 / 30.0)
                        if (
                            session.thrust_ready
                            and features.open_strength >= 0.70
                            and z_velocity >= 1.6
                            and features.z >= 0.20
                        ):
                            switch_to_camera = True
                            session.thrust_ready = False
                            session.thrust_fired_at = now
                else:
                    if session.lost_started_at is None:
                        session.lost_started_at = now
                    elif now - session.lost_started_at >= 0.35:
                        session.active = False
                        session.mode = "idle"
                        session.engaged_at = None
                        session.grab_locked = False
                        session.lost_started_at = None
                        session.thrust_ready = False
                        session.released_at = now
                        memory.learn_from(world_vad)

            if not session.active and session.released_at is not None:
                elapsed = now - session.released_at
                if elapsed >= 5.0:
                    base_vad = memory.current_base()
                    recovery_alpha = 1.0 / max(25.0 * 30.0, 1.0)
                    world_vad = tuple(
                        lerp(world_vad[idx], base_vad[idx], recovery_alpha)
                        for idx in range(3)
                    )

            session.last_z = features.z

            pointer_x = features.x if session.active and features.visible else -1.0
            pointer_y = features.y if session.active and features.visible else -1.0
            payload = compute_final_payload(
                interaction_active=session.active,
                pointer_x=pointer_x,
                pointer_y=pointer_y,
                target_v=world_vad[0],
                target_a=world_vad[1],
                target_d=world_vad[2],
                grab_active=features.grab_active,
                open_strength=features.open_strength,
                grab_strength=features.grab_strength,
                switch_to_camera=switch_to_camera,
            )

            if args.send:
                ue.send(payload)

            draw_hand_overlay(frame, hand_results)
            h, w = frame.shape[:2]
            cx = int(clamp(pointer_x, 0.0, 1.0) * w)
            cy = int(clamp(pointer_y, 0.0, 1.0) * h)
            if session.active and features.visible:
                color = (70, 240, 160) if payload.grab_active > 0.5 else (255, 210, 120)
                radius = max(10, int(18 + features.palm_radius * 28))
                cv2.circle(frame, (cx, cy), radius, color, 2, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), max(4, radius // 5), color, -1, cv2.LINE_AA)
                cv2.putText(
                    frame,
                    "CONTROL",
                    (cx + 10, cy + radius + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                "FINAL_0505 GRAB / OPEN",
                (20, 40),
                cv2.FONT_HERSHEY_DUPLEX,
                0.9,
                (0, 245, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"mode={session.mode} active={session.active} visible={features.visible} side={features.side}",
                (20, 76),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"pointer=({payload.pointer_x:.2f}, {payload.pointer_y:.2f}) z={features.z:.2f} speed={features.speed:.2f}",
                (20, 106),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (235, 235, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"grab={payload.grab_strength:.2f} open={payload.open_strength:.2f} camera_switch={bool(payload.switch_to_camera)}",
                (20, 136),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (220, 245, 220),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"V={payload.target_v:.2f} A={payload.target_a:.2f} D={payload.target_d:.2f}",
                (20, 166),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (140, 230, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                (
                    f"ratios I={features.index_ratio:.2f} "
                    f"M={features.middle_ratio:.2f} "
                    f"R={features.ring_ratio:.2f} "
                    f"P={features.pinky_ratio:.2f}"
                ),
                (20, 196),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 225, 140),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Cosmos Final 0505", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand_extractor.close()
        pose_extractor.close()
        ue.stop()


if __name__ == "__main__":
    main()
