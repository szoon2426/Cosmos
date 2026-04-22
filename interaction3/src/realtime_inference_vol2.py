from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import cv2

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    from conduct_features import ConductFeatures, compute_conduct_features
    from conduct_mapper import ConductDebug, compute_conduct_vad
    from conduct_state import ConductSession, READY_HOLD_SECONDS
    from eeg_profile_loader import BaseVADState, load_eeg_profile, resolve_base_vad
    from feature_engineering import normalize_landmarks
    from hand_extractor import HandExtractor
    from pd_bridge import PDBridge
    from pose_extractor import PoseExtractor
    from realtime_inference import draw_skeleton, draw_vad_meter
    from ue_bridge import UEBridge
    from vad_mapper import compute_unreal_payload
else:
    from .conduct_features import ConductFeatures, compute_conduct_features
    from .conduct_mapper import ConductDebug, compute_conduct_vad
    from .conduct_state import ConductSession, READY_HOLD_SECONDS
    from .eeg_profile_loader import BaseVADState, load_eeg_profile, resolve_base_vad
    from .feature_engineering import normalize_landmarks
    from .hand_extractor import HandExtractor
    from .pd_bridge import PDBridge
    from .pose_extractor import PoseExtractor
    from .realtime_inference import draw_skeleton, draw_vad_meter
    from .ue_bridge import UEBridge
    from .vad_mapper import compute_unreal_payload


READY_OFF_RECOVERY_DELAY_SECONDS = 5.0
READY_OFF_RECOVERY_DURATION_SECONDS = 25.0
BASE_MEMORY_LEARN_RATE = 0.02


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def blend_triplet(
    current: tuple[float, float, float],
    target: tuple[float, float, float],
    alpha: float,
) -> tuple[float, float, float]:
    alpha = clamp(alpha, 0.0, 1.0)
    return (
        lerp(current[0], target[0], alpha),
        lerp(current[1], target[1], alpha),
        lerp(current[2], target[2], alpha),
    )


@dataclass(slots=True)
class BaseMemoryState:
    dv: float = 0.0
    da: float = 0.0
    dd: float = 0.0

    def apply_to(self, base_vad: tuple[float, float, float]) -> tuple[float, float, float]:
        return (
            clamp(base_vad[0] + self.dv, -1.0, 1.0),
            clamp(base_vad[1] + self.da, -1.0, 1.0),
            clamp(base_vad[2] + self.dd, -1.0, 1.0),
        )

    def learn_from(
        self,
        base_vad: tuple[float, float, float],
        observed_vad: tuple[float, float, float],
        alpha: float = BASE_MEMORY_LEARN_RATE,
    ) -> None:
        target_dv = observed_vad[0] - base_vad[0]
        target_da = observed_vad[1] - base_vad[1]
        target_dd = observed_vad[2] - base_vad[2]
        self.dv = clamp(lerp(self.dv, target_dv, alpha), -0.25, 0.25)
        self.da = clamp(lerp(self.da, target_da, alpha), -0.25, 0.25)
        self.dd = clamp(lerp(self.dd, target_dd, alpha), -0.25, 0.25)


class EmaValue:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.initialized = False
        self.value = 0.0

    def apply(self, value: float) -> float:
        if not self.initialized:
            self.initialized = True
            self.value = value
            return value
        self.value = self.value + (value - self.value) * self.alpha
        return self.value


class ConductSmoother:
    def __init__(self) -> None:
        self.right_x = EmaValue(0.22)
        self.right_y = EmaValue(0.22)
        self.left_open = EmaValue(0.18)

    def apply(self, features: ConductFeatures) -> ConductFeatures:
        smoothed = replace(features)
        if features.right_hand_visible:
            smoothed.right_index_x = self.right_x.apply(features.right_index_x)
            smoothed.right_index_y = self.right_y.apply(features.right_index_y)
        if features.left_hand_open_valid:
            smoothed.left_hand_open = self.left_open.apply(features.left_hand_open)
        return smoothed


def send_pd_ready(pd: PDBridge, enabled: bool) -> None:
    pd.send_trigger("READY_ON" if enabled else "READY_OFF")
    pd.send_value("READY_MODE", 1.0 if enabled else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime vol2 conducting inference")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--send", action="store_true", help="Enable sending data to UE and PD bridges")
    parser.add_argument("--eeg-result", help="Path to EEG result.json or a job directory")
    parser.add_argument("--eeg-jobs-root", default="eeg-emotions/artifacts/jobs")
    parser.add_argument("--eeg-min-valid-window-ratio", type=float, default=0.70)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    ue = UEBridge(enabled=args.send)
    pd = PDBridge(enabled=args.send)
    ue.start()
    pd.start()

    if not cap.isOpened():
        print(f"[ERROR] camera {args.camera} could not be opened")
        ue.stop()
        pd.stop()
        sys.exit(1)

    extractor = PoseExtractor()
    hand_extractor = HandExtractor()
    smoother = ConductSmoother()
    session = ConductSession()
    eeg_base_state = BaseVADState()
    base_memory_state = BaseMemoryState()
    world_vad = (0.0, 0.0, 0.0)
    previous_frame_time = time.time()
    conduct_debug = ConductDebug()
    smoothed_features = ConductFeatures()
    ready_off_started_at: float | None = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            now = time.time()
            dt = max(now - previous_frame_time, 1e-3)
            previous_frame_time = now

            pose_results = extractor.process(frame_rgb)
            hand_results = hand_extractor.process(frame_rgb)
            landmarks = extractor.extract_upper_body(pose_results)

            normalized = None
            raw_features = ConductFeatures()
            if landmarks is not None:
                landmarks.update(hand_extractor.extract_keypoints(hand_results))
                draw_skeleton(frame, landmarks)
                normalized = normalize_landmarks(landmarks)
                if normalized is not None:
                    raw_features = compute_conduct_features(normalized)
                    smoothed_features = smoother.apply(raw_features)

            events: list[str] = []
            if normalized is not None:
                events = session.update(smoothed_features, now)
            else:
                events = session.update(ConductFeatures(), now)

            eeg_profile = load_eeg_profile(
                result_path=args.eeg_result,
                jobs_root=args.eeg_jobs_root,
                min_valid_window_ratio=args.eeg_min_valid_window_ratio,
            )
            eeg_base_state = resolve_base_vad(eeg_profile, previous=eeg_base_state)
            eeg_base_vad = (eeg_base_state.v, eeg_base_state.a, eeg_base_state.d)
            base_vad = base_memory_state.apply_to(eeg_base_vad)

            world_vad, conduct_debug = compute_conduct_vad(
                base_vad=base_vad,
                current_world_vad=world_vad,
                session=session,
                features=smoothed_features,
                dt=dt,
            )

            if session.is_active():
                ready_off_started_at = None
            elif ready_off_started_at is None:
                ready_off_started_at = now

            if not session.is_active() and ready_off_started_at is not None:
                elapsed_off = now - ready_off_started_at
                if elapsed_off >= READY_OFF_RECOVERY_DELAY_SECONDS:
                    recovery_alpha = 1.0 / max(READY_OFF_RECOVERY_DURATION_SECONDS * 30.0, 1.0)
                    world_vad = blend_triplet(world_vad, base_vad, recovery_alpha)

            raw_payload = compute_unreal_payload(*world_vad)
            payload = raw_payload.as_dict()

            if args.send:
                if "ready_enabled" in events:
                    send_pd_ready(pd, True)
                if "ready_disabled" in events:
                    send_pd_ready(pd, False)

                pd.send(raw_payload)
                pd.send_value("READY_MODE", 1.0 if session.is_active() else 0.0)
                ue.send(raw_payload)

            if "ready_enabled" in events:
                print(f"[{now:.2f}] conduct mode ON")
            if "ready_disabled" in events:
                base_memory_state.learn_from(eeg_base_vad, world_vad)
                print(f"[{now:.2f}] conduct mode OFF")

            cv2.putText(frame, "VOL2 CONDUCT MODE", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"mode={session.mode} hold={0.0 if session.ready_started_at is None else max(0.0, now - session.ready_started_at):.2f}/{READY_HOLD_SECONDS:.1f}s",
                (20, 74),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"right_index_raised={smoothed_features.right_index_raised} right_visible={smoothed_features.right_hand_visible} left_visible={smoothed_features.left_hand_visible}",
                (20, 107),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (220, 240, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"index x={smoothed_features.right_index_x:+.3f} y={smoothed_features.right_index_y:+.3f} dx={conduct_debug.index_dx:+.3f} dy={conduct_debug.index_dy:+.3f}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (220, 240, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"right ext idx={smoothed_features.right_index_extension:.3f} mid={smoothed_features.right_middle_extension:.3f} ring={smoothed_features.right_ring_extension:.3f} pinky={smoothed_features.right_pinky_extension:.3f}",
                (20, 173),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (220, 240, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"left_open={smoothed_features.left_hand_open:.3f} base_left_open={session.base_left_hand_open:.3f} delta_open={conduct_debug.left_delta_open:+.3f} d_vel={conduct_debug.d_velocity:+.3f}",
                (20, 206),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (220, 240, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"EEG RAW BASE V={eeg_base_vad[0]:+.2f} A={eeg_base_vad[1]:+.2f} D={eeg_base_vad[2]:+.2f} usable={eeg_base_state.usable} reason={eeg_base_state.reason}",
                (20, 430),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 200, 120),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"ADAPTIVE BASE V={base_vad[0]:+.2f} A={base_vad[1]:+.2f} D={base_vad[2]:+.2f} mem=({base_memory_state.dv:+.3f},{base_memory_state.da:+.3f},{base_memory_state.dd:+.3f})",
                (20, 456),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 220, 160),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"WORLD TARGET VAD V={payload['V']:+.2f} A={payload['A']:+.2f} D={payload['D']:+.2f} off_elapsed={0.0 if ready_off_started_at is None else max(0.0, now - ready_off_started_at):.1f}s",
                (20, 482),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (120, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"UE flower={payload['flower']:.2f} tree={payload['tree']:.2f} decay={payload['decay']:.2f} fountain={payload['min_speed']:.0f}-{payload['max_speed']:.0f}",
                (20, 508),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (200, 255, 200),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"PD ready={1.0 if session.is_active() else 0.0:.0f} V={payload['V']:+.2f} A={payload['A']:+.2f} D={payload['D']:+.2f}",
                (20, 534),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (200, 255, 200),
                2,
                cv2.LINE_AA,
            )

            meter_x = max(20, frame.shape[1] - 310)
            draw_vad_meter(frame, "V", payload["V"], meter_x, 42, (80, 220, 255))
            draw_vad_meter(frame, "A", payload["A"], meter_x, 88, (80, 255, 120))
            draw_vad_meter(frame, "D", payload["D"], meter_x, 134, (255, 180, 80))

            cv2.putText(
                frame,
                "keys: q quit",
                (20, frame.shape[0] - 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 220, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("interaction3 vol2 conduct", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty("interaction3 vol2 conduct", cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        ue.stop()
        pd.stop()
        cap.release()
        extractor.close()
        hand_extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
