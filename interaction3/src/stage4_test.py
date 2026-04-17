from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    from feature_engineering import (
        SlidingWindowBuffer,
        compute_frame_features,
        compute_snap_frame_features,
        detect_snap_gate,
        normalize_landmarks,
    )
    from hand_extractor import HandExtractor
    from model_io import load_model
    from pose_extractor import PoseExtractor
    from realtime_inference import (
        BREATH_FULL_FRAMES,
        OPEN_DISTANCE_SCALE,
        PRAYER_FULL_FRAMES,
        RISE_HEIGHT_SCALE,
        InteractionController,
        build_runtime_signals,
        clamp,
        compute_tracking_metrics,
        draw_skeleton,
    )
else:
    from .feature_engineering import (
        SlidingWindowBuffer,
        compute_frame_features,
        compute_snap_frame_features,
        detect_snap_gate,
        normalize_landmarks,
    )
    from .hand_extractor import HandExtractor
    from .model_io import load_model
    from .pose_extractor import PoseExtractor
    from .realtime_inference import (
        BREATH_FULL_FRAMES,
        OPEN_DISTANCE_SCALE,
        PRAYER_FULL_FRAMES,
        RISE_HEIGHT_SCALE,
        InteractionController,
        build_runtime_signals,
        clamp,
        compute_tracking_metrics,
        draw_skeleton,
    )


def _amounts(controller: InteractionController, signals) -> tuple[float, float]:
    open_amount = 0.0
    rise_amount = 0.0
    if controller.active_label == "open":
        open_amount = clamp((signals.open_distance - controller.open_base_distance) / OPEN_DISTANCE_SCALE, -1.0, 1.0)
    if controller.active_label == "rise":
        current_height = signals.rise_height_for_side(controller.rise_side)
        rise_amount = clamp((current_height - controller.rise_base_height) / RISE_HEIGHT_SCALE, -1.0, 1.0)
    return open_amount, rise_amount


@dataclass
class Stage4Progress:
    snap_ready_on: bool = False
    snap_ready_off: bool = False
    open_started: bool = False
    open_expand: bool = False
    open_contract: bool = False
    open_release: bool = False
    rise_started: bool = False
    rise_raise: bool = False
    rise_lower: bool = False
    rise_release: bool = False
    prayer_started: bool = False
    prayer_progress: bool = False
    prayer_release: bool = False
    breath_started: bool = False
    breath_progress: bool = False
    breath_release: bool = False
    open_peak: float = 0.0
    rise_peak: float = 0.0

    def apply(self, events: list[str], controller: InteractionController, open_amount: float, rise_amount: float) -> None:
        if "ready_enabled" in events:
            self.snap_ready_on = True
        if self.snap_ready_on and "ready_disabled" in events:
            self.snap_ready_off = True

        if "open_started" in events:
            self.open_started = True
            self.open_peak = 0.0
        if controller.active_label == "open":
            self.open_peak = max(self.open_peak, open_amount)
            if open_amount > 0.15:
                self.open_expand = True
            if self.open_expand and open_amount < max(self.open_peak - 0.12, 0.02):
                self.open_contract = True
        if "open_ended" in events:
            self.open_release = True

        if "rise_started" in events:
            self.rise_started = True
            self.rise_peak = 0.0
        if controller.active_label == "rise":
            self.rise_peak = max(self.rise_peak, rise_amount)
            if rise_amount > 0.15:
                self.rise_raise = True
            if self.rise_raise and rise_amount < self.rise_peak - 0.12:
                self.rise_lower = True
        if "rise_ended" in events:
            self.rise_release = True

        if "prayer_started" in events:
            self.prayer_started = True
        if controller.active_label == "prayer" and controller.active_frames / PRAYER_FULL_FRAMES > 0.35:
            self.prayer_progress = True
        if "prayer_ended" in events:
            self.prayer_release = True

        if "breath_started" in events:
            self.breath_started = True
        if controller.active_label == "breath" and controller.active_frames / BREATH_FULL_FRAMES > 0.35:
            self.breath_progress = True
        if "breath_ended" in events:
            self.breath_release = True


def _draw_item(frame, text: str, done: bool, x: int, y: int) -> None:
    color = (70, 170, 90) if done else (70, 70, 70)
    cv2.rectangle(frame, (x, y - 18), (x + 340, y + 10), color, -1)
    cv2.putText(frame, text, (x + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4 interaction checklist test for interaction3")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", default="interaction3/models/knn_model_snapgate_v3.joblib")
    parser.add_argument("--snap-model", default="interaction3/models/snap_model_v1.joblib")
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--snap-window-size", type=int, default=8)
    parser.add_argument("--vote-size", type=int, default=5)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model = load_model(model_path)
    snap_model_path = Path(args.snap_model)
    snap_model = load_model(snap_model_path) if snap_model_path.exists() else None
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"camera {args.camera} could not be opened")

    extractor = PoseExtractor()
    hand_extractor = HandExtractor()
    window = SlidingWindowBuffer(window_size=args.window_size)
    snap_window = SlidingWindowBuffer(window_size=args.snap_window_size)
    previous = None
    recent_predictions: deque[str] = deque(maxlen=args.vote_size)
    controller = InteractionController()
    progress = Stage4Progress()
    face_history: deque[bool] = deque(maxlen=5)
    tracking_locked = False
    tracking_state = "searching"
    lock_anchor_center = None
    lock_anchor_torso_height = None
    lost_counter = 0
    back_counter = 0
    drift_counter = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = extractor.process(frame_rgb)
            hand_results = hand_extractor.process(frame_rgb)
            landmarks = extractor.extract_upper_body(pose_results)
            face_valid = extractor.detect_face_presence(landmarks)
            metrics = compute_tracking_metrics(landmarks)
            face_history.append(face_valid)
            face_present = False

            if not tracking_locked:
                face_present = sum(face_history) >= 3 and bool(metrics["torso_present"])
                if face_present:
                    tracking_locked = True
                    tracking_state = "locked"
                    lost_counter = 0
                    back_counter = 0
                    drift_counter = 0
                    lock_anchor_center = metrics["shoulder_center"]
                    lock_anchor_torso_height = float(metrics["torso_height"])
            else:
                torso_present = bool(metrics["torso_present"])
                if not torso_present:
                    lost_counter += 1
                else:
                    lost_counter = max(0, lost_counter - 1)

                head_count = int(metrics["head_count"])
                front_face_count = int(metrics["front_face_count"])
                if torso_present and front_face_count == 0 and head_count >= 1 and not face_valid:
                    back_counter += 1
                else:
                    back_counter = max(0, back_counter - 1)

                if torso_present and lock_anchor_center is not None and lock_anchor_torso_height is not None:
                    current_center = metrics["shoulder_center"]
                    current_torso_height = float(metrics["torso_height"])
                    if current_center is not None:
                        center_dx = current_center[0] - lock_anchor_center[0]
                        center_dy = current_center[1] - lock_anchor_center[1]
                        center_dist = (center_dx * center_dx + center_dy * center_dy) ** 0.5
                        torso_ratio = current_torso_height / max(lock_anchor_torso_height, 1e-5)
                        if center_dist > 1.8 or torso_ratio < 0.35 or torso_ratio > 2.8:
                            drift_counter += 1
                        else:
                            drift_counter = max(0, drift_counter - 1)

                        if face_valid:
                            lock_anchor_center = (
                                lock_anchor_center[0] * 0.9 + current_center[0] * 0.1,
                                lock_anchor_center[1] * 0.9 + current_center[1] * 0.1,
                            )
                            lock_anchor_torso_height = lock_anchor_torso_height * 0.9 + current_torso_height * 0.1
                else:
                    drift_counter += 1

                if lost_counter >= 8 or back_counter >= 15 or drift_counter >= 12:
                    tracking_locked = False
                    tracking_state = "searching"
                    face_present = False
                    face_history.clear()
                    window.clear()
                    snap_window.clear()
                    previous = None
                    recent_predictions.clear()
                    lock_anchor_center = None
                    lock_anchor_torso_height = None
                    lost_counter = 0
                    back_counter = 0
                    drift_counter = 0
                    controller = InteractionController(ready_gate=controller.ready_gate)
                else:
                    face_present = True
                    tracking_state = "lost_pending" if (lost_counter > 0 or back_counter > 0 or drift_counter > 0) else "locked"

            signals = build_runtime_signals({}) if False else None
            if landmarks is not None:
                landmarks.update(hand_extractor.extract_keypoints(hand_results))
                draw_skeleton(frame, landmarks)

            normalized = normalize_landmarks(landmarks) if landmarks is not None else None
            if normalized is not None and face_present:
                signals = build_runtime_signals(normalized)
                frame_features = compute_frame_features(normalized, previous)
                window.add(frame_features)
                snap_side = signals.snap_gate_side
                if snap_side == "none":
                    if signals.left_rise_height > signals.right_rise_height + 0.05:
                        snap_side = "left"
                    elif signals.right_rise_height > signals.left_rise_height + 0.05:
                        snap_side = "right"
                snap_features = compute_snap_frame_features(normalized, previous, snap_side)
                snap_window.add(snap_features)
            else:
                window.clear()
                snap_window.clear()
                previous = None

            if signals is None:
                signals = build_runtime_signals(
                    {
                        "LEFT_SHOULDER": np.array([-0.3, 0.0], dtype=np.float32),
                        "RIGHT_SHOULDER": np.array([0.3, 0.0], dtype=np.float32),
                        "LEFT_ELBOW": np.array([-0.4, 0.2], dtype=np.float32),
                        "RIGHT_ELBOW": np.array([0.4, 0.2], dtype=np.float32),
                        "LEFT_WRIST": np.array([-0.5, 0.4], dtype=np.float32),
                        "RIGHT_WRIST": np.array([0.5, 0.4], dtype=np.float32),
                    }
                )

            prediction = "waiting"
            snap_event_detected = False
            snap_event = {
                "event": False,
                "thumb_index_distance": 0.0,
                "thumb_index_delta": 0.0,
                "thumb_middle_distance": 0.0,
                "thumb_middle_delta": 0.0,
                "thumb_crossed": False,
                "index_extension": 0.0,
                "middle_motion": 0.0,
                "ml_prediction": "waiting",
                "ml_confidence": 0.0,
            }
            if normalized is not None and snap_model is not None and snap_window.is_ready():
                snap_vector = snap_window.to_feature_vector().reshape(1, -1)
                snap_probs = snap_model.predict_proba(snap_vector)[0]
                snap_confidence = float(np.max(snap_probs))
                snap_prediction = str(snap_model.predict(snap_vector)[0])
                snap_event.update(
                    {
                        "event": bool(snap_prediction == "snap" and snap_confidence >= 0.5),
                        "thumb_index_distance": float(snap_vector[0][0]),
                        "thumb_middle_distance": float(snap_vector[0][1]),
                        "thumb_index_delta": float(snap_vector[0][2]),
                        "thumb_middle_delta": float(snap_vector[0][3]),
                        "middle_motion": float(snap_vector[0][4]),
                        "index_extension": float(snap_vector[0][6]),
                        "thumb_crossed": bool(snap_vector[0][11] > 0.5),
                        "ml_prediction": snap_prediction,
                        "ml_confidence": snap_confidence,
                    }
                )

            if face_present and window.is_ready():
                feature_vector = window.to_feature_vector().reshape(1, -1)
                probs = model.predict_proba(feature_vector)[0]
                max_prob = float(max(probs))
                pred_label = str(model.predict(feature_vector)[0])

                if max_prob < 0.5:
                    prediction = "uncertain"
                else:
                    recent_predictions.append(pred_label)
                    prediction = Counter(recent_predictions).most_common(1)[0][0]

                snap_event_detected = bool(snap_event["event"])

            events = controller.update(prediction, face_present, signals, snap_event_detected, now=time.time())
            open_amount, rise_amount = _amounts(controller, signals)
            progress.apply(events, controller, open_amount, rise_amount)

            cv2.putText(frame, f"track={tracking_state} face={face_present} ready={controller.ready_gate}", (20, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"prediction={prediction} active={controller.active_label} state={controller.state}", (20, 66),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            arming_elapsed = max(0.0, time.time() - controller.arming_started_at) if controller.state == "arming" else 0.0
            cv2.putText(frame, f"candidate={controller.candidate} arming_elapsed={arming_elapsed:.2f}s", (20, 98),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"open_amount={open_amount:+.2f} rise_amount={rise_amount:+.2f} rise_side={controller.rise_side}", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 120), 2, cv2.LINE_AA)
            cv2.putText(frame, f"open_ready={signals.open_ready} prayer={signals.prayer_gate} breath={signals.breath_gate} snap_gate={signals.snap_gate_side}", (20, 162),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (180, 220, 255), 2, cv2.LINE_AA)
            if normalized is not None:
                snap_debug = detect_snap_gate(normalized)
                cv2.putText(
                    frame,
                    f"snap_event={snap_event_detected} ml={snap_event['ml_prediction']} conf={snap_event['ml_confidence']:.2f} "
                    f"final_snap={snap_event_detected and signals.snap_gate_side != 'none'} "
                    f"L={snap_debug['left_gate']} R={snap_debug['right_gate']} "
                    f"Lup={snap_debug['left_above']} Rup={snap_debug['right_above']} "
                    f"Lclear={snap_debug['left_clear_side']} Rclear={snap_debug['right_clear_side']} "
                    f"Lhand={snap_debug['left_has_hand']} Rhand={snap_debug['right_has_hand']}",
                    (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 220, 120),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"ti={snap_event['thumb_index_distance']:.3f} tid={snap_event['thumb_index_delta']:.3f} "
                    f"tm={snap_event['thumb_middle_distance']:.3f} tmd={snap_event['thumb_middle_delta']:.3f} "
                    f"midmv={snap_event['middle_motion']:.3f} crossed={snap_event['thumb_crossed']} "
                    f"ext={snap_event['index_extension']:.3f}",
                    (20, 218),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (180, 220, 255),
                    2,
                    cv2.LINE_AA,
                )

            y = 270
            _draw_item(frame, "snap -> ready on", progress.snap_ready_on, 20, y)
            _draw_item(frame, "snap -> ready off", progress.snap_ready_off, 380, y)
            y += 36
            _draw_item(frame, "open start", progress.open_started, 20, y)
            _draw_item(frame, "open expand", progress.open_expand, 380, y)
            _draw_item(frame, "open contract", progress.open_contract, 740, y)
            y += 36
            _draw_item(frame, "open release", progress.open_release, 20, y)
            _draw_item(frame, "rise start", progress.rise_started, 380, y)
            _draw_item(frame, "rise raise", progress.rise_raise, 740, y)
            y += 36
            _draw_item(frame, "rise lower", progress.rise_lower, 20, y)
            _draw_item(frame, "rise release", progress.rise_release, 380, y)
            _draw_item(frame, "prayer start", progress.prayer_started, 740, y)
            y += 36
            _draw_item(frame, "prayer progress", progress.prayer_progress, 20, y)
            _draw_item(frame, "prayer release", progress.prayer_release, 380, y)
            _draw_item(frame, "breath start", progress.breath_started, 740, y)
            y += 36
            _draw_item(frame, "breath progress", progress.breath_progress, 20, y)
            _draw_item(frame, "breath release", progress.breath_release, 380, y)

            cv2.putText(frame, "Stage 4 test: trigger / sustain / release checklist", (20, y + 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "q / ESC to quit", (20, y + 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 180, 180), 2, cv2.LINE_AA)

            cv2.imshow("interaction3 stage4 test", frame)

            if normalized is not None and face_present:
                previous = normalized

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty("interaction3 stage4 test", cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        extractor.close()
        hand_extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
