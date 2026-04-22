from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    from feature_engineering import SlidingWindowBuffer, compute_frame_features, normalize_landmarks
    from hand_extractor import HandExtractor
    from model_io import load_model
    from pd_bridge import PDBridge
    from pose_extractor import PoseExtractor
    from realtime_inference import (
        InteractionController,
        RuntimeSignals,
        build_runtime_signals,
        compute_pd_continuous_values,
        draw_skeleton,
        draw_vad_meter,
    )
    from ue_bridge import UEBridge
    from vad_mapper import compute_unreal_payload
else:
    from .feature_engineering import SlidingWindowBuffer, compute_frame_features, normalize_landmarks
    from .hand_extractor import HandExtractor
    from .model_io import load_model
    from .pd_bridge import PDBridge
    from .pose_extractor import PoseExtractor
    from .realtime_inference import (
        InteractionController,
        RuntimeSignals,
        build_runtime_signals,
        compute_pd_continuous_values,
        draw_skeleton,
        draw_vad_meter,
    )
    from .ue_bridge import UEBridge
    from .vad_mapper import compute_unreal_payload


FORCE_KEYS = {
    ord("1"): "rise",
    ord("2"): "open",
    ord("3"): "prayer",
    ord("4"): "breath",
}

FORCED_TEST_VAD = {
    "rise": (0.2, 0.8, 0.2),
    "open": (0.7, 0.1, 0.7),
    "prayer": (0.2, -0.2, 0.2),
    "breath": (0.0, -0.6, 0.0),
}


def send_pd_ready(pd: PDBridge, enabled: bool, send_enabled: bool) -> None:
    if not send_enabled:
        return
    pd.send_trigger("READY_ON" if enabled else "READY_OFF")
    pd.send_value("READY_MODE", 1.0 if enabled else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime interaction test without snap gate/event")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", default="interaction3/models/knn_model_v3_uncertain.joblib")
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--vote-size", type=int, default=5)
    parser.add_argument("--send", action="store_true", help="Enable sending data to UE and PD bridges")
    parser.add_argument("--ready", action="store_true", default=True, help="Start with ready gate enabled")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model = load_model(model_path)
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
    window = SlidingWindowBuffer(window_size=args.window_size)
    recent_predictions: deque[str] = deque(maxlen=args.vote_size)
    controller = InteractionController(ready_gate=bool(args.ready))
    previous = None
    force_prediction: str | None = None

    if controller.ready_gate:
        send_pd_ready(pd, True, args.send)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = extractor.process(frame_rgb)
            hand_results = hand_extractor.process(frame_rgb)
            landmarks = extractor.extract_upper_body(pose_results)

            normalized = None
            signals = RuntimeSignals()
            face_present = landmarks is not None
            prediction = "waiting"
            max_prob = 0.0

            if landmarks is not None:
                landmarks.update(hand_extractor.extract_keypoints(hand_results))
                draw_skeleton(frame, landmarks)
                normalized = normalize_landmarks(landmarks)

            if normalized is not None:
                signals = build_runtime_signals(normalized)
                frame_features = compute_frame_features(normalized, previous)
                window.add(frame_features)

                if window.is_ready():
                    feature_vector = window.to_feature_vector().reshape(1, -1)
                    expected_dim = getattr(model, "n_features_in_", None)
                    if expected_dim is None and hasattr(model, "named_steps") and "knn" in model.named_steps:
                        expected_dim = getattr(model.named_steps["knn"], "n_features_in_", None)
                    if expected_dim is not None and feature_vector.shape[1] != expected_dim:
                        raise ValueError(
                            f"Model expects {expected_dim} features, but current extractor produced {feature_vector.shape[1]}."
                        )

                    probs = model.predict_proba(feature_vector)[0]
                    max_prob = float(np.max(probs))
                    pred_label = str(model.predict(feature_vector)[0])
                    if max_prob < 0.5:
                        prediction = "uncertain"
                    else:
                        recent_predictions.append(pred_label)
                        prediction = Counter(recent_predictions).most_common(1)[0][0]
            else:
                window.clear()
                recent_predictions.clear()
                previous = None

            used_prediction = force_prediction or prediction
            interaction_events = controller.update(
                used_prediction,
                face_present,
                signals,
                snap_event_detected=False,
                now=time.time(),
            )

            if force_prediction is not None:
                vad = FORCED_TEST_VAD[force_prediction]
                vad_source = f"forced {force_prediction} test vad"
            else:
                vad = controller.current_vad(signals)
                vad_source = controller.vad_source_description(signals)
            raw_payload = compute_unreal_payload(*vad)
            payload = raw_payload.as_dict()
            pd_values, _ = compute_pd_continuous_values(controller, signals, raw_payload)

            if args.send:
                for event in interaction_events:
                    if event == "ready_enabled":
                        pd.send_trigger("READY_ON")
                    elif event == "ready_disabled":
                        pd.send_trigger("READY_OFF")

                pd.send(raw_payload)
                for symbol, value in pd_values.items():
                    pd.send_value(symbol, value)
                if controller.state == "active":
                    ue.send(raw_payload)

            cv2.putText(frame, "NO SNAP TEST MODE", (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"prediction={prediction} used={used_prediction} conf={max_prob:.2f} force={force_prediction or 'off'}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"ready={controller.ready_gate} state={controller.state} active={controller.active_label} candidate={controller.candidate}",
                (20, 103),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            arming_elapsed = max(0.0, time.time() - controller.arming_started_at) if controller.state == "arming" else 0.0
            cv2.putText(
                frame,
                f"arming={arming_elapsed:.2f}s missing={controller.missing_count} rival={controller.rival_candidate}:{controller.rival_count}",
                (20, 136),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"WORLD TARGET VAD V={payload['V']:+.2f} A={payload['A']:+.2f} D={payload['D']:+.2f} source={vad_source}",
                (20, 430),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (120, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"UE flower={payload['density']:.2f} decay={payload['decay']:.2f} fountain={payload['min_speed']:.0f}-{payload['max_speed']:.0f}",
                (20, 456),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 255, 200),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"PD ready={pd_values['READY_MODE']:.0f} flower={pd_values['FLOWER']:.2f} decay={pd_values['DECAY']:.2f} recover={pd_values['RECOVER']:.2f}",
                (20, 482),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 255, 200),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "keys: SPACE ready | 1 rise 2 open 3 prayer 4 breath | 0 auto | q quit",
                (20, frame.shape[0] - 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 220, 255),
                2,
                cv2.LINE_AA,
            )

            meter_x = max(20, frame.shape[1] - 310)
            draw_vad_meter(frame, "V", payload["V"], meter_x, 42, (80, 220, 255))
            draw_vad_meter(frame, "A", payload["A"], meter_x, 88, (80, 255, 120))
            draw_vad_meter(frame, "D", payload["D"], meter_x, 134, (255, 180, 80))

            cv2.imshow("interaction3 no-snap test", frame)

            if normalized is not None and face_present:
                previous = normalized

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                controller.ready_gate = not controller.ready_gate
                controller.state = "idle"
                controller.active_label = "none"
                controller.candidate = "none"
                controller.arming_started_at = 0.0
                send_pd_ready(pd, controller.ready_gate, args.send)
            elif key == ord("0"):
                force_prediction = None
            elif key in FORCE_KEYS:
                force_prediction = FORCE_KEYS[key]

            if cv2.getWindowProperty("interaction3 no-snap test", cv2.WND_PROP_VISIBLE) < 1:
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
