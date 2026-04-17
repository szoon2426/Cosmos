from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    from feature_engineering import SlidingWindowBuffer, compute_frame_features, detect_snap_gate, normalize_landmarks
    from hand_extractor import HandExtractor
    from pose_extractor import PoseExtractor
else:
    from .feature_engineering import SlidingWindowBuffer, compute_frame_features, detect_snap_gate, normalize_landmarks
    from .hand_extractor import HandExtractor
    from .pose_extractor import PoseExtractor


LABEL_MAP = {
    ord("1"): "rise",
    ord("2"): "open",
    ord("3"): "prayer",
    ord("4"): "breath",
    ord("5"): "snap",
}

SKELETON_PAIRS = [
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"),
    ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP", "RIGHT_HIP"),
]

JOINT_COLORS = {
    "LEFT_WRIST": (0, 0, 255),
    "RIGHT_WRIST": (0, 0, 255),
    "LEFT_ELBOW": (0, 165, 255),
    "RIGHT_ELBOW": (0, 165, 255),
    "LEFT_SHOULDER": (255, 255, 0),
    "RIGHT_SHOULDER": (255, 255, 0),
    "LEFT_HAND_INDEX_TIP": (255, 0, 255),
    "RIGHT_HAND_INDEX_TIP": (255, 0, 255),
    "LEFT_HAND_THUMB_TIP": (0, 255, 255),
    "RIGHT_HAND_THUMB_TIP": (0, 255, 255),
}


def draw_skeleton(frame: np.ndarray, landmarks: dict[str, tuple[float, float]]) -> None:
    h, w = frame.shape[:2]
    for a, b in SKELETON_PAIRS:
        if a in landmarks and b in landmarks:
            ax, ay = int(landmarks[a][0] * w), int(landmarks[a][1] * h)
            bx, by = int(landmarks[b][0] * w), int(landmarks[b][1] * h)
            cv2.line(frame, (ax, ay), (bx, by), (0, 255, 0), 2)
    for name, (x, y) in landmarks.items():
        px, py = int(x * w), int(y * h)
        color = JOINT_COLORS.get(name, (200, 200, 200))
        cv2.circle(frame, (px, py), 5, color, -1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect labeled pose windows for interaction3")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", default="interaction3/data/dataset.jsonl")
    parser.add_argument("--window-size", type=int, default=10)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    extractor = PoseExtractor()
    hand_extractor = HandExtractor()
    window = SlidingWindowBuffer(window_size=args.window_size)
    previous = None
    current_label: str | None = None
    last_wd = 0.0
    last_wh = 0.0
    last_gate = "none"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = extractor.process(frame_rgb)
            hand_results = hand_extractor.process(frame_rgb)
            landmarks = extractor.extract_upper_body(pose_results)

            normalized = None
            if landmarks is not None:
                landmarks.update(hand_extractor.extract_keypoints(hand_results))
                draw_skeleton(frame, landmarks)
                normalized = normalize_landmarks(landmarks)
                if normalized is not None:
                    gate_info = detect_snap_gate(normalized)
                    last_gate = str(gate_info["active_side"])
                    frame_features = compute_frame_features(normalized, previous)
                    previous = normalized
                    window.add(frame_features)

                    lw = normalized["LEFT_WRIST"]
                    rw = normalized["RIGHT_WRIST"]
                    last_wd = float(np.linalg.norm(lw - rw))
                    last_wh = float((lw[1] + rw[1]) * 0.5)

                    cv2.putText(
                        frame,
                        f"wrist_dist={last_wd:.3f} wrist_y={last_wh:+.3f}",
                        (10, h - 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"snap_gate={last_gate}",
                        (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 220, 0),
                        2,
                        cv2.LINE_AA,
                    )
            else:
                cv2.putText(frame, "POSE NOT DETECTED", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            key = cv2.waitKey(1) & 0xFF
            if key in LABEL_MAP:
                current_label = LABEL_MAP[key]
            elif key in (ord("q"), 27):
                break
            elif key == ord(" "):
                if current_label and window.is_ready():
                    row = {
                        "label": current_label,
                        "window_size": args.window_size,
                        "timestamp": time.time(),
                        "feature_version": "snap_gate_v2",
                        "features": window.to_feature_vector().tolist(),
                    }
                    with output_path.open("a", encoding="utf-8") as fp:
                        fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(f"[collector] saved label={current_label} gate={last_gate} wrist_dist={last_wd:.3f} wrist_y={last_wh:+.3f}")

            ready_str = "READY" if window.is_ready() else f"{len(window.items)}/{args.window_size}"
            cv2.putText(frame, f"label={current_label or 'none'} window={ready_str} SPACE=save Q=quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "1=rise 2=open 3=prayer 4=breath 5=snap", (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.imshow("interaction3 collector", frame)
    finally:
        cap.release()
        extractor.close()
        hand_extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
