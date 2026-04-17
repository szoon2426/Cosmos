from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    from feature_engineering import SlidingWindowBuffer, compute_frame_features, normalize_landmarks
    from hand_extractor import HandExtractor
    from pose_extractor import PoseExtractor
else:
    from .feature_engineering import SlidingWindowBuffer, compute_frame_features, normalize_landmarks
    from .hand_extractor import HandExtractor
    from .pose_extractor import PoseExtractor


VALID_LABELS = {"rise", "open", "prayer", "breath", "snap"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract training windows from a video file")
    parser.add_argument("--video", required=True)
    parser.add_argument("--label", required=True, choices=sorted(VALID_LABELS))
    parser.add_argument("--output", default="interaction3/data/dataset.jsonl")
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    video_path = Path(args.video)
    output_path = Path(args.output)
    if not video_path.exists():
        print(f"[ERROR] video not found: {video_path}")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    extractor = PoseExtractor()
    hand_extractor = HandExtractor()
    window = SlidingWindowBuffer(window_size=args.window_size)
    previous = None

    saved = 0
    skipped = 0
    frame_idx = 0

    print(f"[video_collector] video={video_path.name} label={args.label}")
    print(f"[video_collector] total_frames={total_frames} fps={fps:.1f} stride={args.stride}")

    with output_path.open("a", encoding="utf-8") as fp:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = extractor.process(frame_rgb)
            hand_results = hand_extractor.process(frame_rgb)
            landmarks = extractor.extract_upper_body(pose_results)

            if landmarks is not None:
                landmarks.update(hand_extractor.extract_keypoints(hand_results))
                normalized = normalize_landmarks(landmarks)
                if normalized is not None:
                    frame_features = compute_frame_features(normalized, previous)
                    previous = normalized
                    window.add(frame_features)

                    if window.is_ready() and (frame_idx % args.stride == 0):
                        row = {
                            "label": args.label,
                            "window_size": args.window_size,
                            "timestamp": time.time(),
                            "source": video_path.name,
                            "feature_version": "snap_gate_v2",
                            "features": window.to_feature_vector().tolist(),
                        }
                        fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                        saved += 1
                else:
                    skipped += 1
            else:
                skipped += 1

            if args.preview:
                cv2.putText(frame, f"label={args.label} saved={saved}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("video_collector preview", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

            frame_idx += 1
            if frame_idx % 60 == 0:
                pct = frame_idx / max(total_frames, 1) * 100
                print(f"  {pct:5.1f}% frame={frame_idx} saved={saved} skipped={skipped}")

    cap.release()
    extractor.close()
    hand_extractor.close()
    cv2.destroyAllWindows()

    print(f"[video_collector] done saved={saved} skipped={skipped} output={output_path}")


if __name__ == "__main__":
    main()
