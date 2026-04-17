from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    from feature_engineering import (
        SlidingWindowBuffer,
        compute_snap_frame_features,
        detect_snap_gate,
        normalize_landmarks,
    )
    from hand_extractor import HandExtractor
    from model_io import save_model
    from pose_extractor import PoseExtractor
    from simple_knn import SimpleKNNClassifier
else:
    from .feature_engineering import (
        SlidingWindowBuffer,
        compute_snap_frame_features,
        detect_snap_gate,
        normalize_landmarks,
    )
    from .hand_extractor import HandExtractor
    from .model_io import save_model
    from .pose_extractor import PoseExtractor
    from .simple_knn import SimpleKNNClassifier


def _infer_active_side(normalized_landmarks: dict[str, np.ndarray]) -> str:
    gate = detect_snap_gate(normalized_landmarks)
    active_side = str(gate["active_side"])
    if active_side != "none":
        return active_side

    lw = normalized_landmarks["LEFT_WRIST"]
    rw = normalized_landmarks["RIGHT_WRIST"]
    left_height = -float(lw[1])
    right_height = -float(rw[1])
    return "left" if left_height >= right_height else "right"


def _iter_videos(video_dir: Path, selected_names: set[str] | None = None) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for path in sorted(video_dir.glob("*.mp4")):
        if selected_names is not None and path.name.lower() not in selected_names:
            continue
        stem = path.stem.lower()
        if stem.startswith("snap"):
            label = "snap"
        elif stem.startswith("rise"):
            label = "non_snap"
        elif stem.startswith("open"):
            label = "non_snap"
        elif stem.startswith("pray"):
            label = "non_snap"
        elif stem.startswith("breath"):
            label = "non_snap"
        else:
            continue
        out.append((path, label))
    return out


def _load_annotations(path: Path) -> dict[str, list[int]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    return {str(name): [int(v) for v in values] for name, values in raw.items()}


def _auto_detect_snap_frames(
    video_path: Path,
    extractor: PoseExtractor,
    hand_extractor: HandExtractor,
    frame_step: int,
    min_separation: int,
    max_marks: int,
) -> list[int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    previous = None
    scored_frames: list[tuple[int, float]] = []
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % frame_step != 0:
                frame_idx += 1
                continue

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = extractor.process(frame_rgb)
            hand_results = hand_extractor.process(frame_rgb)
            landmarks = extractor.extract_upper_body(pose_results)
            if landmarks is None:
                frame_idx += 1
                continue

            landmarks.update(hand_extractor.extract_keypoints(hand_results))
            normalized = normalize_landmarks(landmarks)
            if normalized is None:
                frame_idx += 1
                continue

            active_side = _infer_active_side(normalized)
            features = compute_snap_frame_features(normalized, previous, active_side)
            previous = normalized

            if np.allclose(features, 0.0):
                frame_idx += 1
                continue

            ti = float(features[0])
            tm = float(features[1])
            ti_delta = float(features[2])
            tm_delta = float(features[3])
            middle_motion = float(features[4])
            thumb_motion = float(features[5])
            index_motion = float(features[6])
            index_extension = float(features[7])

            score = (
                max(tm_delta, 0.0) * 6.0
                + max(-ti_delta, 0.0) * 5.0
                + middle_motion * 3.0
                + thumb_motion * 2.0
                + index_motion * 2.0
                + max(index_extension - 0.05, 0.0) * 1.5
                + max(0.20 - ti, 0.0) * 1.5
                + max(tm - 0.12, 0.0) * 1.0
            )
            scored_frames.append((frame_idx, score))
            frame_idx += 1
    finally:
        cap.release()

    if not scored_frames:
        return []

    scores = np.asarray([score for _, score in scored_frames], dtype=np.float32)
    threshold = float(scores.mean() + scores.std())
    peaks: list[tuple[int, float]] = []
    for idx, (frame_no, score) in enumerate(scored_frames):
        prev_score = scored_frames[idx - 1][1] if idx > 0 else -1.0
        next_score = scored_frames[idx + 1][1] if idx + 1 < len(scored_frames) else -1.0
        if score >= threshold and score >= prev_score and score >= next_score:
            peaks.append((frame_no, score))

    if not peaks:
        peaks = sorted(scored_frames, key=lambda item: item[1], reverse=True)[:max_marks]
    else:
        peaks = sorted(peaks, key=lambda item: item[1], reverse=True)

    selected: list[int] = []
    for frame_no, _ in peaks:
        if all(abs(frame_no - existing) > min_separation for existing in selected):
            selected.append(frame_no)
        if len(selected) >= max_marks:
            break

    return sorted(selected)


def _label_for_window(
    video_name: str,
    default_label: str,
    window_center_frame: int,
    annotations: dict[str, list[int]],
    positive_radius: int,
) -> str | None:
    if default_label != "snap":
        return "non_snap"

    marked = annotations.get(video_name, [])
    if not marked:
        return None

    if min(abs(window_center_frame - frame) for frame in marked) <= positive_radius:
        return "snap"
    return "non_snap"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dedicated snap model for interaction3")
    parser.add_argument("--video-dir", default="video")
    parser.add_argument("--dataset", default="interaction3/data/snap_dataset_v1.jsonl")
    parser.add_argument("--output", default="interaction3/models/snap_model_v1.joblib")
    parser.add_argument("--videos", default="", help="Comma-separated mp4 file names to use for snap training")
    parser.add_argument("--annotations", default="interaction3/data/snap_annotations.json")
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--frame-step", type=int, default=6)
    parser.add_argument("--positive-radius", type=int, default=12, help="Raw-frame radius around marked snap frames")
    parser.add_argument("--auto-annotate", action="store_true", help="Auto-detect snap peaks when annotations are missing")
    parser.add_argument("--auto-max-marks", type=int, default=12)
    parser.add_argument("--neighbors", type=int, default=3)
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        raise FileNotFoundError(video_dir)

    selected_names = None
    if args.videos.strip():
        selected_names = {name.strip().lower() for name in args.videos.split(",") if name.strip()}

    videos = _iter_videos(video_dir, selected_names)
    if not videos:
        raise RuntimeError(f"No training videos found in {video_dir}")

    dataset_path = Path(args.dataset)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    annotations_path = Path(args.annotations)
    annotations = _load_annotations(annotations_path)

    extractor = PoseExtractor()
    hand_extractor = HandExtractor()

    rows: list[dict[str, object]] = []
    label_counter: Counter[str] = Counter()

    try:
        if args.auto_annotate:
            changed = False
            min_separation = max(args.positive_radius, args.frame_step * args.window_size)
            for video_path, label in videos:
                if label != "snap" or annotations.get(video_path.name):
                    continue
                auto_marks = _auto_detect_snap_frames(
                    video_path,
                    extractor,
                    hand_extractor,
                    args.frame_step,
                    min_separation=min_separation,
                    max_marks=args.auto_max_marks,
                )
                annotations[video_path.name] = auto_marks
                changed = True
                print(f"[snap_train] auto annotations {video_path.name}: {auto_marks}")
            if changed:
                annotations_path.parent.mkdir(parents=True, exist_ok=True)
                with annotations_path.open("w", encoding="utf-8") as fp:
                    json.dump(annotations, fp, ensure_ascii=False, indent=2)

        for video_path, label in videos:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"[snap_train] skip unreadable video -> {video_path.name}")
                continue

            window = SlidingWindowBuffer(window_size=args.window_size)
            previous = None
            frame_idx = 0
            saved = 0
            skipped = 0

            print(f"[snap_train] video={video_path.name} label={label}")
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_idx % args.frame_step != 0:
                    frame_idx += 1
                    continue

                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_results = extractor.process(frame_rgb)
                hand_results = hand_extractor.process(frame_rgb)
                landmarks = extractor.extract_upper_body(pose_results)
                if landmarks is None:
                    frame_idx += 1
                    continue

                landmarks.update(hand_extractor.extract_keypoints(hand_results))
                normalized = normalize_landmarks(landmarks)
                if normalized is None:
                    frame_idx += 1
                    continue

                active_side = _infer_active_side(normalized)
                snap_features = compute_snap_frame_features(normalized, previous, active_side)
                previous = normalized
                window.add(snap_features)

                if window.is_ready() and frame_idx % args.stride == 0:
                    effective_label = _label_for_window(
                        video_path.name,
                        label,
                        frame_idx,
                        annotations,
                        args.positive_radius,
                    )
                    if effective_label is None:
                        skipped += 1
                        frame_idx += 1
                        continue
                    row = {
                        "label": effective_label,
                        "source": video_path.name,
                        "window_size": args.window_size,
                        "feature_version": "snap_binary_v1",
                        "features": window.to_feature_vector().tolist(),
                    }
                    rows.append(row)
                    label_counter[effective_label] += 1
                    saved += 1

                frame_idx += 1

            cap.release()
            print(f"[snap_train] saved={saved} skipped={skipped}")
    finally:
        extractor.close()
        hand_extractor.close()

    if not rows:
        raise RuntimeError("No snap training rows were extracted")

    with dataset_path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    X = np.asarray([row["features"] for row in rows], dtype=np.float32)
    y = np.asarray([row["label"] for row in rows], dtype=str)
    model = SimpleKNNClassifier(n_neighbors=args.neighbors)
    model.fit(X, y)
    save_model(model, args.output)

    print(f"[snap_train] dataset -> {dataset_path}")
    print(f"[snap_train] model -> {args.output}")
    print(f"[snap_train] samples={len(rows)} labels={dict(sorted(label_counter.items()))}")


if __name__ == "__main__":
    main()
