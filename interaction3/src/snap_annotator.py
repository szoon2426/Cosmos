from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def load_annotations(path: Path) -> dict[str, list[int]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return {str(k): [int(v) for v in values] for k, values in data.items()}


def save_annotations(path: Path, annotations: dict[str, list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(annotations, fp, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate snap event frames for snap-only training")
    parser.add_argument("--video", required=True)
    parser.add_argument("--annotations", default="interaction3/data/snap_annotations.json")
    parser.add_argument("--step", type=int, default=1, help="Display every Nth raw frame")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    annotations_path = Path(args.annotations)
    annotations = load_annotations(annotations_path)
    marked_frames = set(annotations.get(video_path.name, []))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")

    frame_index = 0
    paused = True
    current_frame = None

    try:
        while True:
            if not paused or current_frame is None:
                ok, frame = cap.read()
                if not ok:
                    paused = True
                else:
                    current_frame = cv2.flip(frame, 1)
                    current_frame_index = frame_index
                    frame_index += args.step
                    for _ in range(args.step - 1):
                        if not cap.read()[0]:
                            break
            if current_frame is None:
                break

            display = current_frame.copy()
            color = (0, 255, 255) if current_frame_index in marked_frames else (180, 180, 180)
            cv2.putText(display, f"{video_path.name}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display, f"frame={current_frame_index}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
            cv2.putText(display, "[space]=mark/unmark  [n]=next  [p]=play/pause  [s]=save  [q]=quit", (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)
            cv2.imshow("snap annotator", display)

            key = cv2.waitKey(0 if paused else 1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if current_frame_index in marked_frames:
                    marked_frames.remove(current_frame_index)
                else:
                    marked_frames.add(current_frame_index)
            elif key == ord("n"):
                paused = True
                current_frame = None
            elif key == ord("p"):
                paused = not paused
            elif key == ord("s"):
                annotations[video_path.name] = sorted(marked_frames)
                save_annotations(annotations_path, annotations)
                print(f"[snap_annotator] saved {video_path.name}: {sorted(marked_frames)}")
    finally:
        annotations[video_path.name] = sorted(marked_frames)
        save_annotations(annotations_path, annotations)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
