from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


LABEL_ALIASES = {
    "rise": "rise",
    "open": "open",
    "pray": "prayer",
    "prayer": "prayer",
    "breath": "breath",
    "snap": "snap",
    "uncertain": "uncertain",
}


def infer_label(video_path: Path) -> str | None:
    stem = video_path.stem.lower()
    for prefix, label in LABEL_ALIASES.items():
        if stem.startswith(prefix):
            return label
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build interaction3 dataset from all labeled videos in a directory")
    parser.add_argument("--video-dir", default="video/v3")
    parser.add_argument("--output", default="interaction3/data/dataset_v3.jsonl")
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--clear-output", action="store_true")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        raise FileNotFoundError(video_dir)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.clear_output and output_path.exists():
        output_path.unlink()

    videos = sorted(video_dir.glob("*.mp4"))
    if not videos:
        raise RuntimeError(f"No mp4 files found in {video_dir}")

    collector_path = Path(__file__).resolve().parent / "video_collector.py"
    python_exe = sys.executable

    processed = 0
    skipped: list[str] = []
    for video_path in videos:
        label = infer_label(video_path)
        if label is None:
            skipped.append(video_path.name)
            print(f"[build_dataset] skip unknown label -> {video_path.name}")
            continue

        cmd = [
            python_exe,
            str(collector_path),
            "--video",
            str(video_path),
            "--label",
            label,
            "--output",
            str(output_path),
            "--window-size",
            str(args.window_size),
            "--stride",
            str(args.stride),
        ]
        if args.preview:
            cmd.append("--preview")

        print(f"[build_dataset] {video_path.name} -> {label}")
        subprocess.run(cmd, check=True)
        processed += 1

    print(f"[build_dataset] processed={processed} output={output_path}")
    if skipped:
        print(f"[build_dataset] skipped={skipped}")


if __name__ == "__main__":
    main()
