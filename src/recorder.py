import json
import csv
import os
from datetime import datetime


class MotionRecorder:
    """포즈 랜드마크 데이터를 프레임 단위로 녹화하고 파일로 저장하는 클래스."""

    def __init__(self, output_dir: str = "recordings"):
        self.output_dir = output_dir
        self._frames: list[dict] = []
        self._recording = False
        os.makedirs(output_dir, exist_ok=True)

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def start(self):
        """녹화를 시작합니다."""
        self._frames = []
        self._recording = True
        print("[Recorder] 녹화 시작")

    def stop(self):
        """녹화를 정지합니다."""
        self._recording = False
        print(f"[Recorder] 녹화 정지 — {len(self._frames)} 프레임 수집됨")

    def add_frame(self, frame_index: int, timestamp: float, landmarks: list[dict] | None):
        """
        현재 프레임의 랜드마크를 기록합니다.

        Args:
            frame_index: 프레임 번호
            timestamp: 경과 시간(초)
            landmarks: pose.py의 get_landmarks_as_dict 결과
        """
        if not self._recording:
            return
        self._frames.append({
            "frame": frame_index,
            "timestamp": round(timestamp, 4),
            "landmarks": landmarks or [],
        })

    def save_json(self) -> str:
        """녹화된 데이터를 JSON 파일로 저장하고 파일 경로를 반환합니다."""
        filename = self._make_filename("json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({"frames": self._frames}, f, ensure_ascii=False, indent=2)
        print(f"[Recorder] JSON 저장 완료 → {filename}")
        return filename

    def save_csv(self) -> str:
        """녹화된 데이터를 CSV 파일로 저장하고 파일 경로를 반환합니다."""
        filename = self._make_filename("csv")
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "timestamp", "landmark_name", "x", "y", "z", "visibility"])
            for entry in self._frames:
                for lm in entry["landmarks"]:
                    writer.writerow([
                        entry["frame"],
                        entry["timestamp"],
                        lm["name"],
                        round(lm["x"], 2),
                        round(lm["y"], 2),
                        round(lm["z"], 4),
                        round(lm["visibility"], 4),
                    ])
        print(f"[Recorder] CSV 저장 완료 → {filename}")
        return filename

    def _make_filename(self, ext: str) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.output_dir, f"motion_{ts}.{ext}")
