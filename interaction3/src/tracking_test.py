from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import cv2

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    from pose_extractor import PoseExtractor
else:
    from .pose_extractor import PoseExtractor


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
    "NOSE": (255, 255, 255),
    "LEFT_EYE": (255, 180, 0),
    "RIGHT_EYE": (255, 180, 0),
    "LEFT_SHOULDER": (255, 255, 0),
    "RIGHT_SHOULDER": (255, 255, 0),
    "LEFT_ELBOW": (0, 165, 255),
    "RIGHT_ELBOW": (0, 165, 255),
    "LEFT_WRIST": (0, 0, 255),
    "RIGHT_WRIST": (0, 0, 255),
    "LEFT_HIP": (100, 255, 100),
    "RIGHT_HIP": (100, 255, 100),
}


def _midpoint(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def compute_tracking_metrics(landmarks: dict[str, tuple[float, float]] | None) -> dict[str, object]:
    if not landmarks:
        return {
            "torso_present": False,
            "head_count": 0,
            "front_face_count": 0,
            "shoulder_center": None,
            "torso_height": None,
        }

    has_shoulders = "LEFT_SHOULDER" in landmarks and "RIGHT_SHOULDER" in landmarks
    has_hips = "LEFT_HIP" in landmarks and "RIGHT_HIP" in landmarks
    torso_present = has_shoulders and has_hips

    shoulder_center = None
    torso_height = None
    if torso_present:
        shoulder_center = _midpoint(landmarks["LEFT_SHOULDER"], landmarks["RIGHT_SHOULDER"])
        hip_center = _midpoint(landmarks["LEFT_HIP"], landmarks["RIGHT_HIP"])
        torso_height = max(abs(hip_center[1] - shoulder_center[1]), 1e-5)

    head_count = sum(
        1 for name in ("NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR") if name in landmarks
    )
    front_face_count = sum(
        1 for name in ("LEFT_EYE", "RIGHT_EYE", "MOUTH_LEFT", "MOUTH_RIGHT") if name in landmarks
    )

    return {
        "torso_present": torso_present,
        "head_count": head_count,
        "front_face_count": front_face_count,
        "shoulder_center": shoulder_center,
        "torso_height": torso_height,
    }


def draw_skeleton(frame, landmarks: dict[str, tuple[float, float]] | None) -> None:
    if not landmarks:
        return

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


def draw_status_badge(frame, label: str, enabled: bool, top: int) -> None:
    color = (50, 160, 80) if enabled else (60, 60, 60)
    text_color = (255, 255, 255) if enabled else (180, 180, 180)
    cv2.rectangle(frame, (20, top), (220, top + 34), color, -1)
    cv2.putText(frame, label, (32, top + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3 tracking/presence test for interaction3")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"camera {args.camera} could not be opened")

    extractor = PoseExtractor()

    face_history: deque[bool] = deque(maxlen=5)
    tracking_locked = False
    tracking_state = "searching"
    lock_anchor_center: tuple[float, float] | None = None
    lock_anchor_torso_height: float | None = None
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
            landmarks = extractor.extract_upper_body(pose_results)
            face_valid = extractor.detect_face_presence(landmarks)
            metrics = compute_tracking_metrics(landmarks)
            face_history.append(face_valid)

            if not tracking_locked:
                face_present = sum(face_history) >= 3 and bool(metrics["torso_present"])
                if face_present:
                    tracking_locked = True
                    tracking_state = "locked"
                    lost_counter = 0
                    back_counter = 0
                    drift_counter = 0
                    lock_anchor_center = metrics["shoulder_center"]  # type: ignore[assignment]
                    lock_anchor_torso_height = float(metrics["torso_height"])  # type: ignore[arg-type]
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
                    current_torso_height = float(metrics["torso_height"])  # type: ignore[arg-type]
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
                    face_history.clear()
                    lock_anchor_center = None
                    lock_anchor_torso_height = None
                    lost_counter = 0
                    back_counter = 0
                    drift_counter = 0
                    face_present = False
                else:
                    face_present = True
                    if lost_counter > 0 or back_counter > 0 or drift_counter > 0:
                        tracking_state = "lost_pending"
                    else:
                        tracking_state = "locked"

            draw_skeleton(frame, landmarks)

            draw_status_badge(frame, "FACE VALID", face_valid, 20)
            draw_status_badge(frame, "FACE PRESENT", face_present, 62)
            draw_status_badge(frame, "TORSO PRESENT", bool(metrics["torso_present"]), 104)
            draw_status_badge(frame, f"TRACK {tracking_state.upper()}", tracking_state == "locked", 146)

            cv2.putText(
                frame,
                f"face_history={sum(face_history)}/{len(face_history)}  front_face={metrics['front_face_count']}  head={metrics['head_count']}",
                (20, 225),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"lost={lost_counter}  back={back_counter}  drift={drift_counter}",
                (20, 258),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (180, 220, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "Stage 3 test: lock once, keep tracking, release only on real loss/back-turn.",
                (20, 296),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 220, 120),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "q / ESC to quit",
                (20, 328),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (180, 180, 180),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("interaction3 tracking test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty("interaction3 tracking test", cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
