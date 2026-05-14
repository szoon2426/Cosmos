from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    from final_hand_features import FinalHandTracker
    from final_mapper import compute_final_payload
    from final_ue_bridge import FinalUEBridge
    from hand_extractor import HandExtractor
    from pose_extractor import PoseExtractor
else:
    from .final_hand_features import FinalHandTracker
    from .final_mapper import compute_final_payload
    from .final_ue_bridge import FinalUEBridge
    from .hand_extractor import HandExtractor
    from .pose_extractor import PoseExtractor


def draw_hand_overlay(frame, hand_results) -> None:
    if not getattr(hand_results, "hand_landmarks", None):
        return

    h, w = frame.shape[:2]
    for hand in hand_results.hand_landmarks:
        for landmark in hand:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 3, (120, 170, 255), -1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cosmos final grab/open interaction runtime")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--send", action="store_true", help="Enable sending data to Unreal")
    parser.add_argument(
        "--control-hand",
        choices=("left", "right"),
        default="right",
        help="Visitor-facing control hand. Because the frame is mirrored, MediaPipe handedness is compensated internally.",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] camera {args.camera} could not be opened")
        sys.exit(1)

    hand_extractor = HandExtractor()
    pose_extractor = PoseExtractor()
    tracker = FinalHandTracker(control_hand=args.control_hand.upper())
    ue = FinalUEBridge(enabled=args.send)
    ue.start()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.03)
                continue

            now = time.time()
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose_extractor.process(frame_rgb)
            pose_landmarks = pose_extractor.extract_upper_body(pose_results)
            hand_results = hand_extractor.process(frame_rgb)
            features = tracker.update(hand_results, pose_landmarks, now)
            payload = compute_final_payload(features)

            if args.send:
                ue.send(payload)

            draw_hand_overlay(frame, hand_results)
            h, w = frame.shape[:2]
            cx = int(payload.hand_x * w)
            cy = int(payload.hand_y * h)
            if features.visible:
                color = (70, 240, 160) if payload.grab_active > 0.5 else (255, 210, 120)
                radius = max(10, int(18 + payload.palm_radius * 28))
                cv2.circle(frame, (cx, cy), radius, color, 2, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), max(4, radius // 5), color, -1, cv2.LINE_AA)
                cv2.putText(
                    frame,
                    "CONTROL",
                    (cx + 10, cy + radius + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                "FINAL_0505 GRAB / OPEN",
                (20, 40),
                cv2.FONT_HERSHEY_DUPLEX,
                0.9,
                (0, 245, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"visible={features.visible} side={features.side} grab={payload.grab_strength:.2f} open={payload.open_strength:.2f}",
                (20, 76),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"x={payload.hand_x:.2f} y={payload.hand_y:.2f} z={payload.hand_z:.2f} speed={payload.hand_speed:.2f}",
                (20, 106),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (235, 235, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"grab_active={bool(payload.grab_active)} pose_fallback={features.pose_fallback} radius={payload.palm_radius:.2f} distortion={payload.distortion_strength:.2f}",
                (20, 136),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (220, 245, 220),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Cosmos Final 0505", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand_extractor.close()
        pose_extractor.close()
        ue.stop()


if __name__ == "__main__":
    main()
