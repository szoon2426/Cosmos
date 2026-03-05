"""
main.py — 웹캠 모션 캡처 메인 실행 파일

단축키:
    T  : 세션 시작 / 종료 (현재 감지된 사람 얼굴 잠금)
    Q  : 프로그램 종료
"""

import cv2
import time

from src.capture import WebcamCapture
from src.pose import PoseEstimator
from src.hand import HandEstimator
from src.renderer import Renderer
from src.assets import check_landmarks
from src.gesture import GestureDetector, GESTURES
from src.session import SessionManager


def main():
    capture   = WebcamCapture(camera_index=0, width=1280, height=720)
    estimator = PoseEstimator()
    hand_est  = HandEstimator()
    renderer  = Renderer()
    detector  = GestureDetector()
    session   = SessionManager()

    capture.open()

    frame_index = 0
    hand_info: dict = {}

    feedbacks: list[dict] = []
    FEEDBACK_TTL = 1.8

    # 에셋 HP 값 (0~100) — 세션 간 유지
    MAX_HP = 100.0
    asset_values: dict[str, float] = {
        "statue":   MAX_HP,
        "fountain": MAX_HP,
        "flowers":  MAX_HP,
    }
    DAMAGE = {
        "punch":      -10.0,
        "kick":        -5.0,
        "both_hands":   0.0,
        "meditate":    +8.0,
    }
    TARGET_ASSET = {
        "punch":      "statue",
        "kick":       "flowers",
        "both_hands": "fountain",
        "meditate":   "__all__",
    }
    HEAL_ALL = "__all__"

    print("\n[Main] 시작! 단축키: [T] 세션시작/종료  [Q] 종료\n")

    try:
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                print("[Main] 프레임 읽기 실패, 재시도 중...")
                continue
            frame = cv2.flip(frame, 1)   # 좌우 반전 (거울 모드)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 포즈 추정
            pose_results = estimator.process(frame_rgb)
            h, w = frame.shape[:2]
            landmarks = estimator.get_landmarks_as_dict(pose_results, w, h)

            # 손 상태 감지 (격프레임 — 성능 최적화)
            if frame_index % 2 == 0:
                hand_info = hand_est.process(frame_rgb)

            # 정규화 랜드마크
            norm_landmarks = None
            if landmarks:
                norm_landmarks = [
                    {**lm, "nx": lm["x"] / w, "ny": lm["y"] / h}
                    for lm in landmarks
                ]

            # 세션 관리
            session_lms = session.update(norm_landmarks)
            active_lms  = session_lms if session.is_active else None

            active_assets = check_landmarks(active_lms)
            now = time.time()

            if session.is_active and active_lms is not None:
                for gid in detector.update(active_lms, hand_info):
                    g = GESTURES[gid]

                    existing = [fb for fb in feedbacks if fb["label"] == g["label"]]
                    if existing:
                        existing[0]["triggered_at"] = now
                    else:
                        feedbacks.append({
                            "label":        g["label"],
                            "color":        g["color"],
                            "triggered_at": now,
                        })

                    target = TARGET_ASSET.get(gid)
                    dmg    = DAMAGE.get(gid, 0.0)
                    if target == HEAL_ALL:
                        for k in asset_values:
                            asset_values[k] = min(MAX_HP, asset_values[k] + abs(dmg))
                        print(f"[Gesture] {gid} → {g['label']}  | All +{abs(dmg):.0f} HP")
                    elif target and dmg != 0.0:
                        asset_values[target] = max(0.0, min(MAX_HP, asset_values[target] + dmg))
                        print(f"[Gesture] {gid} → {g['label']}  |  {target}: {asset_values[target]:.0f} HP")
            else:
                detector.update(None, None)

            feedbacks = [fb for fb in feedbacks
                         if now - fb["triggered_at"] <= FEEDBACK_TTL]

            # 렌더링
            frame = renderer.draw_assets(frame, active_assets)
            frame = renderer.draw_asset_values(frame, asset_values, MAX_HP)
            if session.is_active and active_lms is not None:
                frame = renderer.draw_skeleton(frame, landmarks)
                frame = renderer.draw_hand_landmarks(frame, hand_info)
                frame = renderer.draw_hand_status(frame, hand_info)
            frame = renderer.draw_feedback(frame, feedbacks)
            frame = renderer.draw_session_state(frame, session.is_active, session.progress())
            frame = renderer.draw_hud(
                frame,
                is_recording=False,
                frame_count=0,
                detected=active_lms is not None,
            )

            cv2.imshow("Motion Capture", frame)
            frame_index += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[Main] 종료합니다.")
                break
            elif key == ord("t"):
                if session.is_active:
                    session.end()
                else:
                    session.start(norm_landmarks)

    finally:
        capture.release()
        estimator.close()
        hand_est.close()
        cv2.destroyAllWindows()
        print("[Main] 정리 완료.")


if __name__ == "__main__":
    main()
