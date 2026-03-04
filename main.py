"""
main.py — 웹캠 모션 캡처 메인 실행 파일

단축키:
    R  : 녹화 시작 / 정지
    S  : 현재까지 녹화된 데이터 JSON + CSV로 저장
    Q  : 프로그램 종료
"""

import cv2
import time

from src.capture import WebcamCapture
from src.pose import PoseEstimator
from src.hand import HandEstimator
from src.renderer import Renderer
from src.recorder import MotionRecorder
from src.assets import check_landmarks
from src.gesture import GestureDetector, GESTURES


def main():
    capture   = WebcamCapture(camera_index=0, width=1280, height=720)
    estimator = PoseEstimator()
    hand_est  = HandEstimator()
    renderer  = Renderer()
    recorder  = MotionRecorder(output_dir="recordings")
    detector  = GestureDetector()

    capture.open()

    frame_index = 0
    start_time  = time.time()

    feedbacks: list[dict] = []
    FEEDBACK_TTL = 1.8

    # 에셋 HP 값 (0~100)
    MAX_HP = 100.0
    asset_values: dict[str, float] = {
        "statue":   MAX_HP,
        "fountain": MAX_HP,
        "flowers":  MAX_HP,
    }
    # 제스처별 HP 변화량
    DAMAGE = {
        "punch":      -10.0,   # 조각상 주먹질 → -10
        "kick":       -5.0,    # 꽃 발길질 → -5
        "both_hands":  0.0,    # 분수대 손 펼기 → 변화 없음 (customize 가능)
    }
    # 제스처병 클리화 목표에셋 id
    TARGET_ASSET = {
        "punch":      "statue",
        "kick":       "flowers",
        "both_hands": "fountain",
    }

    print("\n[Main] 시작! 단축키: [R] 녹화  [S] 저장  [Q] 종료\n")

    try:
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                print("[Main] 프레임 읽기 실패, 재시도 중...")
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 포즈 추정
            pose_results = estimator.process(frame_rgb)
            h, w = frame.shape[:2]
            landmarks = estimator.get_landmarks_as_dict(pose_results, w, h)

            # 손 상태 감지 (주먹/손바닥)
            hand_info = hand_est.process(frame_rgb)

            # 정규화 랜드마크
            norm_landmarks = None
            if landmarks:
                norm_landmarks = [
                    {**lm, "nx": lm["x"] / w, "ny": lm["y"] / h}
                    for lm in landmarks
                ]

            # 에셋 근접 감지
            active_assets = check_landmarks(norm_landmarks)

            # 제스처 감지 → 피드백 + HP 감소
            now = time.time()
            for gid in detector.update(norm_landmarks, hand_info):
                g = GESTURES[gid]
                feedbacks.append({
                    "label":        g["label"],
                    "color":        g["color"],
                    "triggered_at": now,
                })
                # 에셋 HP 감소
                target = TARGET_ASSET.get(gid)
                dmg    = DAMAGE.get(gid, 0.0)
                if target and dmg != 0.0:
                    asset_values[target] = max(0.0, min(MAX_HP, asset_values[target] + dmg))
                print(f"[Gesture] {gid} → {g['label']}  |  {target}: {asset_values.get(target, '-'):.0f} HP")

            # 만료 피드백 제거
            feedbacks = [fb for fb in feedbacks
                         if now - fb["triggered_at"] <= FEEDBACK_TTL]

            # 녹화
            if recorder.is_recording:
                elapsed = time.time() - start_time
                recorder.add_frame(frame_index, elapsed, landmarks)

            # 렌더링
            frame = renderer.draw_assets(frame, active_assets)
            frame = renderer.draw_asset_values(frame, asset_values, MAX_HP)
            frame = renderer.draw_skeleton(frame, landmarks)
            frame = renderer.draw_hand_landmarks(frame, hand_info)
            frame = renderer.draw_hand_status(frame, hand_info)
            frame = renderer.draw_feedback(frame, feedbacks)
            frame = renderer.draw_hud(
                frame,
                is_recording=recorder.is_recording,
                frame_count=recorder.frame_count,
                detected=landmarks is not None,
            )

            cv2.imshow("Motion Capture", frame)
            frame_index += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[Main] 종료합니다.")
                break
            elif key == ord("r"):
                if recorder.is_recording:
                    recorder.stop()
                else:
                    start_time = time.time()
                    recorder.start()
            elif key == ord("s"):
                if recorder.frame_count > 0:
                    recorder.save_json()
                    recorder.save_csv()
                else:
                    print("[Main] 저장할 데이터 없음. R키로 녹화 먼저 하세요.")

    finally:
        capture.release()
        estimator.close()
        hand_est.close()
        cv2.destroyAllWindows()
        print("[Main] 정리 완료.")


if __name__ == "__main__":
    main()
