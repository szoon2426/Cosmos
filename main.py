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
from src.renderer import Renderer
from src.recorder import MotionRecorder
from src.assets import check_landmarks
from src.gesture import GestureDetector, GESTURES


def main():
    capture  = WebcamCapture(camera_index=0, width=1280, height=720)
    estimator = PoseEstimator()
    renderer  = Renderer()
    recorder  = MotionRecorder(output_dir="recordings")
    detector  = GestureDetector()

    capture.open()

    frame_index = 0
    start_time  = time.time()

    # 활성 피드백 목록: [{"label": str, "color": tuple, "triggered_at": float}]
    feedbacks: list[dict] = []
    FEEDBACK_TTL = 1.8  # 이 시간 지나면 목록에서 제거 (초)

    print("\n[Main] 시작! 단축키: [R] 녹화  [S] 저장  [Q] 종료\n")

    try:
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                print("[Main] 프레임 읽기 실패, 재시도 중...")
                continue

            # BGR → RGB 변환 후 포즈 추정
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results   = estimator.process(frame_rgb)

            h, w = frame.shape[:2]
            landmarks = estimator.get_landmarks_as_dict(results, w, h)

            # 정규화 랜드마크 (에셋·제스처 판단용)
            norm_landmarks = None
            if landmarks:
                norm_landmarks = [{**lm, "nx": lm["x"] / w, "ny": lm["y"] / h}
                                  for lm in landmarks]

            # 에셋 근접 감지
            active_assets = check_landmarks(norm_landmarks)

            # 제스처 감지 → 피드백 추가
            now = time.time()
            for gid in detector.update(norm_landmarks):
                g = GESTURES[gid]
                feedbacks.append({
                    "label":        g["label"],
                    "color":        g["color"],
                    "triggered_at": now,
                })
                print(f"[Gesture] {gid} → {g['label']}")

            # 만료된 피드백 제거
            feedbacks = [fb for fb in feedbacks
                         if now - fb["triggered_at"] <= FEEDBACK_TTL]

            # 녹화 중이면 프레임 추가
            if recorder.is_recording:
                elapsed = time.time() - start_time
                recorder.add_frame(frame_index, elapsed, landmarks)

            # 렌더링 (레이어 순서: 에셋 → 스켈레톤 → 피드백 → HUD)
            frame = renderer.draw_assets(frame, active_assets)
            frame = renderer.draw_skeleton(frame, landmarks)
            frame = renderer.draw_feedback(frame, feedbacks)
            frame = renderer.draw_hud(
                frame,
                is_recording=recorder.is_recording,
                frame_count=recorder.frame_count,
                detected=landmarks is not None,
            )

            cv2.imshow("Motion Capture", frame)
            frame_index += 1

            # 키 입력 처리
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
                    print("[Main] 저장할 녹화 데이터가 없습니다. 먼저 R키로 녹화하세요.")

    finally:
        capture.release()
        estimator.close()
        cv2.destroyAllWindows()
        print("[Main] 정리 완료.")


if __name__ == "__main__":
    main()
