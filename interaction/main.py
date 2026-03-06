"""
main.py — 웹캠 모션 캡처 메인 실행 파일

두 창:
  [Camera]      : 원본 카메라 피드 (클린)
  [Design View] : design.png 위에 관절·에셋·HP바·피드백 오버레이

단축키:
    T  : 세션 시작 / 종료
    Q  : 프로그램 종료
"""

import cv2
import numpy as np
import time
import os
from datetime import datetime

from src.capture import WebcamCapture
from src.pose import PoseEstimator
from src.hand import HandEstimator
from src.assets import check_landmarks
from src.gesture import GestureDetector, GESTURES
from src.session import SessionManager

def main():
    capture   = WebcamCapture(camera_index=0, width=1280, height=720)
    estimator = PoseEstimator()
    hand_est  = HandEstimator()
    detector  = GestureDetector()
    session   = SessionManager()

    capture.open()

    # 비디오 녹화기 설정
    os.makedirs("recordings", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f"recordings/camera_log_{ts}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # mp4 코덱
    out_video = cv2.VideoWriter(video_path, fourcc, 30.0, (1280, 720))
    print(f"[Main] 🎥 백그라운드 비디오 녹화 시작 -> {video_path}")

    frame_index = 0
    hand_info: dict = {}
    feedbacks: list[dict] = []
    FEEDBACK_TTL = 1.8

    # ── 전역 VAD 상태 (중립: 0.0, 범위: -1.0 ~ +1.0) ──────
    vad: dict[str, float] = {"V": 0.0, "A": 0.0, "D": 0.0}

    # 제스처 발동 시 각 VAD 축에 더해지는 델타값
    VAD_DELTA: dict[str, dict[str, float]] = {
        "punch":      {"D": -0.1},                         # 조각상 → Dominance 감소
        "both_hands": {"A": -0.1, "V": -0.1},             # 분수대 → Arousal + Valence 감소
        "kick":       {"V": -0.1},                         # 나무/꽃 → Valence 감소
        "meditate":   {"V": +0.1, "A": +0.1, "D": +0.1}, # 명상   → 전체 회복
    }

    print("\n[Main] 시스템 구동 완료! (종료하려면 터미널에서 Ctrl+C를 누르세요)\n")

    consecutive_failures = 0
    MAX_FAILURES = 30  # 약 1초 동안 프레임 안 들어오면 카메라 연결 끊김으로 간주

    try:
        while True:
            # 1) 카메라 객체가 아예 끊어졌을 때 복구
            if not capture.is_opened():
                print("\n[Main] ⚠️ 카메라 연결 유실 감지! 재연결을 시도합니다...")
                capture.release()
                time.sleep(1.0)
                try:
                    capture.open()
                except Exception as e:
                    print(f"[Main] ❌ 카메라 재연결 실패: {e}")
                    time.sleep(1.0)
                continue

            # 2) 프레임 읽기 시도
            success, frame = capture.read()
            if not success or frame is None:
                consecutive_failures += 1
                if consecutive_failures % 10 == 0:
                    print(f"[Main] ⚠️ 프레임 읽기 실패 연속 ({consecutive_failures}/{MAX_FAILURES})")
                
                # 3) 연속 실패 임계치 도달 -> 내부적으로 카메라가 뻗었다고 간주하고 강제 리셋
                if consecutive_failures >= MAX_FAILURES:
                    print("\n[Main] 🚨 카메라 프레임 응답 없음! 강제로 카메라를 재시작합니다.")
                    capture.release()
                    time.sleep(1.5)  # 윈도우 OS가 디바이스 자원을 회수할 시간 부여
                    try:
                        capture.open()
                        consecutive_failures = 0  # 성공적으로 열렸으면 카운트 초기화
                        print("[Main] ✅ 카메라 재시작 성공!")
                    except Exception as e:
                        print(f"[Main] ❌ 카메라 강제 재시작 실패: {e}")
                        consecutive_failures = 0 # 터지지 않게 일단 넘김
                continue
            
            # 성공적으로 읽었으면 초기화
            consecutive_failures = 0
            
            frame = cv2.flip(frame, 1)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 포즈 추정
            pose_results = estimator.process(frame_rgb)
            h, w = frame.shape[:2]
            landmarks = estimator.get_landmarks_as_dict(pose_results, w, h)

            # 손 상태 감지 (격프레임)
            if frame_index % 2 == 0:
                hand_info = hand_est.process(frame_rgb)

            # 정규화 랜드마크
            norm_landmarks = None
            if landmarks:
                norm_landmarks = [
                    {**lm, "nx": lm["x"] / w, "ny": lm["y"] / h, "nz": lm["z"]}
                    for lm in landmarks
                ]

            # 세션 관리 (자동 시작)
            if not session.is_active and norm_landmarks is not None:
                print("[Main] 👀 사람 감지 - 세션을 자동 시작합니다.")
                session.start(norm_landmarks)
            
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
                    # VAD 값 갱신
                    delta = VAD_DELTA.get(gid, {})
                    for axis, change in delta.items():
                        vad[axis] = max(-1.0, min(1.0, vad[axis] + change))
                    if delta:
                        vad_str = "  ".join(f"{k}:{vad[k]:+.2f}" for k in ("V", "A", "D"))
                        print(f"[Gesture] {gid} → {g['label']}  | {vad_str}")
            else:
                detector.update(None, None)

            feedbacks = [fb for fb in feedbacks
                         if now - fb["triggered_at"] <= FEEDBACK_TTL]

            # ── 화면 출력 대신 백그라운드 녹화 수행 ────────────────────
            out_video.write(frame)

            frame_index += 1

    except KeyboardInterrupt:
        print("\n[Main] 터미널 인터럽트(Ctrl+C) 감지. 종료 절차를 시작합니다.")
    finally:
        capture.release()
        if 'out_video' in locals():
            out_video.release()
        estimator.close()
        hand_est.close()
        cv2.destroyAllWindows()
        print("[Main] 정리 완료.")


if __name__ == "__main__":
    main()
