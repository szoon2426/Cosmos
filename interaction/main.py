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

from src.capture import WebcamCapture
from src.pose import PoseEstimator
from src.hand import HandEstimator
from src.renderer import Renderer
from src.assets import check_landmarks
from src.gesture import GestureDetector, GESTURES
from src.session import SessionManager

DESIGN_PATH   = "design.png"
DESIGN_WIDTH  = 1920    # 표시용 해상도 (시작 시 한 번만 리사이즈)


def load_design(path: str, target_w: int = DESIGN_WIDTH) -> np.ndarray:
    """design.png 로드 후 target_w 기준으로 고품질 리사이즈."""
    if os.path.exists(path):
        img = cv2.imread(path)
        if img is not None:
            oh, ow = img.shape[:2]
            target_h = int(oh * target_w / ow)
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            print(f"[Design] {path} 로드 완료 ({ow}x{oh} → {target_w}x{target_h})")
            return img
    print(f"[Design] {path} 없음 — 기본 배경 생성")
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def main():
    # design.png 한 번만 로드
    design_base = load_design(DESIGN_PATH)
    dh, dw = design_base.shape[:2]

    capture   = WebcamCapture(camera_index=0, width=1280, height=720)
    estimator = PoseEstimator()
    hand_est  = HandEstimator()
    renderer  = Renderer()
    detector  = GestureDetector()
    session   = SessionManager()

    capture.open()

    # 창 생성
    cv2.namedWindow("Camera",      cv2.WINDOW_NORMAL)
    cv2.namedWindow("Design View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera",      640, 360)
    cv2.resizeWindow("Design View", dw, dh)   # design.png 실제 크기로 표시


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

    print("\n[Main] 시작! 단축키: [T] 세션시작/종료  [Q] 종료\n")

    try:
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                print("[Main] 프레임 읽기 실패, 재시도 중...")
                continue
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

            # ── 창 1: 카메라 피드 (클린) ────────────────────
            cv2.imshow("Camera", frame)

            # ── 창 2: design.png 위에 오버레이 ──────────────
            design_frame = design_base.copy()

            # 에셋 영역 + VAD 바
            design_frame = renderer.draw_assets(design_frame, active_assets)
            design_frame = renderer.draw_vad_bars(design_frame, vad)

            # 세션 활성 시: 스켈레톤 + 손 관절 (정규화 좌표로 design 크기에 맞춤)
            if session.is_active and active_lms is not None:
                design_frame = renderer.draw_skeleton_norm(design_frame, active_lms)
                # 손 랜드마크도 정규화 좌표 기반으로 design 프레임에 그리기
                design_frame = renderer.draw_hand_landmarks(design_frame, hand_info)
                design_frame = renderer.draw_hand_status(design_frame, hand_info)

            # 피드백 + 세션 상태
            design_frame = renderer.draw_feedback(design_frame, feedbacks)
            design_frame = renderer.draw_session_state(
                design_frame, session.is_active, session.progress()
            )
            design_frame = renderer.draw_hud(
                design_frame,
                is_recording=False,
                frame_count=0,
                detected=active_lms is not None,
            )

            cv2.imshow("Design View", design_frame)

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
