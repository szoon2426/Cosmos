import cv2
import numpy as np
import time

from src.pose import POSE_CONNECTIONS
from src.assets import ASSETS, PointAsset, LineAsset


class Renderer:
    """포즈 스켈레톤과 UI 정보를 프레임에 렌더링하는 클래스."""

    SKELETON_COLOR = (0, 255, 120)
    JOINT_COLOR    = (0, 200, 255)
    RECORD_COLOR   = (0, 0, 220)
    SHADOW_COLOR   = (30, 30, 30)

    def __init__(self):
        self._fps_times: list[float] = []

    # ── 스켈레톤 ──────────────────────────────────────────
    def draw_skeleton(self, frame: np.ndarray, landmarks: list[dict] | None) -> np.ndarray:
        if not landmarks:
            return frame
        for s, e in POSE_CONNECTIONS:
            if s < len(landmarks) and e < len(landmarks):
                p1 = (int(landmarks[s]["x"]), int(landmarks[s]["y"]))
                p2 = (int(landmarks[e]["x"]), int(landmarks[e]["y"]))
                cv2.line(frame, p1, p2, self.SKELETON_COLOR, 2, cv2.LINE_AA)
        for lm in landmarks:
            cv2.circle(frame, (int(lm["x"]), int(lm["y"])), 4, self.JOINT_COLOR, -1, cv2.LINE_AA)
        return frame

    # ── 에셋 오버레이 ──────────────────────────────────────
    def draw_assets(
        self,
        frame: np.ndarray,
        active: dict[str, bool] | None = None,
    ) -> np.ndarray:
        """
        에셋 위치를 화면에 오버레이합니다.
        active: {"statue": bool, "fountain": bool, ...}
        """
        h, w = frame.shape[:2]
        if active is None:
            active = {}

        for asset in ASSETS:
            is_active  = active.get(asset.id, False)
            base_color = asset.color
            draw_color = tuple(min(255, int(c * 1.4)) for c in base_color) if is_active else base_color
            alpha      = 0.35 if is_active else 0.18
            lw         = 3 if is_active else 2

            if isinstance(asset, PointAsset):
                cx = int(asset.x * w)
                cy = int(asset.y * h)
                r  = int(asset.radius * min(w, h))

                overlay = frame.copy()
                cv2.circle(overlay, (cx, cy), r, draw_color, -1)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                cv2.circle(frame, (cx, cy), r, draw_color, lw, cv2.LINE_AA)
                cv2.drawMarker(frame, (cx, cy), draw_color, cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)

                label = f"{'▶ ' if is_active else ''}{asset.name}"
                self._put_text(frame, label, (cx - 28, cy - r - 10), scale=0.65, color=draw_color)

            elif isinstance(asset, LineAsset):
                x1     = int(asset.x_start * w)
                x2     = int(asset.x_end   * w)
                cy     = int(asset.y        * h)
                band   = int(asset.band / 2 * h)

                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, cy - band), (x2, cy + band), draw_color, -1)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                # 점선 경계
                x = x1
                while x < x2:
                    ex = min(x + 14, x2)
                    cv2.line(frame, (x, cy - band), (ex, cy - band), draw_color, 1, cv2.LINE_AA)
                    cv2.line(frame, (x, cy + band), (ex, cy + band), draw_color, 1, cv2.LINE_AA)
                    x += 22

                cv2.line(frame, (x1, cy), (x2, cy), draw_color, lw, cv2.LINE_AA)
                cv2.line(frame, (x1, cy - band), (x1, cy + band), draw_color, 2, cv2.LINE_AA)
                cv2.line(frame, (x2, cy - band), (x2, cy + band), draw_color, 2, cv2.LINE_AA)

                label = f"{'▶ ' if is_active else ''}{asset.name}"
                self._put_text(frame, label, (x1 + 8, cy - band - 10), scale=0.65, color=draw_color)

        return frame

    # ── HUD ───────────────────────────────────────────────
    def draw_hud(
        self,
        frame: np.ndarray,
        is_recording: bool,
        frame_count: int,
        detected: bool,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        now = time.time()
        self._fps_times.append(now)
        self._fps_times = [t for t in self._fps_times if now - t < 1.0]
        fps = len(self._fps_times)

        self._put_text(frame, f"FPS: {fps}", (12, 32), scale=0.8)

        status_color = (0, 255, 120) if detected else (100, 100, 100)
        self._put_text(frame, "DETECTED" if detected else "NO POSE",
                       (12, 62), scale=0.7, color=status_color)

        if is_recording:
            cv2.circle(frame, (w - 28, 28), 10, self.RECORD_COLOR, -1)
            self._put_text(frame, f"REC  {frame_count}f",
                           (w - 130, 36), scale=0.7, color=self.RECORD_COLOR)

        self._put_text(frame, "[R] record  [S] save  [Q] quit",
                       (12, h - 14), scale=0.55, color=(200, 200, 200))
        return frame

    # ── 제스처 피드백 ──────────────────────────────────────
    def draw_feedback(
        self,
        frame: np.ndarray,
        feedbacks: list[dict],
    ) -> np.ndarray:
        """
        제스처 발동 시 화면 중앙에 크게 텍스트를 표시합니다.

        Args:
            feedbacks: [{"label": str, "color": tuple, "triggered_at": float}, ...]
        """
        h, w = frame.shape[:2]
        now = time.time()
        DURATION = 1.8   # 표시 지속 시간 (초)
        FADE_START = 1.0 # 이 시간 이후부터 서서히 사라짐

        for fb in feedbacks:
            elapsed = now - fb["triggered_at"]
            if elapsed > DURATION:
                continue

            # 투명도 계산 (1.0 → 0.0 으로 페이드)
            if elapsed < FADE_START:
                alpha = 1.0
            else:
                alpha = 1.0 - (elapsed - FADE_START) / (DURATION - FADE_START)

            label = fb["label"]
            color = fb["color"]
            font  = cv2.FONT_HERSHEY_DUPLEX
            scale = 3.2
            thick = 6

            # 텍스트 크기 측정 → 중앙 정렬
            (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)
            tx = (w - tw) // 2
            ty = h // 2 + th // 2

            # 반투명 배경 박스
            pad = 24
            overlay = frame.copy()
            cv2.rectangle(overlay,
                          (tx - pad, ty - th - pad),
                          (tx + tw + pad, ty + baseline + pad),
                          (10, 10, 10), -1)
            cv2.addWeighted(overlay, alpha * 0.55, frame, 1 - alpha * 0.55, 0, frame)

            # 텍스트 그리기 (그림자 + 본문)
            shadow = (20, 20, 20)
            bright = tuple(min(255, int(c * 1.2)) for c in color)
            # 그림자
            cv2.putText(frame, label, (tx + 3, ty + 3), font, scale, shadow,  thick + 2, cv2.LINE_AA)
            # 본문 (alpha 블렌드는 이미 배경으로 처리, 본문은 직접 그림)
            cv2.putText(frame, label, (tx,     ty    ), font, scale, bright,  thick,     cv2.LINE_AA)

        return frame

    # ── 유틸 ──────────────────────────────────────────────
    def _put_text(self, frame, text, pos, scale=0.7, color=(255, 255, 255), thickness=2):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, (pos[0]+1, pos[1]+1), font, scale, self.SHADOW_COLOR, thickness+1, cv2.LINE_AA)
        cv2.putText(frame, text, pos,                  font, scale, color,             thickness,   cv2.LINE_AA)

