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

    # ── 손 관절 시각화 ─────────────────────────────────────
    def draw_hand_landmarks(self, frame: np.ndarray, hand_info: dict | None) -> np.ndarray:
        """
        손 21개 관절과 연결선을 화면에 그립니다.
        hand_info: {"Left": {"landmarks": [(nx,ny)×21], "fist": bool, ...} | None, ...}
        """
        if not hand_info:
            return frame
        h, w = frame.shape[:2]

        # MediaPipe 손 연결선 정의
        connections = [
            # 손바닥
            (0,1),(1,2),(2,3),(3,4),           # 엄지
            (0,5),(5,6),(6,7),(7,8),            # 검지
            (0,9),(9,10),(10,11),(11,12),       # 중지
            (0,13),(13,14),(14,15),(15,16),     # 약지
            (0,17),(17,18),(18,19),(19,20),     # 새끼
            (5,9),(9,13),(13,17),               # 손바닥 가로
        ]

        for side, info in hand_info.items():
            if info is None or "landmarks" not in info:
                continue

            lms = info["landmarks"]  # [(nx, ny) × 21]

            # 주먹=빨강, 열린손=초록, 중간=cyan
            if info.get("fist"):
                joint_color = (60, 60, 255)
                line_color  = (0,  0, 180)
            elif info.get("open"):
                joint_color = (60, 220, 80)
                line_color  = (0, 160, 40)
            else:
                joint_color = (220, 200, 0)
                line_color  = (140, 120, 0)

            # 연결선
            for a, b in connections:
                if a < len(lms) and b < len(lms):
                    p1 = (int(lms[a][0] * w), int(lms[a][1] * h))
                    p2 = (int(lms[b][0] * w), int(lms[b][1] * h))
                    cv2.line(frame, p1, p2, line_color, 2, cv2.LINE_AA)

            # 관절 점
            for i, (nx, ny) in enumerate(lms):
                pt = (int(nx * w), int(ny * h))
                radius = 5 if i == 0 else 3   # 손목은 조금 크게
                cv2.circle(frame, pt, radius, joint_color, -1, cv2.LINE_AA)

        return frame

    # ── 손 상태 표시 ──────────────────────────────────────
    def draw_hand_status(self, frame: np.ndarray, hand_info: dict | None) -> np.ndarray:
        """
        화면 우측 상단에 양손의 주먹/손바닥 상태를 표시합니다.
        hand_info: {"Left": {"fist": bool, "open": bool} | None, "Right": ...}
        """
        if not hand_info:
            return frame
        _, w = frame.shape[:2]
        y = 32
        for side, label_prefix in [("Right", "R"), ("Left", "L")]:
            info = hand_info.get(side)
            if info is None:
                text  = f"{label_prefix}: --"
                color = (100, 100, 100)
            elif info.get("fist"):
                text  = f"{label_prefix}: FIST"
                color = (0, 80, 255)    # 빨강
            elif info.get("open"):
                text  = f"{label_prefix}: OPEN"
                color = (80, 220, 80)   # 초록
            else:
                text  = f"{label_prefix}: ..."
                color = (180, 180, 180)
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            self._put_text(frame, text, (w - tw - 12, y), scale=0.7, color=color)
            y += 30
        return frame

    # ── 에셋 값(HP 바) 표시 ────────────────────────────────
    def draw_asset_values(
        self,
        frame: np.ndarray,
        values: dict[str, float],
        max_value: float = 100.0,
    ) -> np.ndarray:
        """
        각 에셋 옆에 현재 값을 HP 바로 표시합니다.

        Args:
            values:    {"statue": 80.0, "fountain": 100.0, "flowers": 60.0, ...}
            max_value: 최대값 (기본 100)
        """
        h, w = frame.shape[:2]
        BAR_W, BAR_H = 120, 12
        PAD = 6

        for asset in ASSETS:
            val = values.get(asset.id, max_value)
            ratio = max(0.0, min(1.0, val / max_value))

            # 에셋 위치 기반으로 바 위치 결정
            if isinstance(asset, PointAsset):
                cx = int(asset.x * w)
                cy = int(asset.y * h)
                r  = int(asset.radius * min(w, h))
                bx = cx - BAR_W // 2
                by = cy + r + 8
            elif isinstance(asset, LineAsset):
                bx = int(asset.x_start * w) + 8
                by = int(asset.y * h) + int(asset.band / 2 * h) + 8
            else:
                continue

            # 배경 바
            cv2.rectangle(frame, (bx, by), (bx + BAR_W, by + BAR_H),
                          (40, 40, 40), -1)

            # 값에 따라 색상 변화: 초록 → 노랑 → 빨강
            if ratio > 0.6:
                bar_color = (0, 200, 80)
            elif ratio > 0.3:
                bar_color = (0, 180, 220)
            else:
                bar_color = (0, 60, 230)

            filled_w = int(BAR_W * ratio)
            if filled_w > 0:
                cv2.rectangle(frame, (bx, by), (bx + filled_w, by + BAR_H),
                              bar_color, -1)

            # 테두리
            cv2.rectangle(frame, (bx, by), (bx + BAR_W, by + BAR_H),
                          (180, 180, 180), 1)

            # 숫자 표시
            label = f"{int(val)}/{int(max_value)}"
            if val <= 0:
                label = "DESTROYED!"
            self._put_text(frame, label,
                           (bx, by - PAD),
                           scale=0.45, color=bar_color)

        return frame

    # ── 세션 상태 표시 ─────────────────────────────────────
    def draw_session_state(
        self,
        frame: np.ndarray,
        is_active: bool,
        progress: float = 1.0,
    ) -> np.ndarray:
        """
        세션 상태를 화면에 표시합니다.

        Args:
            is_active: 세션 활성 여부
            progress:  1.0=정상, 0.0=타임아웃 직전 (코 안 보임)
        """
        h, w = frame.shape[:2]

        if not is_active:
            # IDLE — 반투명 오버레이 + 안내 메시지
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

            msg1 = "WAITING FOR PLAYER"
            msg2 = "[T] to start session"
            font = cv2.FONT_HERSHEY_DUPLEX
            (w1, h1), _ = cv2.getTextSize(msg1, font, 1.2, 2)
            (w2, h2), _ = cv2.getTextSize(msg2, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(frame, msg1, ((w - w1) // 2, h // 2 - 10),
                        font, 1.2, (200, 200, 200), 2, cv2.LINE_AA)
            cv2.putText(frame, msg2, ((w - w2) // 2, h // 2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (140, 140, 140), 2, cv2.LINE_AA)

        else:
            # ACTIVE — 상단 세션 배지
            badge_text = "SESSION ACTIVE"
            cv2.putText(frame, badge_text, (w // 2 - 90, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 220, 80), 2, cv2.LINE_AA)

            # 타임아웃 진행 바 (코가 안 보일 때만 줄어듦)
            if progress < 1.0:
                bar_w = 200
                bx = w // 2 - bar_w // 2
                by = 36
                cv2.rectangle(frame, (bx, by), (bx + bar_w, by + 6), (40, 40, 40), -1)
                filled = int(bar_w * progress)
                bar_color = (0, 200, 80) if progress > 0.4 else (0, 80, 220)
                if filled > 0:
                    cv2.rectangle(frame, (bx, by), (bx + filled, by + 6), bar_color, -1)
                cv2.rectangle(frame, (bx, by), (bx + bar_w, by + 6), (120, 120, 120), 1)

        return frame

    # ── 유틸 ──────────────────────────────────────────────
    def _put_text(self, frame, text, pos, scale=0.7, color=(255, 255, 255), thickness=2):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, (pos[0]+1, pos[1]+1), font, scale, self.SHADOW_COLOR, thickness+1, cv2.LINE_AA)
        cv2.putText(frame, text, pos,                  font, scale, color,             thickness,   cv2.LINE_AA)
