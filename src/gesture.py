"""
gesture.py — 제스처 시퀀스 감지기 (Hand Landmarker 연동)

  punch      : 손목이 조각상 범위에 빠르게 진입 + 주먹 쥔 상태  → HITTED
  both_hands : 양 손목이 분수대 범위에 유지 + 양 손바닥 펼친 상태 → BLOCKED
  kick       : 발목이 꽃 범위에 진입 + 아래로 빠르게 내려오는 모션 → THREATED
"""

from collections import deque
import time

from src.assets import ASSETS

# ── 랜드마크 인덱스 (Pose) ───────────────────────────────────
IDX_LEFT_WRIST  = 15
IDX_RIGHT_WRIST = 16
IDX_LEFT_ANKLE  = 27
IDX_RIGHT_ANKLE = 28

# ── 제스처 정의 ──────────────────────────────────────────────
GESTURES = {
    "punch": {
        "label": "HITTED",
        "color": (0, 0, 220),    # 빨강 (BGR)
    },
    "both_hands": {
        "label": "BLOCKED",
        "color": (200, 80, 0),   # 파랑 (BGR)
    },
    "kick": {
        "label": "THREATED",
        "color": (40, 200, 40),  # 초록 (BGR)
    },
}


class GestureDetector:
    """
    슬라이딩 윈도우 + Hand Landmarker로 제스처를 감지합니다.

    update(norm_lms, hand_info) 호출 시 발동된 제스처 id 목록 반환.
    """

    BUFFER_SIZE  = 12     # 슬라이딩 윈도우 (프레임)
    VEL_PUNCH    = 0.08   # 펀치 최소 이동 거리 (정규화)
    VEL_KICK     = 0.07   # 킥 최소 이동 거리
    KICK_DOWN    = 0.04   # 킥 하향 변위 최솟값 (y 증가량, 아래 방향이 양수)
    HOLD_BOTH    = 30     # both_hands 확정 연속 프레임 수
    COOLDOWN_SEC = 2.0    # 재발동 최소 간격 (초)

    def __init__(self):
        self._buf: deque = deque(maxlen=self.BUFFER_SIZE)
        self._cooldown: dict[str, float] = {}
        self._both_hold: int = 0
        self._assets = {a.id: a for a in ASSETS}

    # ── 공개 API ─────────────────────────────────────────────
    def update(
        self,
        norm_lms: list[dict] | None,
        hand_info: dict | None = None,
    ) -> list[str]:
        """
        Args:
            norm_lms:  Pose 랜드마크 (nx, ny 포함)
            hand_info: HandEstimator.process() 결과
                       {"Left": {"fist": bool, "open": bool} | None,
                        "Right": {"fist": bool, "open": bool} | None}

        Returns:
            발동된 제스처 id 목록
        """
        self._buf.append(norm_lms)
        triggered: list[str] = []
        now = time.time()

        if not norm_lms or len(self._buf) < self.BUFFER_SIZE:
            self._both_hold = 0
            return triggered

        hand_info = hand_info or {}

        # 어느 손이든 주먹/열린 상태 확인
        any_fist = any(
            h is not None and h.get("fist", False)
            for h in hand_info.values()
        )
        any_open = any(
            h is not None and h.get("open", False)
            for h in hand_info.values()
        )
        both_open = all(
            hand_info.get(side) is not None and hand_info[side].get("open", False)
            for side in ("Left", "Right")
        )

        # ── 1. 주먹질 → 조각상 ──────────────────────────────
        # 조건: 손목이 조각상 범위에 빠르게 진입 + 주먹 쥔 상태
        if not self._in_cooldown("punch", now) and any_fist:
            statue = self._assets["statue"]
            for idx in [IDX_LEFT_WRIST, IDX_RIGHT_WRIST]:
                cur = self._lm(norm_lms, idx)
                if cur and statue.is_inside(cur["nx"], cur["ny"]):
                    if self._displacement(idx) >= self.VEL_PUNCH:
                        self._fire("punch", now, triggered)
                        break

        # ── 2. 두 손 뻗기 → 분수대 ──────────────────────────
        # 조건: 양 손목이 분수대 범위에 유지 + 양 손바닥 펼친 상태
        if not self._in_cooldown("both_hands", now):
            fountain = self._assets["fountain"]
            wl = self._lm(norm_lms, IDX_LEFT_WRIST)
            wr = self._lm(norm_lms, IDX_RIGHT_WRIST)
            both_in_zone = (
                wl is not None and fountain.is_inside(wl["nx"], wl["ny"]) and
                wr is not None and fountain.is_inside(wr["nx"], wr["ny"])
            )
            if both_in_zone and both_open:
                self._both_hold += 1
                if self._both_hold >= self.HOLD_BOTH:
                    self._fire("both_hands", now, triggered)
                    self._both_hold = 0
            else:
                self._both_hold = 0
        else:
            self._both_hold = 0

        # ── 3. 발길질 → 꽃 ──────────────────────────────────
        # 조건: 발목이 꽃 범위에 진입 + 빠르게 아래로 움직임
        if not self._in_cooldown("kick", now):
            flowers = self._assets["flowers"]
            for idx in [IDX_LEFT_ANKLE, IDX_RIGHT_ANKLE]:
                cur = self._lm(norm_lms, idx)
                if cur and flowers.is_inside(cur["nx"], cur["ny"]):
                    total_disp = self._displacement(idx)
                    down_disp  = self._downward_displacement(idx)
                    if total_disp >= self.VEL_KICK and down_disp >= self.KICK_DOWN:
                        self._fire("kick", now, triggered)
                        break

        return triggered

    # ── 내부 헬퍼 ────────────────────────────────────────────
    @staticmethod
    def _lm(lms: list[dict], idx: int) -> dict | None:
        return lms[idx] if lms and idx < len(lms) else None

    def _oldest(self, idx: int) -> dict | None:
        old_frame = list(self._buf)[0]
        return self._lm(old_frame, idx) if old_frame else None

    def _displacement(self, idx: int) -> float:
        """총 이동 거리 (방향 무관)."""
        old = self._oldest(idx)
        cur = self._lm(list(self._buf)[-1], idx)
        if not old or not cur:
            return 0.0
        return ((cur["nx"] - old["nx"]) ** 2 + (cur["ny"] - old["ny"]) ** 2) ** 0.5

    def _downward_displacement(self, idx: int) -> float:
        """아래 방향(y 증가) 변위만 반환. 위로 올라간 경우 0."""
        old = self._oldest(idx)
        cur = self._lm(list(self._buf)[-1], idx)
        if not old or not cur:
            return 0.0
        dy = cur["ny"] - old["ny"]
        return max(0.0, dy)   # 아래 방향만

    def _in_cooldown(self, gid: str, now: float) -> bool:
        return now - self._cooldown.get(gid, 0.0) < self.COOLDOWN_SEC

    def _fire(self, gid: str, now: float, out: list):
        self._cooldown[gid] = now
        out.append(gid)
