"""
gesture.py — 제스처 시퀀스 감지기

감지 제스처:
  punch      : 한쪽 손목이 조각상 범위에 빠르게 진입  → HITTED  (빨강)
  both_hands : 양 손목이 동시에 분수대 범위에 머무름   → BLOCKED (파랑)
  kick       : 한쪽 발목이 꽃 범위에 빠르게 진입       → THREATED (초록)
"""

from collections import deque
import time

from src.assets import ASSETS, PointAsset, LineAsset

# ── 랜드마크 인덱스 ──────────────────────────────────────────
IDX_LEFT_WRIST  = 15
IDX_RIGHT_WRIST = 16
IDX_LEFT_ANKLE  = 27
IDX_RIGHT_ANKLE = 28

# ── 제스처 정의 ──────────────────────────────────────────────
GESTURES = {
    "punch": {
        "label": "HITTED",
        "color": (0, 0, 220),       # 빨강 (BGR)
    },
    "both_hands": {
        "label": "BLOCKED",
        "color": (200, 80, 0),      # 파랑 (BGR)
    },
    "kick": {
        "label": "THREATED",
        "color": (40, 200, 40),     # 초록 (BGR)
    },
}


class GestureDetector:
    """
    슬라이딩 윈도우 + 상태 머신으로 제스처를 감지합니다.

    각 프레임마다 update(norm_landmarks)를 호출하면
    이번 프레임에 새로 발생한 제스처 id 리스트를 반환합니다.
    """

    BUFFER_SIZE     = 12        # 슬라이딩 윈도우 크기 (프레임)
    VEL_PUNCH       = 0.08      # 펀치 감지 최소 변위 (정규화)
    VEL_KICK        = 0.06      # 킥 감지 최소 변위
    HOLD_BOTH       = 30         # both_hands 확정에 필요한 연속 프레임
    COOLDOWN_SEC    = 2.0       # 한 제스처 재발동 최소 간격 (초)

    def __init__(self):
        self._buf: deque = deque(maxlen=self.BUFFER_SIZE)
        self._cooldown: dict[str, float] = {}
        self._both_hold: int = 0
        self._assets = {a.id: a for a in ASSETS}

    # ── 공개 API ─────────────────────────────────────────────
    def update(self, norm_lms: list[dict] | None) -> list[str]:
        """
        Args:
            norm_lms: get_landmarks_as_dict() 결과에 "nx", "ny" (0~1) 를 추가한 리스트

        Returns:
            이번 프레임에 발동된 제스처 id 목록 (예: ["punch"])
        """
        self._buf.append(norm_lms)
        triggered: list[str] = []
        now = time.time()

        if not norm_lms or len(self._buf) < self.BUFFER_SIZE:
            self._both_hold = 0
            return triggered

        # ── 1. 주먹질 → 조각상 ──────────────────────────────
        if not self._in_cooldown("punch", now):
            statue = self._assets["statue"]
            for idx in [IDX_LEFT_WRIST, IDX_RIGHT_WRIST]:
                cur = self._lm(norm_lms, idx)
                if cur and statue.is_inside(cur["nx"], cur["ny"]):
                    if self._displacement(idx) >= self.VEL_PUNCH:
                        self._fire("punch", now, triggered)
                        break

        # ── 2. 두 손 뻗기 → 분수대 ──────────────────────────
        if not self._in_cooldown("both_hands", now):
            fountain = self._assets["fountain"]
            wl = self._lm(norm_lms, IDX_LEFT_WRIST)
            wr = self._lm(norm_lms, IDX_RIGHT_WRIST)
            both_in = (
                wl is not None and fountain.is_inside(wl["nx"], wl["ny"]) and
                wr is not None and fountain.is_inside(wr["nx"], wr["ny"])
            )
            if both_in:
                self._both_hold += 1
                if self._both_hold >= self.HOLD_BOTH:
                    self._fire("both_hands", now, triggered)
                    self._both_hold = 0
            else:
                self._both_hold = 0
        else:
            self._both_hold = 0

        # ── 3. 발길질 → 꽃 ──────────────────────────────────
        if not self._in_cooldown("kick", now):
            flowers = self._assets["flowers"]
            for idx in [IDX_LEFT_ANKLE, IDX_RIGHT_ANKLE]:
                cur = self._lm(norm_lms, idx)
                if cur and flowers.is_inside(cur["nx"], cur["ny"]):
                    if self._displacement(idx) >= self.VEL_KICK:
                        self._fire("kick", now, triggered)
                        break

        return triggered

    # ── 내부 헬퍼 ────────────────────────────────────────────
    @staticmethod
    def _lm(lms: list[dict], idx: int) -> dict | None:
        return lms[idx] if lms and idx < len(lms) else None

    def _displacement(self, idx: int) -> float:
        """윈도우 첫 프레임 → 현재 프레임의 랜드마크 이동 거리 (정규화)."""
        old_frame = list(self._buf)[0]
        old = self._lm(old_frame, idx) if old_frame else None
        cur = self._lm(list(self._buf)[-1], idx)
        if not old or not cur:
            return 0.0
        return ((cur["nx"] - old["nx"]) ** 2 + (cur["ny"] - old["ny"]) ** 2) ** 0.5

    def _in_cooldown(self, gid: str, now: float) -> bool:
        return now - self._cooldown.get(gid, 0.0) < self.COOLDOWN_SEC

    def _fire(self, gid: str, now: float, out: list):
        self._cooldown[gid] = now
        out.append(gid)
