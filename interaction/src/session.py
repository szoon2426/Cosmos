"""
session.py — 세션 관리자

상태:
  IDLE   → 대기 중. T키로 세션 시작 가능.
  ACTIVE → 특정 인물 잠금 중. 제스처 인식 활성화.

얼굴(NOSE) 기반 인물 잠금:
  - 세션 시작 시 코(NOSE) 위치를 기억
  - 매 프레임마다 코 위치가 LOCK_RADIUS 이상 튀면 다른 사람으로 간주 → 해당 프레임 무시
  - 코가 IDLE_TIMEOUT 초 이상 사라지면 세션 종료 (IDLE로 복귀)
"""

import time

# Pose 랜드마크에서 NOSE 인덱스
IDX_NOSE = 0


class SessionManager:
    LOCK_RADIUS  = 0.28   # 코가 이 이상 이동하면 다른 사람으로 판단
    IDLE_TIMEOUT = 5.0    # 코가 N초 이상 없으면 세션 종료 (초)
    SMOOTH       = 0.15   # 잠금 위치 업데이트 스무딩 계수 (0=고정, 1=즉시)

    def __init__(self):
        self._active       = False
        self._locked_nx    = None    # 잠금된 코 x (0~1)
        self._locked_ny    = None    # 잠금된 코 y (0~1)
        self._last_seen    = None    # 마지막으로 코가 보인 시각

    # ── 공개 API ─────────────────────────────────────────────
    @property
    def is_active(self) -> bool:
        return self._active

    def start(self, norm_lms: list[dict] | None):
        """T키 입력 시 호출. 현재 감지된 코 위치로 세션 시작."""
        nose = self._get_nose(norm_lms)
        if nose is None:
            print("[Session] 코(얼굴)가 감지되지 않아 세션 시작 불가")
            return
        self._locked_nx = nose[0]
        self._locked_ny = nose[1]
        self._last_seen = time.time()
        self._active    = True
        print(f"[Session] 세션 시작 — 코 위치 잠금 ({self._locked_nx:.2f}, {self._locked_ny:.2f})")

    def end(self):
        """세션 수동 종료."""
        self._active    = False
        self._locked_nx = None
        self._locked_ny = None
        self._last_seen = None
        print("[Session] 세션 종료")

    def update(self, norm_lms: list[dict] | None) -> list[dict] | None:
        """
        매 프레임 호출. 세션이 활성화된 경우 잠금된 인물의 랜드마크를 반환.
        세션이 비활성이면 None 반환.
        타임아웃 시 자동 종료.

        Returns:
            유효한 norm_lms (동일 인물), 또는 None (무시/세션 없음)
        """
        if not self._active:
            return None

        now  = time.time()
        nose = self._get_nose(norm_lms)

        if nose is None:
            # 코가 안 보임 — 타임아웃 체크
            if self._last_seen and now - self._last_seen >= self.IDLE_TIMEOUT:
                print(f"[Session] 코 미감지 {self.IDLE_TIMEOUT:.0f}초 초과 → 세션 종료")
                self.end()
            return None

        # 코 위치가 잠금 위치에서 너무 멀면 다른 사람으로 판단
        dist = ((nose[0] - self._locked_nx) ** 2 + (nose[1] - self._locked_ny) ** 2) ** 0.5
        if dist > self.LOCK_RADIUS:
            # 이 프레임의 랜드마크 무시 (잠금 위치 유지)
            return None

        # 동일 인물 → 잠금 위치 부드럽게 업데이트 (이동 허용)
        self._locked_nx = (1 - self.SMOOTH) * self._locked_nx + self.SMOOTH * nose[0]
        self._locked_ny = (1 - self.SMOOTH) * self._locked_ny + self.SMOOTH * nose[1]
        self._last_seen = now

        return norm_lms

    def progress(self) -> float:
        """
        IDLE 복귀까지 남은 시간 비율 (0~1).
        1.0이면 막 감지됨, 0.0이면 타임아웃 직전.
        세션 비활성이면 1.0 반환.
        """
        if not self._active or self._last_seen is None:
            return 1.0
        elapsed = time.time() - self._last_seen
        return max(0.0, 1.0 - elapsed / self.IDLE_TIMEOUT)

    # ── 내부 헬퍼 ────────────────────────────────────────────
    @staticmethod
    def _get_nose(norm_lms: list[dict] | None) -> tuple[float, float] | None:
        if not norm_lms or IDX_NOSE >= len(norm_lms):
            return None
        n = norm_lms[IDX_NOSE]
        return (n["nx"], n["ny"])
