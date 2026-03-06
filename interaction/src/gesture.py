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
IDX_LEFT_WRIST    = 15
IDX_RIGHT_WRIST   = 16
IDX_LEFT_ANKLE    = 27
IDX_RIGHT_ANKLE   = 28
IDX_LEFT_SHOULDER = 11
IDX_RIGHT_SHOULDER= 12
IDX_LEFT_HIP      = 23
IDX_RIGHT_HIP     = 24

# ── 제스처 정의 ──────────────────────────────────────────────
GESTURES = {
    "punch": {
        "label": "HITTED",
        "color": (0, 0, 220),     # 빨강 (BGR)
    },
    "both_hands": {
        "label": "BLOCKED",
        "color": (200, 80, 0),    # 파랑 (BGR)
    },
    "kick": {
        "label": "THREATED",
        "color": (40, 200, 40),   # 초록 (BGR)
    },
    "meditate": {
        "label": "HEALING...",
        "color": (220, 180, 60),  # 하늘색 (BGR)
    },
}


class GestureDetector:
    """
    슬라이딩 윈도우 + Hand Landmarker로 제스처를 감지합니다.

    update(norm_lms, hand_info) 호출 시 발동된 제스처 id 목록 반환.
    """

    BUFFER_SIZE    = 12     # 슬라이딩 윈도우 (프레임)
    VEL_PUNCH      = 0.08   # 펀치 최소 이동 거리 (정규화)
    VEL_KICK       = 0.07   # 킥 최소 이동 거리
    KICK_DOWN      = 0.04   # 킥 하향 변위 최솟값
    HOLD_BOTH      = 30     # both_hands 확정 연속 프레임 수
    COOLDOWN_SEC   = 2.0    # 재발동 최소 간격 (초)
    # 명상 관련
    MEDITATE_STILL = 0.012  # 손목 최대 속도 (이 이하여야 정적 판정)
    MEDITATE_HOLD  = 3.0    # 명상 판정까지 필요한 유지 시간 (초)
    MEDITATE_TICK  = 1.0    # 회복 발동 간격 (초)

    def __init__(self):
        self._buf: deque = deque(maxlen=self.BUFFER_SIZE)
        self._cooldown: dict[str, float] = {}
        self._both_hold: int = 0
        self._assets = {a.id: a for a in ASSETS}
        # 명상 상태
        self._calm_since: float | None = None    # 정적 상태 시작 시각
        self._meditate_last_tick: float = 0.0    # 마지막 회복 발동 시각

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

        # ── 4. 명상/심호흡 → 전체 HP 회복 ───────────────────
        # 조건: 양 손목이 어깨~엉덩이 사이 + 낮은 속도 + 3초 유지
        is_calm = self._is_meditating(norm_lms, hand_info)
        if is_calm:
            if self._calm_since is None:
                self._calm_since = now
            held = now - self._calm_since
            # 3초 이상 유지되고, 마지막 틱에서 1초 이상 지났으면 발동
            if held >= self.MEDITATE_HOLD and (now - self._meditate_last_tick) >= self.MEDITATE_TICK:
                triggered.append("meditate")
                self._meditate_last_tick = now
        else:
            self._calm_since = None

        return triggered

    def _is_meditating(self, norm_lms: list[dict], hand_info: dict | None = None) -> bool:
        """
        심호흡 자세 판단 (T포즈 심호흡):
        1) 팔을 옆으로 쭉 뻗어 손목이 어깨보다 바깥에 위치
        2) 손목이 어깨 높이 근처
        3) 양 손바닥 펼침 (OPEN)
        4) 두 발목이 가까이 붙어 있음
        """
        wl = self._lm(norm_lms, IDX_LEFT_WRIST)
        wr = self._lm(norm_lms, IDX_RIGHT_WRIST)
        ls = self._lm(norm_lms, IDX_LEFT_SHOULDER)
        rs = self._lm(norm_lms, IDX_RIGHT_SHOULDER)
        al = self._lm(norm_lms, IDX_LEFT_ANKLE)
        ar = self._lm(norm_lms, IDX_RIGHT_ANKLE)

        if not all([wl, wr, ls, rs, al, ar]):
            return False

        shoulder_y = (ls["ny"] + rs["ny"]) / 2

        # 1) 팔이 옆으로 펼쳐짐: 왼손목 < 왼어깨, 오른손목 > 오른어깨
        ARM_SPREAD = 0.12
        left_spread  = ls["nx"] - wl["nx"] > ARM_SPREAD
        right_spread = wr["nx"] - rs["nx"] > ARM_SPREAD
        if not (left_spread and right_spread):
            return False

        # 2) 손목이 어깨 높이 근처 (위아래 허용오차 0.18)
        HEIGHT_TOL = 0.18
        if not (abs(wl["ny"] - shoulder_y) < HEIGHT_TOL and
                abs(wr["ny"] - shoulder_y) < HEIGHT_TOL):
            return False

        # 3) 양 손바닥 펼침 (OPEN) — hand_info 있을 때만 체크
        if hand_info:
            l_open = hand_info.get("Left")  is not None and hand_info["Left"].get("open",  False)
            r_open = hand_info.get("Right") is not None and hand_info["Right"].get("open", False)
            if not (l_open and r_open):
                return False

        # 4) 두 발목이 붙어 있음
        ANKLE_CLOSE = 0.10
        if abs(al["nx"] - ar["nx"]) >= ANKLE_CLOSE:
            return False

        return True



    # ── 내부 헬퍼 ────────────────────────────────────────────
    @staticmethod
    def _lm(lms: list[dict], idx: int) -> dict | None:
        return lms[idx] if lms and idx < len(lms) else None

    def _oldest(self, idx: int) -> dict | None:
        old_frame = list(self._buf)[0]
        return self._lm(old_frame, idx) if old_frame else None

    def _displacement(self, idx: int) -> float:
        """총 이동 거리 (방향 무관, Z축 포함)."""
        old = self._oldest(idx)
        cur = self._lm(list(self._buf)[-1], idx)
        if not old or not cur:
            return 0.0
        dz = (cur.get("nz", 0.0) - old.get("nz", 0.0)) * 3.0  # Z는 스케일이 작으므로 가중치 부여
        return ((cur["nx"] - old["nx"]) ** 2 + (cur["ny"] - old["ny"]) ** 2 + dz ** 2) ** 0.5

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
