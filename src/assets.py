"""
assets.py — 공간 에셋 정의 및 근접 감지

좌표계: 정규화 좌표 (0.0~1.0), 화면 좌상단이 (0, 0), 우하단이 (1, 1)

에셋 타입:
  - "point" : 원형 범위를 가진 단일 지점 (조각상, 분수대)
  - "line"  : 수평선으로 표현되는 구간 (꽃밭)
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class PointAsset:
    """단일 지점 에셋 (조각상, 분수대 등)."""
    id: str
    name: str
    x: float          # 정규화 x (0~1)
    y: float          # 정규화 y (0~1)
    radius: float     # 감지 범위 (정규화, 반지름)
    color: tuple      # BGR
    type: Literal["point"] = "point"

    def is_inside(self, nx: float, ny: float) -> bool:
        """정규화 좌표 (nx, ny) 가 감지 범위 안인지 확인."""
        return ((nx - self.x) ** 2 + (ny - self.y) ** 2) ** 0.5 <= self.radius


@dataclass
class LineAsset:
    """수평선으로 표현되는 구간 에셋 (꽃밭 등)."""
    id: str
    name: str
    x_start: float    # 시작 x (0~1)
    x_end: float      # 끝 x (0~1)
    y: float          # 중심 y (0~1)
    band: float       # 수직 감지 범위 (±band/2)
    color: tuple      # BGR
    type: Literal["line"] = "line"

    def is_inside(self, nx: float, ny: float) -> bool:
        """정규화 좌표 (nx, ny) 가 감지 범위 안인지 확인."""
        in_x = self.x_start <= nx <= self.x_end
        in_y = abs(ny - self.y) <= self.band / 2
        return in_x and in_y


# ─────────────────────────────────────────
# 에셋 정의 — 좌표를 여기서 조정하세요
# ─────────────────────────────────────────
ASSETS: list[PointAsset | LineAsset] = [
    PointAsset(
        id="statue",
        name="statue",
        x=0.69,
        y=0.48,
        radius=0.12,
        color=(80, 200, 255),   # 노란 계열 (우측)
    ),
    PointAsset(
        id="fountain",
        name="fountain",
        x=0.40,
        y=0.60,
        radius=0.12,
        color=(255, 200, 60),   # cyan 계열 (좌측)
    ),
    LineAsset(
        id="flowers",
        name="flowers",
        x_start=0.05,
        x_end=0.95,
        y=0.85,
        band=0.15,
        color=(130, 80, 255),   # 분홍/보라 (하단 전체)
    ),
]


def check_landmarks(landmarks: list[dict] | None) -> dict[str, bool]:
    """
    현재 프레임 랜드마크 중 하나라도 각 에셋 범위 안에 있으면 True.

    Args:
        landmarks: pose.py 의 get_landmarks_as_dict() 결과 (x, y 는 픽셀값)
                   → 내부에서 정규화하지 않으므로, 호출 전에 정규화된 값으로 넘겨야 함.
                   여기서는 normalized_landmarks 리스트를 받습니다.

    Returns:
        {"statue": bool, "fountain": bool, "flowers": bool, ...}
    """
    result = {a.id: False for a in ASSETS}
    if not landmarks:
        return result
    for a in ASSETS:
        for lm in landmarks:
            if a.is_inside(lm["nx"], lm["ny"]):
                result[a.id] = True
                break
    return result
