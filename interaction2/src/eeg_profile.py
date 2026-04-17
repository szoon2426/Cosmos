from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class EmotionProfile:
    baseline_v: float = 0.0
    baseline_a: float = 0.0
    baseline_d: float = 0.0

    v_gain: float = 1.0
    a_gain: float = 1.0
    d_gain: float = 1.0

    v_return_rate: float = 0.45
    a_return_rate: float = 0.55
    d_return_rate: float = 0.35

    v_min: float = -1.0
    v_max: float = 1.0
    a_min: float = -1.0
    a_max: float = 1.0
    d_min: float = -1.0
    d_max: float = 1.0

    @classmethod
    def from_json(cls, path: str | Path) -> "EmotionProfile":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

