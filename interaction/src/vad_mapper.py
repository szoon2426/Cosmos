"""
vad_mapper.py — VAD 값을 에셋별 언리얼 파라미터로 변환

VAD 범위: -1.0 ~ +1.0

분수 (fountain) 속도 매핑:
  VAD = -1  →  VelocityMin=0,   VelocityMax=20
  VAD =  0  →  VelocityMin=200, VelocityMax=230
  VAD = +1  →  VelocityMin=500, VelocityMax=570
"""

def _lerp(a: float, b: float, t: float) -> float:
    """a에서 b로 t(0~1) 비율만큼 선형 보간"""
    return a + (b - a) * t


def _map_fountain_velocity(arousal: float) -> dict:
    """
    Arousal 값(-1~1)을 분수 Niagara velocity 파라미터로 변환.
    음수 구간(-1~0)과 양수 구간(0~+1)을 각각 선형 보간.
    """
    if arousal < 0:
        t = arousal + 1.0           # -1 → 0,  0 → 1
        vel_min = _lerp(0,   200, t)
        vel_max = _lerp(20,  230, t)
        force_z = _lerp(0,   500, t)  # Linear Force Z축
    else:
        t = arousal                 #  0 → 0, +1 → 1
        vel_min = _lerp(200, 500, t)
        vel_max = _lerp(230, 570, t)
        force_z = _lerp(500, 1500, t) # Linear Force Z축

    return {
        "VelocitySpeedMin": round(vel_min, 2),
        "VelocitySpeedMax": round(vel_max, 2),
        "LinearForceX": 0.0,
        "LinearForceY": 0.0,
        "LinearForceZ": round(force_z, 2),
    }


def map_vad_to_assets(vad: dict) -> dict:
    """
    VAD 딕셔너리를 받아 에셋별 파라미터 딕셔너리를 반환.

    반환 형식:
    {
        "fountain": { "VelocityMin": float, "VelocityMax": float },
        ...   (향후 에셋 추가 시 여기에 추가)
    }
    """
    A = vad.get("A", 0.0)

    return {
        "fountain": _map_fountain_velocity(A),
    }
