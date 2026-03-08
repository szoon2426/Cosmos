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
    else:
        t = arousal                 #  0 → 0, +1 → 1
        vel_min = _lerp(200, 500, t)
        vel_max = _lerp(230, 570, t)

    return {
        "VelocitySpeedMin": round(vel_min, 2),
        "VelocitySpeedMax": round(vel_max, 2)
    }

def _map_statue_decay(valence: float, arousal: float) -> dict:
    """
    Valence(부정적, -1에 가까움)일 때 부식이 심해짐.
    V = -1.0 -> Decay = 1.0
    V =  0.0 -> Decay = 0.3
    V = +1.0 -> Decay = 0.0
    """
    if valence < 0:
        # -1.0 ~ 0.0 구간: 1.0 ~ 0.3
        # 인자 t = -valence (음수를 양수로 변환하여 0~1 비율로 사용)
        decay_amount = _lerp(0.3, 1.0, -valence)
    else:
        # 0.0 ~ 1.0 구간: 0.3 ~ 0.0
        # 인자 t = valence
        decay_amount = _lerp(0.3, 0.0, valence)
        
    # Arousal(흥분도, 활력)이 추가 요인으로 결합될 때 부식을 살짝 가속 (최대 +10%)
    # 원치 않으시면 이 줄을 주석 처리하셔도 됩니다.
    if arousal > 0:
        decay_amount += (arousal * 0.1)

    # 0.0 ~ 1.0 양자화(클램핑)
    decay_amount = max(0.0, min(1.0, decay_amount))

    return {
        "DecayAmount": round(decay_amount, 2)
    }


def map_vad_to_assets(vad: dict) -> dict:
    """
    VAD 딕셔너리를 받아 에셋별 파라미터 딕셔너리를 반환.

    반환 형식:
    {
        "fountain": { "VelocityMin": float, "VelocityMax": float },
        "statue": { "DecayAmount": float },
        ...   (향후 에셋 추가 시 여기에 추가)
    }
    """
    V = vad.get("V", 0.0)
    A = vad.get("A", 0.0)

    return {
        "fountain": _map_fountain_velocity(A),
        "statue": _map_statue_decay(V, A)
    }
