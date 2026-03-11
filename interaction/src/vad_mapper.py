"""
vad_mapper.py — VAD 값을 에셋별 언리얼 파라미터로 변환

VAD 범위: -1.0 ~ +1.0

분수 (fountain) 속도 매핑:
  VAD = -1  →  VelocityMin=0,   VelocityMax=20
  VAD =  0  →  VelocityMin=200, VelocityMax=230
  VAD = +1  →  VelocityMin=500, VelocityMax=570

꽃 (flower) 블룸 매핑:
  V = -1.0  →  FlowerBloomAmount = -0.6  (꽃이 가장 오므라듦)
  V =  0.0  →  FlowerBloomAmount =  0.2  (중립)
  V = +1.0  →  FlowerBloomAmount =  1.0  (꽃이 활짝 핌)
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

def _map_statue_decay(dominance: float, arousal: float) -> dict:
    """
    Dominance(지배감, -1에 가까울수록 압도당함)일 때 부식이 심해짐.
    D = -1.0 -> Decay = 1.0
    D =  0.0 -> Decay = 0.3
    D = +1.0 -> Decay = 0.0
    """
    if dominance < 0:
        # -1.0 ~ 0.0 구간: 1.0 ~ 0.3
        decay_amount = _lerp(0.3, 1.0, -dominance)
    else:
        # 0.0 ~ 1.0 구간: 0.3 ~ 0.0
        decay_amount = _lerp(0.3, 0.0, dominance)
        
    # Arousal(흥분도, 활력)이 추가 요인으로 결합될 때 부식을 살짝 가속 (최대 +10%)
    # 원치 않으시면 이 줄을 주석 처리하셔도 됩니다.
    if arousal > 0:
        decay_amount += (arousal * 0.1)

    # 0.0 ~ 1.0 양자화(클램핑)
    decay_amount = max(0.0, min(1.0, decay_amount))

    return {
        "DecayAmount": round(decay_amount, 2)
    }


def _map_flower_bloom(valence: float) -> dict:
    """
    Valence(-1~+1)를 꽃 BloomAmount(-0.6~1.0)로 선형 보간.

    V = -1.0  →  FlowerBloomAmount = -0.6  (꽃 오므라듦)
    V =  0.0  →  FlowerBloomAmount =  0.2  (중립)
    V = +1.0  →  FlowerBloomAmount =  1.0  (꽃 활짝)
    """
    # valence -1~+1 를 0~1 로 정규화 후 -0.6~1.0 구간으로 선형 보간
    t = (valence + 1.0) / 2.0          # -1→0, 0→0.5, +1→1
    bloom = _lerp(-0.6, 1.0, t)
    bloom = max(-0.6, min(1.0, bloom))  # 클램핑
    return {"FlowerBloomAmount": round(bloom, 3)}


def map_vad_to_assets(vad: dict) -> dict:
    """
    VAD 딕셔너리를 받아 에셋별 파라미터 딕셔너리를 반환.

    반환 형식:
    {
        "fountain": { "VelocitySpeedMin": float, "VelocitySpeedMax": float },
        "statue":   { "DecayAmount": float },
        "flower":   { "FlowerBloomAmount": float },
    }
    """
    V = vad.get("V", 0.0)
    A = vad.get("A", 0.0)
    D = vad.get("D", 0.0)

    return {
        "fountain": _map_fountain_velocity(A),
        "statue":   _map_statue_decay(D, A),
        "flower":   _map_flower_bloom(V),
    }
