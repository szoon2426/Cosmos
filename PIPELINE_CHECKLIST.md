# Cosmos EEG World Pipeline Checklist

현재 파이프라인은 3D asset generation 없이 동작합니다.

```text
eeg_json/*.json
-> windows_world_pipeline watcher
-> eeg_interpreter instruction
-> world_instructions/world_xxxx.json
-> world_spawn_json/world_xxxx.json
-> planet_spawn/planet_layout.json
-> planet_spawn/planets_layout.json
-> person_world_map.json
-> Unreal Remote Control SpawnPlanet
```

## 실행 흐름

- [x] `eeg_json/` 폴더 생성
- [x] 샘플 EEG JSON 7명분 생성
- [x] `windows_world_pipeline/run_pipeline.py` watcher 추가
- [x] 기존 파일은 기본적으로 skip하고 새 파일 또는 수정 파일만 처리
- [x] 처리 상태를 `pipeline_state/eeg_pipeline_state.json`에 저장
- [x] `world_instructions/`에 LLM instruction 저장
- [x] `world_spawn_json/`에 월드별 spawn JSON 저장
- [x] `planet_spawn/planet_layout.json` 최신 planet 저장
- [x] `planet_spawn/planets_layout.json` 전체 planet 누적
- [x] `person_world_map.json` 사람 이름과 world id 매핑
- [x] Remote Control `SpawnPlanet` 호출 코드 추가

## 사용자 실행 명령

기본 watcher:

```powershell
python windows_world_pipeline\run_pipeline.py --config windows_world_pipeline\config.example.json
```

이미 있는 샘플까지 처리:

```powershell
python windows_world_pipeline\run_pipeline.py --config windows_world_pipeline\config.example.json --process-existing
```

한 번만 테스트:

```powershell
python windows_world_pipeline\run_pipeline.py --config windows_world_pipeline\config.example.json --once --process-existing
```

처리 기록 초기화:

```powershell
python windows_world_pipeline\run_pipeline.py --config windows_world_pipeline\config.example.json --reset-state --process-existing
```

## EEG JSON 입력 형식

```json
{
  "timestamp": "2026-05-20T15:00:00+09:00",
  "person_name": "person_001",
  "features": {
    "alpha": 0.64,
    "beta": 0.28,
    "theta": 0.52,
    "gamma": 0.19,
    "engagement": 0.36,
    "relaxation": 0.78
  },
  "vad": {
    "valence": 0.62,
    "arousal": 0.24,
    "dominance": 0.34
  }
}
```

## LLM Instruction 출력

```json
{
  "world_id": "world_0015",
  "world_number": 15,
  "person_name": "person_001",
  "world_concept": "quiet inner weather held inside a sealed vertical shrine",
  "planet_mesh_type": "smooth_planet",
  "planet_material_type": "blue",
  "flower_color_group": "purple",
  "flower_types": ["leadwort_1", "pentas_1", "silver_downy_1"],
  "emotional_state": {"valence": "mid", "arousal": "low", "dominance": "mid"}
}
```

## Planet 규칙

사용 가능한 `mesh_type`:

```text
basic_planet
sharp_planet
smooth_planet
complicated_planet
simple_planet
```

사용 가능한 `material_type`:

```text
yellow
blue
green
orange
purple
gold
red
pink
```

위치 공간:

```text
front plane:
x = -2000
y = -1550 ~ 1550
z = -900 ~ 900

back plane:
x = 300
y = -3700 ~ 3700
z = -2100 ~ 2100
```

생성 방식:

```text
x = random(-2000, 300)
t = (x - front_x) / (back_x - front_x)
y_limit = lerp(1550, 3700, t)
z_limit = lerp(900, 2100, t)
y = random(-y_limit, y_limit)
z = random(-z_limit, z_limit)
```

기존 planet과 너무 가까우면 다시 뽑습니다.

## Flower 규칙

Purple:

```text
silver_downy_1
pentas_1
pentas_2
leadwort_1
leadwort_2
bigleaf
```

Red:

```text
silver_downy_2
bougainv_1
bougainv_2
dianthus_1
dianthus_2
daisy_1
daisy_2
```

Yellow:

```text
campion_1
campion_2
gazania_1
gazania_2
crownbeard_1
crownbeard_2
windflower_1
windflower_2
```

LLM이 `flower_color_group`과 `flower_types`를 제안하고, Python이 유효한 mesh key만 저장합니다.

## Fountain / Statue Mesh 규칙

`fountain`과 `statue` asset은 공통 필드에 더해 `mesh` 필드를 가집니다.

```json
{
  "asset_key": "fountain",
  "location": [-430, -260, 0],
  "rotation": [0.0, 0.0, 145.0],
  "scale": [1.05, 1.05, 1.05],
  "mesh": "rock_octagon"
}
```

Fountain mesh:

```text
rock_octagon
plate_round
pillar_round
basic
```

Statue mesh:

```text
inner_quietness
neural_tempo
resonance_clarity
arousal_drift
frontal_tilt
```

Statue는 EEG JSON의 `world_style` 값 중 가장 강한 성향을 기준으로 고릅니다.

```json
{
  "world_style": {
    "quietness": 0.78,
    "tempo": 0.42,
    "clarity": 0.66,
    "bandwidth": 0.81,
    "drift": 0.35,
    "frontal_tilt": -0.24,
    "texture": 0.58,
    "ecology": 0.73
  }
}
```

기본 매핑:

```text
quietness -> inner_quietness
tempo -> neural_tempo
clarity -> resonance_clarity
drift / bandwidth -> arousal_drift
abs(frontal_tilt) -> frontal_tilt
```

Fountain은 VAD와 EEG feature를 기준으로 고릅니다.

Statue placement:

```text
x range: 30 ~ 310
x = 30  -> scale 1.25
x = 310 -> scale 1.35
rotation yaw -> project convention using camera reference (-900, 40)
0 = front
90 = left
180 = back
270 = right
right-side statue -> yaw between 0 and 90
```

## Unreal Remote Control

기본 preset:

```text
RCP_WorldVariable
```

기본 함수 후보:

```text
Spawn Planet
SpawnPlanet
Spawn Planets
SpawnPlanets
```

남은 확인:

- [ ] Unreal Remote Control Preset에 실제 `SpawnPlanet` 함수 노출
- [ ] 함수 이름이 다르면 `windows_world_pipeline/config.example.json` 수정
- [ ] `SpawnPlanet`이 `planet_spawn/planet_layout.json`을 읽는지 확인
- [ ] `BP_Galaxy`가 `planets_layout.json`으로 기존 planet 복원하는지 확인
- [ ] planet 선택 시 `world_spawn_json/{world_id}.json`을 찾는지 확인

## 남은 작업

- [ ] 실제 EEG WebSocket에서 저장되는 JSON 포맷 확인
- [ ] WebSocket 수신 파일을 `eeg_json/`에 저장하는 어댑터 작성
- [ ] Gemini `dry_run: false` 실제 호출 테스트
- [ ] Unreal `SpawnPlanet` Remote Control 실호출 테스트
- [ ] `BP_Galaxy` planet spawn / restore 구현
- [ ] planet 접근 카메라 이동 구현
- [ ] world 체험 후 galaxy 복귀 구현
- [ ] final_interaction footprint 저장 방식 확정
