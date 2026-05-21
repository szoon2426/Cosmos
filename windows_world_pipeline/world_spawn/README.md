# Cosmos World Spawn

`world_spawn`은 Mac Mini 쪽 LLM instruction JSON을 받아 Unreal이 사용할 월드 스폰 JSON과 갤럭시 플래닛 JSON을 저장합니다.

3D 에셋 생성이나 GLB import는 여기서 하지 않습니다.

## 출력 구조

```text
world_spawn_json/
  world_001.json
  world_002.json

planet_spawn/
  planet_layout.json
  planets_layout.json

person_world_map.json
```

## world_spawn_json

`world_spawn_json/`에는 월드별 JSON을 개별 파일로 저장합니다.

```text
world_spawn_json/world_001.json
```

파일 이름은 `world_id`와 동일합니다.

## planet_layout.json

`planet_spawn/planet_layout.json`은 가장 최근 생성된 planet 하나만 저장합니다.

배열이 아니라 단일 object입니다.

```json
{
  "planet_id": "world_001",
  "location": {
    "x": -1200.0,
    "y": 1800.0,
    "z": -650.0
  },
  "scale": 0.4,
  "mesh_type": "basic_planet",
  "material_type": "orange"
}
```

## planets_layout.json

`planet_spawn/planets_layout.json`은 모든 planet을 누적 저장합니다.

```json
{
  "planets": [
    {
      "planet_id": "world_001",
      "location": {
        "x": -1200.0,
        "y": 1800.0,
        "z": -650.0
      },
      "scale": 0.4,
      "mesh_type": "basic_planet",
      "material_type": "orange"
    }
  ]
}
```

`world_json` 필드는 쓰지 않습니다. `planet_id`가 world id와 동일하므로 Unreal 쪽에서 필요하면 `planet_id + ".json"`으로 world spawn JSON을 찾으면 됩니다.

## Planet Location Rule

플래닛은 정육면체가 아니라 앞쪽이 좁고 뒤쪽이 넓은 찌그러진 3D 공간 안에서 생성됩니다.

front plane:

```text
x = -2000
y = -1550 ~ 1550
z = -900 ~ 900
```

back plane:

```text
x = 300
y = -3700 ~ 3700
z = -2100 ~ 2100
```

생성 방식:

```text
x를 -2000 ~ 300 사이에서 랜덤 선택
t = (x - front_x) / (back_x - front_x)
y_limit = lerp(1550, 3700, t)
z_limit = lerp(900, 2100, t)
y = random(-y_limit, y_limit)
z = random(-z_limit, z_limit)
```

기존 planet과 너무 가까우면 다시 뽑습니다.

## LLM이 정하는 값

LLM instruction JSON에서 아래 값을 받을 수 있습니다.

```json
{
  "planet_mesh_type": "basic_planet",
  "planet_material_type": "orange"
}
```

없으면 Python이 fallback 값을 고릅니다.

사용 가능한 `material_type`은 Unreal의 Planet Material Map 키와 동일하게 아래 8개입니다.

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

사용 가능한 flower mesh는 색 그룹 기준으로 아래 값을 사용합니다.

```text
purple:
silver_downy_1, pentas_1, pentas_2, leadwort_1, leadwort_2, bigleaf

red:
silver_downy_2, bougainv_1, bougainv_2, dianthus_1, dianthus_2, daisy_1, daisy_2

yellow:
campion_1, campion_2, gazania_1, gazania_2, crownbeard_1, crownbeard_2, windflower_1, windflower_2

LLM은 `flower_color_group`과 `flower_types`를 제안할 수 있고, Python은 유효한 mesh key만 world spawn JSON에 저장합니다.

## Fountain / Statue Mesh

`fountain`과 `statue` asset에는 `mesh` 필드가 추가됩니다.

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

Statue는 EEG JSON의 `world_style` 중 가장 높은 값을 기준으로 선택합니다.

```text
quietness -> inner_quietness
tempo -> neural_tempo
clarity -> resonance_clarity
drift / bandwidth -> arousal_drift
abs(frontal_tilt) -> frontal_tilt
```

Fountain은 VAD와 EEG feature의 relaxation, engagement, alpha, beta, theta, gamma 값을 기준으로 선택합니다.

Statue 위치와 스케일:

```text
x range: 30 ~ 310
x = 30  -> scale 1.25
x = 310 -> scale 1.35
rotation yaw -> camera reference (-900, 40)을 기준으로 프로젝트 회전 규칙에 맞춰 계산
0 = 정면
90 = 좌측
180 = 뒤
270 = 우측
오른쪽 조각상 -> 0 ~ 90도 사이
```

## 실행

```powershell
python windows_world_pipeline\world_spawn\build_world_spawn.py --config windows_world_pipeline\world_spawn\config.example.json --instruction world_instructions\world_001.json
```

사람 이름을 덮어쓰고 싶으면:

```powershell
python windows_world_pipeline\world_spawn\build_world_spawn.py --config windows_world_pipeline\world_spawn\config.example.json --instruction world_instructions\world_001.json --person-name "손기훈"
```
