# Remaining Work Checklist

현재 구조는 3D 에셋 생성 없이 진행합니다.

```text
EEG JSON
-> eeg_json watcher
-> LLM instruction JSON
-> world_spawn_json/world_xxx.json
-> planet_spawn/planet_layout.json
-> planet_spawn/planets_layout.json
-> Remote Control SpawnPlanet
-> Unreal world / galaxy spawn
```

## 지금 완료된 것

- [x] 3D asset generation 중심 구조에서 분리
- [x] `world_spawn_json/` 출력 구조 추가
- [x] 월드별 `world_xxx.json` 저장
- [x] `planet_spawn/planet_layout.json` 최신 planet 단일 object 저장
- [x] `planet_spawn/planets_layout.json` 누적 planet 배열 저장
- [x] `person_world_map.json` 사람 이름과 world id 매핑 유지
- [x] `windows_world_pipeline/run_pipeline.py` watcher 파이프라인 추가
- [x] `shared/` 폴더 의존성 제거
- [x] `world_instructions/` instruction 저장 위치 추가
- [x] Remote Control `SpawnPlanet` 호출 단계 추가
- [x] planet 위치를 찌그러진 3D 공간 안에서 랜덤 생성
- [x] 기존 planet과 너무 가깝지 않게 거리 검사
- [x] LLM instruction에 `planet_mesh_type`, `planet_material_type`, `flower_color_group`, `flower_types` 필드 추가

## 1. Mac EEG + LLM

현재 상태:

- [x] `eeg.json` 파일 입력
- [x] WebSocket EEG JSON 입력 스크립트 추가
- [x] LLM instruction JSON 생성
- [x] `person_name` 저장
- [x] `planet_mesh_type` 저장
- [x] `planet_material_type` 저장
- [x] `flower_color_group` 저장
- [x] `flower_types` 저장

남은 작업:

- [ ] `eeg-emotions v3` 실제 WebSocket 메시지 포맷 확인
- [ ] 실제 Mac Mini에서 WebSocket 연결 테스트
- [ ] 사람 이름 입력 방식 확정
- [ ] instruction JSON이 Windows 공유 폴더로 안정적으로 전달되는지 확인
- [ ] watcher가 실제 EEG WebSocket 저장 JSON을 문제없이 감지하는지 확인

## 2. World Spawn JSON

출력 위치:

```text
world_spawn_json/
  world_0001.json
  world_0002.json
```

현재 상태:

- [x] `world_spawn_json/{world_id}.json` 저장
- [x] 기존 `world_layout.json`, `worlds_layout.json` 중심 구조에서 분리
- [x] GLB / mesh path 의존성 제거
- [x] 기본 월드 에셋 배치 유지

남은 작업:

- [ ] Unreal `SpawnWorld`가 `world_spawn_json/{world_id}.json`을 읽도록 경로 연결
- [ ] `world_id` 입력으로 어떤 world JSON을 열지 결정
- [ ] 기존 `world_layout.json`만 읽는 BP/WorldLoader가 있으면 새 구조에 맞게 수정

## 3. Planet Spawn JSON

최신 planet:

```text
planet_spawn/planet_layout.json
```

누적 planet:

```text
planet_spawn/planets_layout.json
```

`planet_layout.json` 형식:

```json
{
  "planet_id": "world_0001",
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

위치 규칙:

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

남은 작업:

- [ ] Unreal `BP_Galaxy`가 `planet_layout.json`을 읽어 최신 planet spawn
- [ ] Unreal `BP_Galaxy`가 `planets_layout.json`을 읽어 전체 planet 복원
- [ ] `mesh_type`별 planet mesh 매핑
- [ ] `material_type`별 material 매핑 확인

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
- [x] flower mesh key를 색상 그룹 기준으로 정리
- [x] world spawn JSON에서 유효한 flower mesh만 저장
- [ ] planet overlap이 Unreal 화면에서 괜찮은지 실제 확인
- [ ] 필요하면 `planet_min_distance` 값 튜닝

## 4. BP_Galaxy 허브

목표 구조:

```text
BP_Galaxy
 ├─ Space
 ├─ CineCamera
 ├─ PlanetSpawnRoot
 ├─ PlanetMap
 └─ Galaxy UI
```

행성용 BP:

```text
BP_WorldPlanet
 ├─ Sphere Mesh
 ├─ WorldID
 ├─ LinkedWorld
 └─ ApproachPoint(Scene Component)
```

카메라 흐름:

```text
Space 상태
-> Galaxy Camera(Home Position)

텍스트 입력
-> world_id 찾기
-> PlanetMap에서 target planet 찾기

Galaxy Camera
-> target planet 근처 ApproachPoint까지 lerp / timeline 이동

Approach Point 도착
-> SetViewTargetWithBlend
-> target BP_World Camera로 전환

월드 체험

Return To Space
-> Galaxy Camera 다시 활성화
-> 저장해둔 Home Transform으로 복귀
```

중요 규칙:

```text
planet 중심까지 이동 X
planet 표면 근처 ApproachPoint까지 이동 O
```

남은 작업:

- [ ] `BP_Galaxy` 생성
- [ ] `BP_WorldPlanet` 생성
- [ ] `PlanetMap: world_id -> BP_WorldPlanet` 관리
- [ ] world_id 텍스트 입력 UI
- [ ] target planet lookup
- [ ] Galaxy Camera home transform BeginPlay 저장
- [ ] Galaxy Camera -> ApproachPoint timeline 이동
- [ ] ApproachPoint 도착 후 `SetViewTargetWithBlend`
- [ ] target BP_World Camera 연결
- [ ] `ReturnToGalaxy()` 구현
- [ ] 복귀 시 Galaxy Camera를 Home Transform으로 lerp

## 5. SpawnPlanet / SpawnWorld 연결

- [ ] `world_spawn_json/{world_id}.json` 읽기
- [ ] `planet_id`와 `world_id`를 동일하게 사용
- [ ] planet 선택 시 연결된 world JSON 찾기
- [x] watcher에서 Remote Control `SpawnPlanet` 호출 코드 추가
- [ ] Unreal에서 실제 `SpawnPlanet` 함수명 / 파라미터 확인
- [ ] Remote Control로 `SpawnWorld` 또는 world 전환 함수 호출
- [ ] 이미 존재하는 world와 새 world가 겹치지 않는지 확인
- [ ] `world_number * world_space` 배치 유지 확인

## 6. final_interaction footprint

아직 설계가 필요한 부분입니다.

추천 저장 위치:

```text
interaction_footprints/world_0001_footprint.json
```

예상 JSON:

```json
{
  "world_id": "world_0001",
  "person_name": "손기훈",
  "base_vad": {"v": 0.2, "a": 0.5, "d": 0.3},
  "final_vad": {"v": 0.7, "a": 0.8, "d": 0.4},
  "samples": [
    {"time": 0.0, "v": 0.2, "a": 0.5, "d": 0.3},
    {"time": 1.0, "v": 0.3, "a": 0.6, "d": 0.35}
  ]
}
```

남은 작업:

- [ ] footprint 저장 포맷 확정
- [ ] `final_interaction`에서 world_id/person_name 입력 방식 확정
- [ ] 일정 간격으로 VAD sample 기록
- [ ] interaction 종료 시 footprint JSON 저장

## 7. Three.js 시각화

후순위입니다.

- [ ] `planet_spawn/planets_layout.json` 읽기
- [ ] `person_world_map.json` 읽기
- [ ] footprint JSON 읽기
- [ ] world별 planet / VAD trajectory 시각화
- [ ] 특정 사람 또는 world_id 선택 UI

## 다음 작업 추천 순서

1. Unreal `BP_Galaxy`에서 `planet_layout.json` 읽어 planet spawn
2. `planets_layout.json`으로 기존 planet들 복원
3. planet 선택 시 `world_spawn_json/{world_id}.json` 찾기
4. `SetViewTargetWithBlend` 카메라 이동/전환 구현
5. Mac EEG WebSocket 실제 연결 확인
6. final_interaction footprint 저장 구현
