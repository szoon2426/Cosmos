# Cosmos

Cosmos는 두 개의 큰 프로젝트로 구성됩니다.

```text
1. Exhibition Interaction Project
2. Windows World Pipeline Project
```

## 1. Exhibition Interaction Project

전시 중 관객의 움직임을 읽고 Unreal과 사운드 시스템에 실시간 값을 보내는 프로젝트입니다.

관련 폴더:

```text
final_interaction/
cosmos_sound/
interaction3/
```

### `final_interaction/`

최종 전시에 사용할 웹캠 기반 인터랙션 코드입니다.

역할:

- MediaPipe hand/pose tracking
- 손, 팔, 포즈 기반 인터랙션 분석
- VAD 변화값 계산
- Unreal Remote Control로 `Target V`, `Target A`, `Target D`, speed, density 값 전송
- 특정 thrust 동작에서 `SwitchToWorld(galaxy)` 호출

### `cosmos_sound/`

전시 공간 사운드를 위한 Pure Data 패치와 사운드 파일 폴더입니다.

역할:

- 전시 공간 배경 사운드
- 인터랙션과 연결될 수 있는 Pd 패치 관리

### `interaction3/`

이전 인터랙션 기준 코드입니다.

현재 최종 실행 경로는 아니지만, Unreal에 값을 어떻게 보내는지 참고하기 위해 남겨둔 기준 코드입니다.

## 2. Windows World Pipeline Project

EEG JSON을 받아 감정 월드와 갤럭시 planet을 생성하고 Unreal에 spawn 요청을 보내는 프로젝트입니다.

관련 폴더:

```text
windows_world_pipeline/
eeg_json/
world_instructions/
world_spawn_json/
planet_spawn/
pipeline_state/
person_world_map.json
```

현재 버전에서는 3D asset generation, ComfyUI, GLB import를 사용하지 않습니다.

### Flow

```text
eeg_json/*.json 생성 또는 수정
-> windows_world_pipeline watcher 감지
-> eeg_interpreter가 EEG/VAD 해석
-> world_instructions/world_xxxx.json 저장
-> world_spawn_json/world_xxxx.json 저장
-> planet_spawn/planet_layout.json 덮어쓰기
-> planet_spawn/planets_layout.json 누적 업데이트
-> person_world_map.json 업데이트
-> Unreal Remote Control SpawnPlanet 호출
-> watcher 상태로 복귀
```

### Run

프로젝트 루트에서 실행합니다.

```powershell
python windows_world_pipeline\run_pipeline.py --config windows_world_pipeline\config.example.json
```

이미 `eeg_json/`에 들어있는 샘플까지 처리하려면:

```powershell
python windows_world_pipeline\run_pipeline.py --config windows_world_pipeline\config.example.json --process-existing
```

한 번만 스캔하고 종료하려면:

```powershell
python windows_world_pipeline\run_pipeline.py --config windows_world_pipeline\config.example.json --once --process-existing
```

### `windows_world_pipeline/`

월드 생성 파이프라인의 코드가 모여 있는 폴더입니다.

```text
windows_world_pipeline/
  run_pipeline.py
  config.example.json
  eeg_interpreter/
  world_spawn/
  legacy_asset_generation/
```

### `eeg_json/`

EEG JSON 입력 폴더입니다. 지금은 실제 EEG WebSocket 대신 이 폴더를 감시합니다.

### `world_instructions/`

EEG 해석 결과가 저장되는 폴더입니다.

### `world_spawn_json/`

Unreal world spawn JSON이 월드별로 저장되는 폴더입니다.

### `planet_spawn/`

갤럭시 planet spawn JSON이 저장되는 폴더입니다.

- `planet_layout.json`: 최신 planet 하나
- `planets_layout.json`: 전체 planet 누적 배열

### `person_world_map.json`

사람 이름과 world id를 매핑합니다.

## Documents

- [PIPELINE_CHECKLIST.md](PIPELINE_CHECKLIST.md)
- [REMAINING_WORK_CHECKLIST.md](REMAINING_WORK_CHECKLIST.md)
- [windows_world_pipeline/README.md](windows_world_pipeline/README.md)
