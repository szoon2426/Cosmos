# Windows World Pipeline

`windows_world_pipeline`은 EEG JSON을 감지해서 Unreal이 사용할 world/planet JSON을 만들고, Remote Control API로 `SpawnPlanet`을 호출하는 프로젝트입니다.

현재는 3D asset generation을 사용하지 않습니다. 예전 ComfyUI/GLB 실험 코드는 `legacy_asset_generation/` 안에 보관되어 있습니다.

## Flow

```text
eeg_json/*.json 감지
-> eeg_interpreter instruction 생성
-> world_instructions/world_xxxx.json 저장
-> world_spawn_json/world_xxxx.json 저장
-> planet_spawn/planet_layout.json 덮어쓰기
-> planet_spawn/planets_layout.json 누적 업데이트
-> person_world_map.json 업데이트
-> Unreal Remote Control SpawnPlanet 호출
-> watch 상태로 복귀
```

## Run

프로젝트 루트에서 실행합니다.

```powershell
python windows_world_pipeline\run_pipeline.py --config windows_world_pipeline\config.example.json
```

기본 설정은 실행 전에 이미 `eeg_json/`에 있던 파일을 처리하지 않습니다. 실행 후 새 파일을 넣거나 기존 파일을 수정하면 처리합니다.

이미 있는 샘플까지 처리하려면:

```powershell
python windows_world_pipeline\run_pipeline.py --config windows_world_pipeline\config.example.json --process-existing
```

한 번만 스캔하고 종료하려면:

```powershell
python windows_world_pipeline\run_pipeline.py --config windows_world_pipeline\config.example.json --once --process-existing
```

처리 기록을 초기화하고 다시 테스트하려면:

```powershell
python windows_world_pipeline\run_pipeline.py --config windows_world_pipeline\config.example.json --reset-state --process-existing
```

## Structure

```text
windows_world_pipeline/
  run_pipeline.py
  config.example.json
  eeg_interpreter/
  world_spawn/
  legacy_asset_generation/
```

### `run_pipeline.py`

메인 watcher입니다. `eeg_json/`을 감시하고, 새 EEG JSON이 들어오면 전체 파이프라인을 실행합니다.

### `eeg_interpreter/`

EEG/VAD를 해석해서 world instruction을 만드는 모듈입니다.

현재 `dry_run: true`라 Gemini API 없이 테스트용 해석으로 동작합니다. 실제 LLM을 쓰려면 `eeg_interpreter/config.example.json`에서 `dry_run`을 `false`로 바꾸고 `GEMINI_API_KEY`를 설정합니다.

### `world_spawn/`

world instruction을 받아 `world_spawn_json/`, `planet_spawn/`, `person_world_map.json`을 갱신하는 모듈입니다.

### `legacy_asset_generation/`

이전 ComfyUI, Hunyuan3D, GLB import 실험 코드입니다. 현재 전시 파이프라인의 실행 경로에는 포함되지 않습니다.

## Remote Control

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

Unreal의 실제 함수명이 다르면 `config.example.json`의 `remote_control.function_display_name` 또는 `fallback_function_names`를 수정하면 됩니다.
