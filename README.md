# Cosmos Emotional World Pipeline

Cosmos는 EEG 데이터와 관객 인터랙션을 바탕으로 감정적인 전시 월드를 생성하고 Unreal Engine에서 실행하기 위한 프로젝트입니다.

전체 목표는 다음 흐름을 안정적으로 연결하는 것입니다.

```text
EEG
-> VAD / emotional features
-> LLM interpretation
-> world concept + asset prompt
-> image / 3D asset generation
-> Unreal asset import
-> procedural world layout
-> Unreal Remote Control SpawnWorld
-> playable emotional world
```

## 전체 파이프라인

### 1. Mac Mini: EEG + LLM

Mac Mini에서는 `eeg.json`을 읽고 LLM을 통해 감정 상태를 해석합니다.

역할:

- EEG feature와 VAD 관련 값을 읽기
- 현재 감정 상태 해석
- 월드 컨셉 생성
- 상징적 구조물의 asset prompt 생성
- negative prompt 생성
- 분위기 키워드와 archetype 생성
- Windows Unreal Desktop이 읽을 generation request JSON 저장

현재 구현 위치:

```text
mac_eeg_llm/
```

예상 출력:

```json
{
  "world_id": "world_0007",
  "world_number": 7,
  "world_concept": "quiet inner weather held inside a sealed vertical shrine",
  "asset_prompt": "single centered sealed observatory...",
  "negative_prompt": "castle, city, large environment...",
  "archetype": "sealed_observatory",
  "atmosphere": ["dreamlike", "misty", "soft", "sacred"]
}
```

LLM은 월드의 감정, 컨셉, 프롬프트, 에셋 방향성을 정합니다. 실제 위치 배치와 겹침 방지, pond/rock/statue/fountain/tree 규칙은 Windows 쪽 Python layout solver가 담당합니다.

### 2. Windows Desktop: Asset + Unreal

Windows Desktop에서는 Mac Mini가 공유 폴더에 저장한 JSON을 받아 에셋 생성과 Unreal 배치를 진행합니다.

역할:

- shared folder의 generation request JSON 감시
- ComfyUI 또는 외부 3D provider로 에셋 생성
- PNG / transparent PNG / GLB 저장
- GLB를 Unreal Content Browser에 자동 import
- world layout JSON 생성
- Unreal Remote Control API로 `SpawnWorld` 실행

현재 구현 위치:

```text
windows_world_pipeline/
```

현재 기본 output:

```text
shared/generated_worlds/
D:/Unreal_Projects/GG/Saved/world_layout.json
```

## 월드 생성 규칙

Unreal에서는 `world_number * world_space`로 BP_World 자체를 배치합니다. 그래서 `world_layout.json` 안의 asset 위치는 각 BP_World 내부 local coordinate로 유지합니다.

주요 규칙:

- `fountain`, `statue`, `generated_symbolic_structure`는 필수 에셋
- pond는 1~3개까지 허용
- pond는 너무 뒤로 밀지 않고, 서로 붙지 않게 spot을 나누어 배치
- pond scale은 X/Y가 달라도 되며, Y가 긴 연못도 허용
- rock path는 pond 쪽으로 연결되지만 pond mesh 안으로 들어가지 않음
- pond를 감싸는 돌은 pond 크기와 ellipse radius를 기준으로 계산
- 큰 에셋 뒤에는 pond처럼 중요한 시각 요소를 가리지 않도록 배치
- fountain은 카메라에 잡히도록 grass line 안쪽에 여유 있게 배치
- statue는 pond 뒤쪽이나 옆쪽에 두는 구성을 우선
- large rock이 pond보다 앞에서 pond를 가리지 않도록 제한
- tree는 주로 back/side, edge 쪽에 배치
- 큰 에셋 우선순위는 `tree > statue > fountain > rock_l/m/s > glow_sphere`

## 폴더 소개

### `cosmos_sound/`

전시 공간의 사운드와 관련된 폴더입니다. Pd 패치, 사운드 실험, 전시 공간에서 사용할 오디오 로직이 여기에 포함됩니다.

### `final_interaction/`

최종 전시에 사용할 카메라 인터랙션 런타임입니다. `final_0505` 브랜치에서 가져온 최종 전시용 코드입니다.

코드 기준으로 보면 이 인터랙션은 MediaPipe 기반 hand/pose tracking을 사용합니다. 양손이 화면에 보이고 open 상태가 되면 interaction이 시작되고, 양손 grab 상태에서는 손 사이의 거리와 깊이 변화로 VAD 값을 조정합니다.

조작 방식:

- 양손 open: 인터랙션 시작 및 유지
- 양손 grab: VAD 조정 모드
- 손의 좌우 간격 변화: valence 조정
- 손의 세로 간격 또는 모임 변화: arousal 조정
- 손을 앞으로/뒤로 움직이는 깊이 변화: dominance 조정
- open 상태에서 특정 thrust 움직임: camera switch trigger

Unreal로 보내는 값:

- `Interaction Active`
- `Pointer X`
- `Pointer Y`
- `Target V`
- `Target A`
- `Target D`
- `Density`
- `Decay Amount`
- `Min Speed`
- `Max Speed`
- `Grab Active`
- `Switch To Camera`

전송 방식은 Unreal Remote Control Preset인 `NewRemoteControlPreset`의 property를 `PUT /remote/preset/.../property/...`로 갱신하는 구조입니다.

### `interaction3/`

이전 인터랙션 실험과 Unreal 값 전송 기준을 보존한 폴더입니다.

현재 최종 전시 런타임은 `final_interaction/`을 사용하지만, `interaction3/`는 Unreal에 어떤 값들을 어떤 식으로 보냈는지 참고하기 위한 기준 코드로 남겨둡니다.

특히 VAD 값, gesture state, Unreal Remote Control bridge, Pd bridge 등 이전 구조를 다시 확인해야 할 때 참고합니다.

### `mac_eeg_llm/`

Mac Mini에서 실행할 EEG + LLM instruction generator입니다.

역할:

- `eeg.json` 읽기
- EEG/VAD 기반 감정 상태 해석
- LLM API를 통해 월드 컨셉 생성
- ComfyUI/3D generation에 사용할 asset prompt 생성
- Windows Desktop에서 사용할 generation request JSON 저장

이 프로젝트는 에셋을 직접 생성하지 않습니다. Unreal Desktop 쪽이 사용할 정보를 정리해서 shared folder에 넘기는 역할만 합니다.

### `windows_world_pipeline/`

Unreal Desktop에서 실행할 에셋 생성 및 월드 생성 파이프라인입니다.

역할:

- shared folder 감시
- generation request JSON 읽기
- ComfyUI API workflow 실행
- generated structure PNG/GLB 저장
- Unreal Editor Python으로 GLB import
- world layout JSON 생성
- Unreal Remote Control API로 SpawnWorld 호출

에셋 배치 규칙과 pond/tree/statue/fountain/rock/glow sphere 배치 로직도 이쪽 Python 코드에서 관리합니다.

### `shared/`

Mac Mini와 Windows Desktop 사이에서 JSON과 생성 결과물을 주고받는 공유 폴더입니다.

주요 용도:

- Mac Mini가 만든 generation request JSON 저장
- Windows Desktop이 생성한 world output 저장
- generated asset, manifest, world layout JSON 저장

예상 구조:

```text
shared/
  generation_requests/
  generated_worlds/
```

### `scratch/`

임시 실험용 폴더입니다.

현재는 Pd 사운드 관련 파싱/테스트 스크립트가 남아 있습니다. 명확히 파이프라인에 들어간 코드는 아니므로, 필요한 실험만 남기고 정리해도 되는 후보입니다.

### `PIPELINE_CHECKLIST.md`

전체 파이프라인 구조와 연결 상태를 정리한 문서입니다.

### `REMAINING_WORK_CHECKLIST.md`

남은 작업을 체크리스트로 관리하는 문서입니다. 완료된 작업과 아직 남은 핵심 작업을 구분해서 추적합니다.

## 현재 작업 현황

완료된 것:

- Mac Mini용 EEG + LLM instruction generator 기본 구조
- Gemini API 연결 구조
- dry-run emotion/world instruction 생성
- shared generation request 저장 구조
- Windows watcher / once 처리 구조
- ComfyUI API workflow 연결 구조
- full asset workflow API JSON 연결
- asset generator wrapper 구조
- dry-run asset generation
- procedural world layout solver
- pond/tree/statue/fountain/rock 배치 규칙 반영
- pond occlusion 및 large rock occlusion 보정
- world별 고유 generated mesh path 구조
- Unreal GLB import script 초안
- `final_interaction/` 복원
- 오래된 `interaction`, `interaction2`, `interaction_test` 정리

남은 핵심 작업:

- 실제 ComfyUI workflow 실행 검증
- 실제 GLB 생성 결과 확인
- Unreal Editor 자동 import 실검증
- Unreal `WorldLoader`에서 `mesh_asset_path` 로드 지원
- Remote Control `SpawnWorld` 실호출 검증
- 실패 처리, retry, cached fallback 구현
- 전시/시연용 대기 화면과 안정화

## 다음 우선순위

```text
1. Mock EEG JSON으로 전체 파이프라인 한 번 관통
2. 생성된 GLB가 Unreal에 자동 import되는지 확인
3. WorldLoader에서 mesh_asset_path를 읽어 실제 StaticMesh로 배치
4. SpawnWorld Remote Control 호출 연결
5. 실패해도 전시가 멈추지 않도록 fallback asset/cache 적용
```

## 실행 예시

Mac side dry run:

```powershell
python mac_eeg_llm/generate_world_instruction.py --config mac_eeg_llm/config.example.json
```

Windows side once:

```powershell
python windows_world_pipeline/pipeline_worker.py --config windows_world_pipeline/config.example.json --once
```

Windows side watch:

```powershell
python windows_world_pipeline/pipeline_worker.py --config windows_world_pipeline/config.example.json
```

## 작업 문서

- [PIPELINE_CHECKLIST.md](PIPELINE_CHECKLIST.md)
- [REMAINING_WORK_CHECKLIST.md](REMAINING_WORK_CHECKLIST.md)
