# Windows Project: ComfyUI, Hunyuan3D, Unreal World Spawn

이 프로젝트는 Mac Mini가 공유 폴더에 저장한 생성 지시 JSON을 감시하고, ComfyUI로 이미지와 GLB를 만든 뒤, Unreal 에셋 import와 월드 스폰까지 이어줍니다.

## 실행

```powershell
cd windows_world_pipeline
python -m pip install -r requirements.txt
python pipeline_worker.py --config config.example.json
```

한 번만 처리하려면:

```powershell
python pipeline_worker.py --config config.example.json --once
```

특정 파일만 처리하려면:

```powershell
python pipeline_worker.py --config config.example.json --file ..\shared\generation_requests\world_0001.json
```

## Dry Run

`config.example.json`은 기본적으로 `dry_run: true`입니다. 이 상태에서는 ComfyUI, Unreal import, SpawnWorld 없이도 파이프라인 구조를 확인할 수 있습니다.

## 실제 생성 연결

1. `dry_run`을 `false`로 바꿉니다.
2. `comfyui.full_asset_workflow`에 API export workflow JSON 경로를 넣습니다.
3. `prompt_node_id`, `negative_prompt_node_id`를 실제 workflow 노드 ID에 맞춥니다.
4. `unreal_import.enabled`를 `true`로 바꾸고 Unreal Editor 실행 파일과 `.uproject` 경로를 맞춥니다.
5. `unreal.enabled`를 `true`로 바꾸고 Remote Control endpoint/body를 실제 프로젝트에 맞춥니다.

## Generated GLB Import

ComfyUI가 만든 `*.glb`는 Unreal asset으로 import되어야 실제 월드에서 사용할 수 있습니다.

현재 파이프라인은 각 월드마다 고유 경로로 GLB를 import합니다. 이전 월드가 새 mesh로 바뀌지 않게 하기 위해 고정 경로 덮어쓰기는 사용하지 않습니다.

```text
/Game/Generated/world_0007/SM_world_0007_structure
```

`world_layout.json`의 생성 구조물에는 `mesh_asset_path`가 추가됩니다.

```json
{
  "asset_key": "generated_symbolic_structure",
  "mesh_asset_path": "/Game/Generated/world_0007/SM_world_0007_structure.SM_world_0007_structure"
}
```

Unreal `WorldLoader`는 `mesh_asset_path`가 있으면 해당 Static Mesh를 직접 로드하고, 없으면 기존 `asset_key` map을 사용하면 됩니다.

자동 import 스크립트:

```text
unreal_scripts/import_generated_glb.py
```

설정 예시:

```json
{
  "asset_provider": "comfyui",
  "unreal_import": {
    "enabled": true,
    "destination_path": "/Game/Generated/{world_id}",
    "asset_name": "SM_{world_id}_structure",
    "asset_object_path": "/Game/Generated/{world_id}/SM_{world_id}_structure.SM_{world_id}_structure"
  }
}
```

`asset_provider`는 현재 `comfyui`를 지원합니다. Meshy/Tripo를 선택하면 같은 wrapper에 provider 구현을 추가하면 됩니다.

## 월드 JSON 규칙

Unreal은 `world_number * world_space`로 BP_World 자체를 배치합니다. JSON 안의 asset 좌표는 BP_World 내부 로컬 좌표입니다.

Unreal이 읽는 `world_layout.json`에는 다음 최상위 필드를 씁니다.

- `world_id`
- `world_number`
- `world_name`
- `seed`
- `flower_density`
- `flower_type`
- `assets`

앞쪽에는 작은 오브젝트만 놓고, 생성 구조물과 나무, statue, 큰 바위는 back/side 쪽에 둡니다. Pond는 1~3개를 spot 기반으로 분산 배치하고, rock path는 primary pond를 향하되 pond 안으로 들어가지 않습니다.

`fountain`과 `statue`는 필수 에셋입니다. Fountain은 grass line 안쪽에 여유 있게 두고 pond를 가리지 않으며, statue는 가능한 pond 뒤쪽에 둡니다. `rock_l`, `rock_m1`, `rock_m2`는 pond 앞 시야 lane에 들어오지 않게 검사합니다.

전체 두 머신 파이프라인과 남은 작업은 [`../PIPELINE_CHECKLIST.md`](../PIPELINE_CHECKLIST.md)를 기준으로 관리합니다.
