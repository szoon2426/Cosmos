# Cosmos EEG World Pipeline Checklist

남은 실행 작업은 [`REMAINING_WORK_CHECKLIST.md`](REMAINING_WORK_CHECKLIST.md)에서 체크리스트로 추적합니다.

## Goal

EEG data becomes an emotionally symbolic playable Unreal world.

```text
EEG JSON
-> Mac Mini LLM interpretation
-> generation instruction JSON
-> Windows watcher
-> ComfyUI image/RMBG/Hunyuan3D workflow
-> generated asset files
-> Unreal automated GLB import
-> Python layout solver
-> Unreal world_layout.json
-> Unreal Remote Control SpawnWorld
```

## Project Split

### Project 1: Mac Mini

Folder:

```text
mac_eeg_llm/
```

Role:

- Read `eeg.json`.
- Interpret emotion and VAD.
- Generate world concept.
- Generate symbolic structure prompt.
- Generate negative prompt.
- Generate atmosphere keywords.
- Save generation instruction JSON into a shared folder.

Current script:

```text
mac_eeg_llm/generate_world_instruction.py
```

Current output:

```json
{
  "schema_version": 1,
  "world_id": "world_0007",
  "world_number": 7,
  "world_concept": "quiet inner weather held inside a sealed vertical shrine",
  "asset_prompt": "single centered sealed observatory...",
  "negative_prompt": "castle, city...",
  "archetype": "sealed_observatory",
  "atmosphere": ["dreamlike", "misty", "soft", "sacred"],
  "emotional_state": {
    "valence": "mid",
    "arousal": "low",
    "dominance": "mid"
  }
}
```

Recommended next output extension:

```json
{
  "asset_plan": {
    "required": ["fountain", "statue", "generated_symbolic_structure"],
    "pond_count": 2,
    "tree_count": 3,
    "layout_bias": "spread_out",
    "structure_role": "back_center_silhouette"
  }
}
```

The LLM should decide emotional intent and asset intent. It should not generate final coordinates.

### Project 2: Windows Desktop

Folder:

```text
windows_world_pipeline/
```

Role:

- Watch the shared instruction folder.
- Run ComfyUI API workflow.
- Save generated image, transparent image, and GLB.
- Import generated GLB into Unreal as a reusable generated mesh asset.
- Generate Unreal `world_layout.json`.
- Call Unreal Remote Control `SpawnWorld`.

Current script:

```text
windows_world_pipeline/pipeline_worker.py
```

Current Unreal output path:

```text
D:/Unreal_Projects/GG/Saved/world_layout.json
```

## Shared Folder Contract

Mac writes:

```text
shared/generation_requests/world_0007.json
```

Windows reads:

```text
shared/generation_requests/*.json
```

Windows writes generated artifacts:

```text
shared/generated_worlds/world_0007/
  assets/
    world_0007_source.png
    world_0007_transparent.png
    world_0007.glb
  world_0007_manifest.json
```

Windows writes Unreal layout:

```text
D:/Unreal_Projects/GG/Saved/world_layout.json
```

Windows imports each generated GLB into a world-specific Unreal asset path:

```text
/Game/Generated/world_0007/SM_world_0007_structure
```

The generated structure asset in `world_layout.json` should include:

```json
{
  "asset_key": "generated_symbolic_structure",
  "mesh_asset_path": "/Game/Generated/world_0007/SM_world_0007_structure.SM_world_0007_structure"
}
```

## ComfyUI Workflow

The ComfyUI graph should be exported as API workflow JSON.

Expected graph:

```text
Prompt
-> image generation
-> RMBG
-> Hunyuan3D
-> SaveGLB
```

Save API workflow here:

```text
windows_world_pipeline/workflows/full_asset_workflow_api.json
```

Config:

```json
{
  "comfyui": {
    "base_url": "http://127.0.0.1:8188",
    "full_asset_workflow": "./workflows/full_asset_workflow_api.json",
    "prompt_node_id": "6",
    "negative_prompt_node_id": "7"
  }
}
```

Important:

- Use ComfyUI API format, not normal UI workflow JSON.
- Confirm the positive prompt node ID.
- Confirm the negative prompt node ID.
- Confirm SaveGLB appears in ComfyUI history output.

## Layout Solver Rules

Python generates final positions. The LLM does not.

### Core World Rules

- Unreal places the BP_World using `world_number * world_space`.
- Asset locations inside JSON are local BP_World coordinates.
- Do not add world offset to asset locations.
- No world scale.
- No world rotation.
- No camera rotation.
- No offset vector.

Required Unreal JSON:

```json
{
  "world_id": "world_0007",
  "world_number": 7,
  "world_name": "...",
  "seed": 788127,
  "flower_density": 0.74,
  "flower_type": ["sea_thrift", "leadwort"],
  "assets": []
}
```

### Required Assets

Must never be omitted:

- `fountain`
- `statue`
- `generated_symbolic_structure`

### Pond Rules

- Pond count: 1 to 3.
- Ponds are distributed using spots, not pure random.
- Ponds can be near grass edges, including outside the central camera framing.
- Ponds should not go too far back.
- Ponds should not be clustered.
- Pond scale can be non-uniform.
- X and Y scale do not need to match.
- Y-long ponds are allowed.
- Rock path should lead toward a primary pond.
- Rock path must stop before entering pond mesh.
- Pond border rocks are placed using the pond ellipse radius.

### Fountain Rules

- Fountain is mandatory.
- Fountain must stay inside the grass line with margin.
- Fountain must not hide ponds from the camera.
- Fountain candidate positions are tested against pond occlusion lanes.

### Statue Rules

- Statue is mandatory.
- Statue should be placed behind or beside a pond when possible.
- Statue should remain readable.
- Statue should not exceed roughly `X=310`.

### Occlusion Rules

Camera reference:

```text
X=-900
Y=40
Z=160
```

Camera looks generally toward positive X.

Large asset priority:

```text
tree > statue > fountain > rock_l / rock_m / rock_s > glow_sphere
```

Avoid placing larger assets in front of important smaller/flat assets.

Specifically:

- `rock_l`, `rock_m1`, `rock_m2` should not sit in front of ponds.
- `fountain` should not sit in front of ponds.
- `tree` should live mostly in back/side zones.
- If a candidate location blocks pond visibility, reject it.

### Tree Rules

- Tree count can vary.
- Trees should mostly be in back/side zones.
- Preferred X range: `500~700`.
- Trees can be multiple, but must not overlap major assets.

### Flower Rules

Flower types:

- `poppy`
- `bermuda_buttercup`
- `sea_thrift`
- `desert_cotton`
- `leadwort`
- `silver_downy`
- `elderberry`

Rules:

- Randomly mix 2 to 5 flower types.
- Flower density handled by PCG.
- Seed should affect PCG randomness.

## Current Implementation Status

Done:

- Mac EEG instruction generator.
- Gemini API integration skeleton.
- Dry-run emotional interpretation.
- Shared instruction JSON writer.
- Windows shared folder watcher.
- ComfyUI API client skeleton.
- Full ComfyUI asset workflow support.
- Dry-run asset generation.
- Hunyuan3D command hook.
- Unreal automated GLB import hook.
- Unreal Editor Python import script.
- World-specific generated mesh import path.
- `mesh_asset_path` included in generated structure JSON.
- Asset generator wrapper skeleton.
- Unreal world JSON writer.
- Unreal Remote Control call hook.
- Procedural layout solver.
- Pond spot distribution.
- Pond ellipse border rock placement.
- Required fountain/statue handling.
- Pond and large-rock occlusion checks.
- Grass-line margin for fountain.

Needs setup:

- Export actual ComfyUI API workflow JSON.
- Put workflow at `windows_world_pipeline/workflows/full_asset_workflow_api.json`.
- Set correct positive prompt node ID.
- Set correct negative prompt node ID.
- Set `dry_run` to `false`.
- Confirm ComfyUI server URL.
- Confirm Hunyuan3D SaveGLB output is visible through ComfyUI history.
- Compare Meshy / Tripo / ComfyUI output quality with 5 to 10 prompts.
- Choose one first production asset provider.
- Confirm local Unreal Editor executable path.
- Confirm local `.uproject` path.
- Confirm `WorldLoader` supports `mesh_asset_path` for generated symbolic structures.
- Confirm Unreal Remote Control endpoint body.
- Set `unreal.enabled` to `true`.

Needs coding:

- Add `asset_plan` to Mac LLM output.
- Make Windows layout solver read `asset_plan`.
- Implement real Meshy provider if selected.
- Implement real Tripo provider if selected.
- Add validation report after layout generation.
- Add retry if ComfyUI output is missing GLB.
- Add processed/error folder for instruction JSONs.
- Add better logging per world ID.
- Add cached asset fallback for demo timeout.

## Run Commands

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

## Final Production Switches

Mac config:

```json
{
  "llm": {
    "dry_run": false
  }
}
```

Windows config:

```json
{
  "dry_run": false,
  "unreal_import": {
    "enabled": true
  },
  "unreal": {
    "enabled": true
  }
}
```

## Next Best Step

Export the current ComfyUI graph as API workflow JSON, then inspect it to identify:

- positive prompt node ID
- negative prompt node ID
- Save Image output field
- SaveGLB output field

After that, run the Windows pipeline with `dry_run: false` while Unreal spawning remains disabled. Once asset generation works, enable Unreal Remote Control.

Before enabling spawn, verify one generated GLB import:

```text
GLB file exists
-> Unreal import command succeeds
-> /Game/Generated/{world_id}/SM_{world_id}_structure exists
-> world_layout.json includes mesh_asset_path
-> WorldLoader resolves mesh_asset_path before falling back to asset_key
```
