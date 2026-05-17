# Remaining Work Checklist

이 문서는 사용자가 정리한 남은 작업 목록을 기준으로, 현재 완료/미완료 상태를 추적합니다.

## 핵심 우선순위

- [x] Mock EEG JSON으로 전체 파이프라인 먼저 뚫기
- [ ] Meshy/Tripo 중 하나 선택
- [ ] Unreal import + spawn 자동화
- [ ] 실패해도 시연이 안 죽게 fallback 만들기

## 1. Meshy / Tripo 테스트

- [ ] 같은 프롬프트로 Meshy 5~10개 생성
- [ ] 같은 프롬프트로 Tripo 5~10개 생성
- [ ] GLB 품질 비교
- [ ] 텍스처 품질 비교
- [ ] Unreal import 가능 여부 확인
- [ ] 하루 안에 1개 provider 선택

현재 상태:

- 아직 실제 Meshy/Tripo API 테스트는 하지 않음.
- ComfyUI API workflow 연결 준비는 완료.
- OpenAI 이미지 API smoke test는 결제 한도 문제로 폐기.

## 2. Asset Generator Wrapper

- [x] `generate_asset(prompt, provider)` 형태의 wrapper 뼈대 추가
- [x] `asset_provider` 설정 추가
- [x] 현재 provider로 `comfyui` 연결
- [ ] Meshy provider 구현
- [ ] Tripo provider 구현
- [ ] provider별 실패/성공 metadata 통일

관련 파일:

```text
windows_world_pipeline/asset_generator.py
windows_world_pipeline/pipeline_worker.py
windows_world_pipeline/config.example.json
```

## 3. 파일 저장 규칙

- [x] `world_id` 기반 저장 규칙 사용
- [x] `world_layout.json` 생성
- [x] `asset.glb` 역할의 world별 GLB 저장
- [x] `preview/source image` 저장 자리 있음
- [x] `metadata/manifest.json` 저장
- [ ] `session_id` 개념 추가
- [ ] processed/error/archive 폴더 규칙 추가

현재 저장 구조:

```text
shared/generated_worlds/world_0007/
  assets/
    world_0007_source.png
    world_0007_transparent.png
    world_0007.glb
  world_0007_manifest.json
```

## 4. Unreal 자동 Import

- [x] Unreal GLB import hook 추가
- [x] Unreal Editor Python import script 추가
- [x] world별 고유 Unreal asset path 설계
- [x] `mesh_asset_path`를 generated structure JSON에 추가
- [ ] 실제 Unreal Editor에서 import command 성공 확인
- [ ] material/texture 연결 확인
- [ ] WorldLoader가 `mesh_asset_path`를 직접 로드하도록 수정
- [ ] GLB/FBX 감지 일반화

관련 파일:

```text
windows_world_pipeline/unreal_scripts/import_generated_glb.py
```

현재 intended path:

```text
/Game/Generated/{world_id}/SM_{world_id}_structure
```

## 5. Unreal Spawn Logic

- [x] Python layout solver에서 zone 성격을 좌표로 변환
- [x] scale/yaw 적용
- [x] pond 규칙 반영
- [x] tree 규칙 반영
- [x] statue 필수 배치 반영
- [x] fountain 필수 배치 반영
- [x] pond occlusion / large rock occlusion 일부 반영
- [ ] Unreal WorldLoader에서 `mesh_asset_path` spawn 지원
- [ ] Remote Control `SpawnWorld` endpoint 실제 호출 확인
- [ ] Unreal 쪽 PCG seed 적용 확인

## 6. 전체 파이프라인 테스트

- [x] mock EEG JSON 생성
- [x] Mac LLM instruction JSON 생성
- [x] Windows watcher/once 처리 구조 생성
- [x] dry-run 에셋 생성
- [x] dry-run world layout 생성
- [ ] 실제 LLM API 출력 확인
- [ ] 실제 ComfyUI workflow 실행 확인
- [ ] 실제 GLB 생성 확인
- [ ] 실제 Unreal import 확인
- [ ] 실제 SpawnWorld 확인

## 7. 실패 처리

- [ ] LLM JSON 깨짐 검증/복구
- [x] dry-run fallback asset 생성
- [x] asset generation result metadata 추가
- [x] fallback config 자리 추가
- [ ] 에셋 생성 실패 시 cached asset 사용
- [ ] 다운로드 실패 retry
- [ ] Unreal import 실패 처리
- [ ] 실패 instruction을 error 폴더로 이동
- [ ] 실패 시 기본 월드 표시

## 8. 전시/시연용 안정화

- [ ] 생성 대기 중 보여줄 화면
- [x] 이전 월드 유지가 가능한 world별 mesh path 설계
- [ ] 실패 시 기본 월드 표시
- [ ] 너무 오래 걸리면 cached asset 사용
- [ ] timeout 기준 정하기
- [ ] operator용 runbook 작성

## 지금 남은 진짜 핵심

- [ ] 실제 3D provider 결정
- [ ] provider API로 GLB 생성 성공
- [ ] Unreal import 실제 성공
- [ ] WorldLoader `mesh_asset_path` 지원
- [ ] SpawnWorld 실제 호출
- [ ] fallback/cached asset으로 실패 방지

## 다음 추천 작업 순서

1. `WorldLoader`에 `mesh_asset_path` 지원 추가
2. ComfyUI API workflow를 `dry_run: false`로 한 번 실행
3. 생성된 GLB를 Unreal 자동 import
4. SpawnWorld 호출
5. 실패 처리와 cached fallback 추가
