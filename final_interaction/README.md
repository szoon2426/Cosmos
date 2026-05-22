# final_interaction 실행 방법

`final_interaction`은 최종 전시에 사용할 웹캠 기반 인터랙션 런타임입니다.
MediaPipe로 손과 상체 포즈를 추적하고, 계산된 VAD 및 조작 값을 Unreal Remote Control과 Pure Data로 보낼 수 있습니다.

## 구성 파일

```text
final_interaction/
  pyproject.toml
  hand_landmarker.task
  pose_landmarker.task
  src/
    realtime_inference_final.py
    hand_extractor.py
    pose_extractor.py
    final_mapper.py
    final_ue_bridge.py
    final_pd_bridge.py
    final_hud_state.py
    final_preview_overlay.py
    final_hud_qt.py
    final_windows_overlay.py
```

## 필요 조건

- `uv`
- 웹캠
- `hand_landmarker.task`, `pose_landmarker.task` 파일
- Unreal로 값을 보낼 경우:
  - Unreal Remote Control 서버가 `http://localhost:30010`에서 실행 중이어야 합니다.
  - Remote Control Preset 이름은 `RCP_WorldVariable`이어야 합니다.
- Pure Data로 값을 보낼 경우:
  - Pure Data 쪽 TCP 수신 포트가 `127.0.0.1:30025`로 열려 있어야 합니다.
- Unreal 위 HUD를 사용할 경우:
  - Windows 실행 환경을 기준으로 합니다.
  - Unreal은 exclusive fullscreen이 아니라 borderless fullscreen 또는 windowed fullscreen으로 실행하는 것을 권장합니다.

## 설치

`final_interaction` 폴더에서 의존성을 설치합니다.

```bash
cd final_interaction
uv sync
```

## 기본 실행

`final_interaction` 폴더에서 실행합니다.

```bash
uv run final-interaction
```

카메라 번호를 지정하려면 `--camera`를 사용합니다.

```bash
uv run final-interaction --camera 0
```

## Unreal로 값 보내기

Unreal Remote Control 서버가 실행 중일 때 `--send`를 붙입니다.

```bash
uv run final-interaction --camera 0 --send
```

전송 대상 기본값은 다음과 같습니다.

```text
URL: http://localhost:30010
Preset: RCP_WorldVariable
```

주요 전송 값:

- `Target V`
- `Target A`
- `Target D`
- `Flower Density`
- `decay`
- `Min Speed`
- `Max Speed`
- `L_Grip`, `R_Grip`
- `L_Y Location`, `L_ Z Location`
- `R_Y Location`, `R_ Z Location`

특정 thrust 동작이 감지되면 Unreal의 `SwitchToWorld` 또는 `Switch to World` 함수를 호출해 `galaxy` 월드로 전환합니다.

## Pure Data로 값 보내기

Pure Data TCP 수신 패치가 실행 중일 때 `--send-pd`를 붙입니다.

```bash
uv run final-interaction --camera 0 --send-pd
```

Unreal과 Pure Data를 동시에 사용하려면 두 옵션을 모두 붙입니다.

```bash
uv run final-interaction --camera 0 --send --send-pd
```

## 주요 옵션

```text
--camera INDEX              사용할 카메라 번호. 기본값: 0
--send                      Unreal Remote Control로 값 전송
--send-pd                   Pure Data로 값 전송
--control-hand left|right   관객 기준 조작 손. 기본값: right
--base-v FLOAT              기본 valence 값. 기본값: 0.0
--base-a FLOAT              기본 arousal 값. 기본값: 0.0
--base-d FLOAT              기본 dominance 값. 기본값: 0.0
--world-json-dir PATH       world_XXXX.json 파일 폴더. 기본값: ../world_spawn_json
--world-poll-sec FLOAT      Unreal에서 현재 world id를 읽는 주기. 기본값: 0.1
--vad-footprint-dir PATH    world별 VAD footprint 저장 폴더. 기본값: ../vad_footprint
--camera-width INT          요청할 웹캠 너비. 기본값: 640
--camera-height INT         요청할 웹캠 높이. 기본값: 360
--camera-fps INT            요청할 웹캠 FPS. 기본값: 30
--pose-every INT            N프레임마다 pose detection 실행. 기본값: 3, 0이면 pose fallback 비활성화
--preview-overlay           카메라 프리뷰에 핵심 상태와 손/포즈 스켈레톤 표시. 기본값: 켜짐
--no-preview-overlay        카메라 프리뷰 overlay 비활성화
--debug-overlay             카메라 프리뷰에 디버그 라벨과 상세 상태 표시
--hud                       투명 HUD 창 표시
--hud-monitor INT           HUD를 띄울 모니터 인덱스. 기본값: 0
--hud-opacity FLOAT         HUD 시각 요소 opacity 배율. 기본값: 1.0
--hud-scale FLOAT           HUD 시각 요소 크기 배율. 기본값: 1.0
--hud-click-through         HUD 마우스 입력 통과. 기본값: 켜짐
--no-hud-click-through      HUD 마우스 입력 통과 비활성화
--hud-topmost               HUD를 항상 위에 표시. 기본값: 켜짐
--no-hud-topmost            HUD 항상 위 표시 비활성화
--perf-log-sec FLOAT        N초마다 FPS와 단계별 처리 시간 출력. 기본값: 0.0
```

## 디버그 실행 예시

```bash
uv run final-interaction --camera 0 --debug-overlay --perf-log-sec 2
```

## Unreal 위 HUD 실행 예시

Windows 전시 PC에서 Unreal을 borderless fullscreen 또는 windowed fullscreen으로 띄운 뒤 같은 모니터 인덱스를 지정합니다.

```powershell
uv run final-interaction --camera 0 --send --send-pd --hud --hud-monitor 0 --perf-log-sec 2
```

HUD는 Windows에서 `pywin32`로 click-through/topmost 창 스타일을 적용합니다. macOS에서는 HUD 레이아웃 검증은 가능하지만 Windows native click-through는 적용되지 않습니다.

## 종료

실행 중 표시되는 카메라 프리뷰 창에서 `q`를 누르면 종료합니다.
