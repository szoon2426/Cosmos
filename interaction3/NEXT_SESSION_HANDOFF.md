# Interaction3 Next Session Handoff

이 문서는 `interaction3` 인터랙션 작업을 다음 세션에서 바로 이어가기 위한 현재 상태 요약이다.

## 현재 큰 흐름

- 체크리스트 기준 진행 순서:
  1. 3단계 테스트
  2. 4단계 테스트
  3. 5단계 노이즈 튜닝
  4. 최종 통합 테스트
  5. Unreal 직접 연결
- 현재는 `5단계 노이즈 튜닝` 초입이다.
- 사용자가 피로해서 잠시 중단했고, 다음 세션에서 바로 이어서 진행해야 한다.

## 현재 핵심 인터랙션 구조

- `snap`은 ready gate 토글 역할
  - 첫 snap: `ready_gate=True`
  - 두 번째 snap: `ready_gate=False`
- `ready_gate=True`일 때만 다음 인터랙션 가능
  - `open`
  - `rise`
  - `prayer`
  - `breath`
- `open`
  - trigger 후 `base_open_distance` 저장
  - 손 간격 변화로 amount 계산
- `rise`
  - trigger 후 `base_rise_height` 저장
  - 손 높이 변화로 amount 계산
  - 손을 내린다고 바로 종료하지 않음
  - 옆으로 벗어나거나 tracking lost일 때 종료
- `prayer`, `breath`
  - 유지 시간 기반 회복형 인터랙션
  - 나중에 `eeg-emotions` JSON의 base V/A/D와 연결 예정

## snap 관련 현재 상태

### 현재 의도

- `snap_gate` = 팔 자세
- `snap_event` = 손가락 순간
- `final_snap` = `snap_gate`와 `snap_event`를 모두 만족할 때만 true

### 현재 구현

- `snap_gate`는 pose 기반
- `snap_event`는 별도 학습 모델 기반
- 관련 파일:
  - [`interaction3/src/feature_engineering.py`](c:/Users/이성준/Desktop/Cosmos/interaction3/src/feature_engineering.py)
  - [`interaction3/src/train_snap_model.py`](c:/Users/이성준/Desktop/Cosmos/interaction3/src/train_snap_model.py)
  - [`interaction3/src/realtime_inference.py`](c:/Users/이성준/Desktop/Cosmos/interaction3/src/realtime_inference.py)
  - [`interaction3/src/stage4_test.py`](c:/Users/이성준/Desktop/Cosmos/interaction3/src/stage4_test.py)

### snap 전용 학습 자산

- 데이터셋:
  - [`interaction3/data/snap_dataset_v1.jsonl`](c:/Users/이성준/Desktop/Cosmos/interaction3/data/snap_dataset_v1.jsonl)
- 모델:
  - [`interaction3/models/snap_model_v1.joblib`](c:/Users/이성준/Desktop/Cosmos/interaction3/models/snap_model_v1.joblib)
- 자동 추출된 snap 후보 프레임:
  - [`interaction3/data/snap_annotations.json`](c:/Users/이성준/Desktop/Cosmos/interaction3/data/snap_annotations.json)

### snap 현재 문제

- 사용자가 보고한 증상:
  - `snap gate`만 열려도 `ready on/off`처럼 느껴짐
- 대응:
  - `snap` 전용 feature를 손가락 중심으로 변경
  - `snap.mp4` 전체를 positive로 보지 않고, 자동 검출된 후보 프레임 주변만 positive로 학습
- 하지만 아직 사용자는 충분히 만족하지 못했고, 다음 세션에서 재검증 필요

## 노이즈/트리거 현재 상태

### 최근 변경

- `rise/open/prayer/breath`는 이제 프레임 수가 아니라 `1초 유지`일 때만 trigger
- `snap`만 짧은 유지 시간으로 둠
- `stage4_test.py` / `realtime_inference.py` 화면에 `arming_elapsed` 표시 추가

### 현재 사용자 관찰

- `snap gate`를 열고 있는 상태에서 `rise`가 실행되는 일이 있음
- 사용자는 이게:
  - 학습 부족인지
  - 노이즈인지
  - trigger 조건 문제인지
확신하지 못하고 있음

### 현재 판단

- 우선은 4단계를 다시 뜯기보다 5단계 노이즈 튜닝으로 보는 게 맞다
- 다만 다음 세션에서 아래를 먼저 확인해야 함:
  - `prediction=rise`가 실제로 1초 이상 유지되는지
  - `candidate=rise`로 얼마나 빨리 들어가는지
  - `arming_elapsed`가 1초를 채우는지
  - `rise_side`가 실제로도 열려 있는지

## 다음 세션에서 바로 할 일

### 1. stage4 다시 확인

실행:

```powershell
py -3 interaction3\src\stage4_test.py
```

확인할 것:

- `snap_gate`
- `ml=snap/non_snap`
- `snap_event`
- `final_snap`
- `prediction`
- `candidate`
- `arming_elapsed`
- `rise_side`

목표:

- `snap gate`만 열려 있을 때는 `final_snap=False`
- `snap` 손가락 순간에서만 `final_snap=True`
- `snap gate` 상태에서 `rise`가 1초 이상 유지되지 않는지 확인

### 2. rise 오검출 원인 분리

다음 중 어디가 문제인지 판단:

- `prediction=rise`가 실제로 오래 뜬다
  - 학습 데이터 / 모델 문제 가능성 큼
- `prediction`은 잠깐인데 `candidate/active`까지 간다
  - 상태 머신 / 노이즈 억제 문제
- `rise_side`가 쉽게 열린다
  - rise gate 문제

### 3. 필요시 다음 액션

#### A. 노이즈 튜닝 계속

- rival label 억제
- release/hysteresis 조정
- ready 상태에서 다른 인터랙션 start 조건 완화/강화

#### B. neutral/idle 클래스 추가 검토

사용자가 제안한 방향:

- 인터랙션을 하지 않는 영상
- 팔을 조금만 움직이는 영상
- `snap gate` 준비 상태지만 실제 인터랙션은 아닌 영상

이런 영상을 `uncertain`보다는 `idle` 또는 `neutral` 클래스로 학습하는 방향 검토

현재 판단:

- 다음 세션에서 `rise` 오검출이 실제로 prediction 단계에서 오래 유지되면
  - `idle/neutral` 클래스 추가 학습이 유효함

## 관련 실행 명령

### 3단계 트래킹 확인

```powershell
py -3 interaction3\src\tracking_test.py
```

### 4단계 인터랙션 확인

```powershell
py -3 interaction3\src\stage4_test.py
```

### 실시간 추론

```powershell
py -3 interaction3\src\realtime_inference.py
```

### snap 전용 학습 재실행 예시

```powershell
py -3 interaction3\src\train_snap_model.py --videos snap.mp4,snap2.mp4,open2.mp4,rise2.mp4,pray2.mp4,breath2.mp4 --frame-step 8 --stride 2 --auto-annotate
```

## 다음 세션 시작 멘트 가이드

사용자가 다음에 이렇게 말하면 바로 이 문서를 기준으로 이어간다.

- `인터랙션 이어서 하자`
- `interaction3 계속하자`
- `인터랙션 체크리스트 알려줘`
- `snap 문제 다시 보자`

그때 우선순위:

1. 이 문서 확인
2. `stage4_test.py` 기준 현상 재확인
3. `snap`과 `rise` 오검출 원인 분리
4. 5단계 노이즈 튜닝 계속
