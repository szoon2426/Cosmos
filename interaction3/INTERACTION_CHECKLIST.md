# Interaction Checklist

`interaction3`의 현재 인터랙션 개발/검증 순서를 정리한 체크리스트다.
나중에 "인터랙션 체크리스트 알려줘"라고 요청하면 이 문서를 기준으로 다시 안내한다.

## 현재 단계 요약

우리는 지금 `3단계`, `4단계` 설계/구현을 끝내고 실제 테스트로 검증해야 하는 시점이다.
이 검증이 끝나야 `5단계` 노이즈 튜닝으로 넘어간다.

## 전체 체크리스트

### 1. 3단계 테스트: Presence Gate / Tracking

- 얼굴 기반 presence gate가 안정적으로 lock되는지 확인
- 관람객으로 한 번 잡히면 tracking이 자연스럽게 유지되는지 확인
- 측면 보기, 고개 돌리기, 짧은 landmark 튐에서 바로 해제되지 않는지 확인
- 뒤돌기 또는 tracking lost 상황에서 의도대로 해제되는지 확인

성공 기준:
- `face_present`, `track=locked`가 과하게 흔들리지 않는다
- 뒤돌기 해제 의도는 유지되지만, 너무 민감하게 끊기지 않는다

### 2. 4단계 테스트: Trigger / Sustain / Release

#### `snap -> ready gate`
- 첫 snap으로 `ready=True`가 되는지
- 두 번째 snap으로 `ready=False`가 되는지
- ready가 아닐 때 `open/rise/prayer/breath`가 시작되지 않는지

#### `open`
- trigger가 안정적으로 걸리는지
- 시작 시점 손 간격이 base로 잡히는지
- 손 간격이 벌어지면 `open_amount`가 증가하는지
- 손 간격이 줄어들면 `open_amount`가 감소하는지
- 팔을 아래로 내리면 release 되는지

#### `rise`
- trigger가 안정적으로 걸리는지
- 시작 시점 손 높이가 base로 잡히는지
- 손이 더 올라가면 `rise_amount`가 증가하는지
- 손이 조금 내려오면 `rise_amount`가 감소하는지
- 손이 내려온다고 바로 종료되지 않는지
- 손이 옆으로 벗어났을 때만 종료되는지

#### `prayer`
- prediction과 pose가 안정적으로 prayer로 인식되는지
- 유지 시간에 따라 회복량이 점점 증가하는지
- 자세가 풀리면 종료되는지

#### `breath`
- breath pose gate가 의도대로 동작하는지
- `open`과 충분히 구분되는지
- 유지 시간에 따라 회복량이 점점 증가하는지
- 자세가 풀리면 종료되는지

#### `snap`
- snap gate가 먼저 열리는지
- 엄지/검지 교차 순간만 snap으로 인식되는지
- 팔만 든 상태에서는 snap이 잘못 뜨지 않는지

성공 기준:
- 각 인터랙션이 의도한 trigger / amount / release 흐름을 만족한다
- `snap -> ready gate` 중심 구조가 자연스럽게 동작한다

### 3. 5단계 진행: Noise Tuning

3단계와 4단계 테스트를 모두 만족한 뒤 진행한다.

튜닝 대상:
- prediction 흔들림
- `uncertain` 스파이크
- 다른 인터랙션 라벨의 찰나 튐
- active / release / cooldown 파라미터
- rival label 흡수 정도
- tracking lost / back / drift 민감도

성공 기준:
- active 상태가 필요 이상으로 출렁이지 않는다
- 짧은 노이즈 프레임이 인터랙션을 깨지 않는다
- 실제 관람객 움직임에서는 자연스럽고, 오작동은 줄어든다

### 4. 최종 통합 테스트

- noise tuning 이후 전체 흐름을 다시 처음부터 끝까지 검증
- `snap -> ready on -> interaction -> ready off`가 전체 시나리오로 자연스럽게 이어지는지 확인
- `open / rise / prayer / breath / snap` 모두 실제 동작 시퀀스에서 점검
- face/tracking/release가 의도대로 연결되는지 최종 확인

성공 기준:
- 인터랙션 흐름 전체가 작품 경험으로 자연스럽다
- 개별 기능이 아니라 전체 시퀀스로 봐도 안정적이다

### 5. Unreal 직접 연결

- 최종 통합 테스트가 끝난 뒤 Unreal과 실제 연결
- `V/A/D`, `decay`, `density`, `min/max speed`가 작품에서 의도대로 반영되는지 확인
- 필요하면 Unreal 쪽 보간/연출 파라미터를 추가 튜닝

성공 기준:
- 인터랙션 결과가 Unreal 월드에 의도대로 전달된다
- 작품에서 실제로 사용할 수준의 반응성이 나온다

## 현재 우리가 기억해야 할 순서

1. `3단계 테스트`
2. `4단계 테스트`
3. `5단계 노이즈 튜닝`
4. `최종 통합 테스트`
5. `Unreal 직접 연결`

## 메모

- `prayer`, `breath`의 최종 회복 기준은 나중에 `eeg-emotions` JSON의 base `V/A/D`와 연결할 예정
- 현재는 구조와 흐름을 먼저 검증하는 단계
