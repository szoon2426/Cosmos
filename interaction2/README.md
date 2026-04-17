# interaction2

`interaction2`는 `interaction`의 구조를 참고하되, 유지형 인터랙션과 Unreal 연동을 위해 새로 정리한 파이프라인입니다.

현재 흐름:

1. 카메라 입력
2. Pose / Hand 추정
3. 인터랙션 판정 (`open`, `rise`, `gather`, `deep_breath`)
4. 내부 VAD 상태 갱신
5. Unreal 송신용 파라미터 계산
6. UE 브리지 전송

## EEG 감정 프로파일

`eeg-emotions`는 `interaction2`와 동시에 동작하는 입력원이 아니라, 먼저 개인의 감정 반응 프로파일을 추출하는 역할로 사용합니다.

그 결과물은 [`eeg_profile.example.json`](c:/Users/이성준/Desktop/Cosmos/interaction2/eeg_profile.example.json) 같은 JSON으로 생각하면 됩니다.

이 프로파일은 아래를 재정의합니다.

- `baseline_v`, `baseline_a`, `baseline_d`
  - 이 사람이 기본적으로 돌아가는 감정 중심점
- `v_gain`, `a_gain`, `d_gain`
  - 제스처가 각 축을 얼마나 민감하게 움직이는지
- `v_return_rate`, `a_return_rate`, `d_return_rate`
  - 목표값을 향해 얼마나 빨리 이동하는지
- `v_min/max`, `a_min/max`, `d_min/max`
  - 개인별 감정 범위

즉 `interaction2`는 동일한 제스처를 써도, EEG에서 만든 감정 프로파일에 따라 증가율/감소율/복원 속도/안정권이 달라지게 됩니다.

## 연결 포인트

프로파일 로더는 [`src/eeg_profile.py`](c:/Users/이성준/Desktop/Cosmos/interaction2/src/eeg_profile.py)에 있고, 인터랙션 엔진은 [`src/gesture.py`](c:/Users/이성준/Desktop/Cosmos/interaction2/src/gesture.py)에서 이 값을 사용합니다.

실제 운영에서는 `eeg-emotions`가 만든 JSON을 로드해서 `InteractionEngine.set_profile(...)`로 주입하면 됩니다.
