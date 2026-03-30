# interaction2

`interaction2`는 `interaction`의 구조를 참고하되, 새로운 유지형 인터랙션 파이프라인을 위한 작업 공간입니다.

현재 목표 흐름:

1. 카메라 입력
2. Pose / Hand 추정
3. 유지형 인터랙션 판정 (`open`, `rise`, `gather`, `deep_breath`)
4. 내부 VAD 상태 갱신
5. Unreal 송신용 파라미터 계산
6. UE 브리지 전송

구조는 `interaction`과 유사하게 유지합니다.
