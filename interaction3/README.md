# interaction3

`interaction3`는 rule-based gesture 판정 대신, 포즈 시퀀스를 학습해서 분류하는 ML 기반 인터랙션 파이프라인입니다.

핵심 흐름:

1. MediaPipe Pose로 상체 랜드마크 추출
2. 정규화/상대좌표 기반 특징 추출
3. sliding window 시퀀스 특징 생성
4. 라벨링 데이터 수집
5. kNN 학습
6. 실시간 분류
7. 분류 결과를 VAD로 변환
8. Unreal / Purr-Data로 실시간 송신

## 주요 모듈

- `src/pose_extractor.py`
  - MediaPipe Pose 기반 상체 랜드마크 추출
- `src/feature_engineering.py`
  - shoulder-center / shoulder-width 정규화
  - 프레임별 및 시퀀스 특징 생성
- `src/data_collector.py`
  - 키보드 라벨링 기반 데이터 수집
- `src/train_model.py`
  - kNN 학습 및 모델 저장
- `src/realtime_inference.py`
  - 실시간 분류와 smoothing
- `src/vad_mapper.py`
  - 분류 결과를 Cosmos 출력 파라미터로 변환
- `src/ue_bridge.py`
  - Unreal Remote Control Preset 송신
- `src/pd_bridge.py`
  - Purr-Data UDP 송신

## 기본 라벨

기본적으로 아래 4개 클래스를 가정합니다.

- `rise`
- `open`
- `prayer`
- `breath`

## 실행 흐름

1. 데이터 수집
2. 모델 학습
3. 실시간 추론
4. 필요 시 Unreal / Pd 브리지 연결

