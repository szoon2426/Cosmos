# Final Interaction Checklist

기준 브랜치: `final_0505`  
작업 기준 폴더: `final_interaction/`

## 현재 구현 완료
- [x] `final_interaction/` 전용 구조 분리
  - `final_interaction/src/` 아래에서만 새 인터랙션 작업
  - `hand_landmarker.task`, `pose_landmarker.task`를 전용 경로로 분리
- [x] 손 추적 기본 파이프라인
  - 카메라 입력
  - MediaPipe hand landmark 추출
  - pose landmark 보조 추출
- [x] One Euro Filter 적용
  - `x`, `y`, `z`
  - `open_strength`, `grab_strength`
  - `palm_radius`
- [x] control hand 안정화
  - 기대 손 우선 추적
  - handedness가 잠깐 뒤집혀도 바로 반대 손으로 넘어가지 않음
  - 손이 잠깐 사라져도 grace로 짧게 유지
- [x] pose fallback
  - 손 인식이 약해질 때 팔 끝 정보로 보조
- [x] 현재 `grab/open` 판정 방식 정리
  - 손가락별 비율 계산:
    - `INDEX = distance(5, 8) / distance(0, 5)`
    - `MIDDLE = distance(9, 12) / distance(0, 9)`
    - `RING = distance(13, 16) / distance(0, 13)`
    - `PINKY = distance(17, 20) / distance(0, 17)`
  - `open_strength`
    - 각 비율을 `0.45 ~ 0.95` 구간으로 `0~1` remap
    - `0.82 * mean + 0.18 * min`
  - `grab_strength`
    - 각 비율을 `0.28 ~ 0.62` 구간에서 반대로 해석
    - `0.72 * mean + 0.28 * max`
  - `grab_active`
    - `grab_strength >= 0.85`
    - 그리고 `open_strength <= 0.42`
- [x] 손가락 landmark 이름 및 ratio 디버그 표시
  - `WRIST`, `INDEX_MCP`, `INDEX_TIP` 등 표시
  - `I/M/R/P` ratio 표시
- [x] final 인터랙션 상태 골격
  - `idle / open / grab`
  - pointer 좌표 출력
  - base VAD 적응 및 느린 복귀 골격
  - camera switch pulse 골격

## 현재 판정 해석
- [x] 펼친 손과 오므린 손의 ratio 차이는 충분히 분리됨
  - 펼친 손 예시: `0.85 ~ 1.02`
  - 오므린 손 예시: `0.18 ~ 0.31`
- [x] 문제는 feature가 아니라 score 매핑 구간이었다
  - 이전에는 펼친 손도 open으로 충분히 안 올라감
  - 지금은 ratio 분포에 맞게 `open/grab` 매핑을 다시 잡음

## 바로 다음 작업
- [ ] `idle -> open` 시작 조건 다시 튜닝
  - 펼친 손으로 더 안정적으로 인터랙션 진입
- [ ] `grab` 상태에서 V/A/D 조작 연결 완성
  - 좌우로 벌리기/좁히기 -> `V`
  - 위아래로 벌리기/좁히기 -> `A`
  - 뒤로 빼기 -> `D 감소`
- [ ] `open` 상태 유지 로직 정리
  - grab 이후 open으로 풀면 현재 상태 유지
  - 종료 후 adaptive base에 조금 반영
  - 천천히 원래 base로 복귀
- [ ] camera switch 동작 완성
  - open 상태에서 뒤로 뺐다가 빠르게 앞으로 던질 때 pulse 발생
- [ ] Unreal 연동용 값 정리
  - `Interaction Active`
  - `Pointer X/Y`
  - `Target V/A/D`
  - `Grab Active`
  - `Open Strength`
  - `Grab Strength`
  - `Switch To Camera`

## 나중에 고려
- [ ] 손 ratio 외에 angle 기반 straightness 보조 특징 추가 가능성
- [ ] MMPose 기반 손/전신 입력 확장 가능성
- [ ] Momentum식 물리 해석 방식 참고 가능성
