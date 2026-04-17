# Sound Checklist

현재 기준:
- 배경 루프는 [`test7_VAD.pd`](c:/Users/이성준/Desktop/Cosmos/cosmos_sound/test7_VAD.pd)를 기준선으로 사용
- 오디오 소스는 `C:/cosmos_audio` 아래 파일 사용
- 아직 효과음/상태 전환 사운드는 미완성

## 1. Background Loop Locked

목표:
- 배경 루프의 기본 결을 고정한다
- V/A/D 변화가 최소한 납득 가능한 수준으로 들린다

확인할 것:
- [`test7_VAD.pd`](c:/Users/이성준/Desktop/Cosmos/cosmos_sound/test7_VAD.pd)에서 기본 음색/여백이 만족스러운지
- `V=-1`일 때 전체가 더 낮고 무겁게 들리는지
- `V=+1`일 때 전체가 더 밝고 가볍게 들리는지
- `A`가 낮을 때 느슨하고, 높을 때 더 활기차게 느껴지는지
- `D`가 filter openness로 자연스럽게 들리는지

성공 기준:
- "기본 월드 루프는 이걸로 간다"라고 말할 수 있는 상태

현재 상태:
- 거의 완료
- 필요 시 미세 조정만

## 2. Flower Sound State Machine

목표:
- 꽃이 피고 지는 변화에 맞춰 부스럭 사운드가 자연스럽게 이어진다

사용 파일:
- `grow_start.wav`
- `grow_ing.wav`
- `grow_end.wav`
- 후보: `grow1.wav`, `grow2.wav`, `grow3.wav`

구조:
- `grow_start`
- `grow_loop`
- `grow_end`

확인할 것:
- 변화 시작 시 `grow_start`가 자연스럽게 붙는지
- 변화 지속 중 `grow_ing` loop가 어색하지 않은지
- 값 변화가 멈췄을 때 `grow_end`로 빠지는지
- 너무 반복적으로 들리면 `grow1~3`을 보조로 섞을지

성공 기준:
- 꽃 변화가 들리는 사운드 흐름으로 납득됨

## 3. Ready On / Ready Off

목표:
- `snap`으로 interaction mode 진입/이탈이 귀로도 즉시 느껴진다
- 하지만 실세계 효과음처럼 들리지 않고, 배경 루프와 같은 세계의 재질을 유지한다

방향:
- 길이 약 `0.6 ~ 0.9s`
- `ready_on`: 열리는 느낌, "도로롱~"
- `ready_off`: 닫히는 느낌
- 둘 다 배경 루프와 비슷한 음색/재질 사용

확인할 것:
- `ready_on`이 너무 UI 효과음처럼 들리지 않는지
- `ready_off`가 억지 역재생처럼 들리지 않는지
- loop와 톤이 같은 세계에 속하는지

성공 기준:
- ready 상태 전환이 명확하면서도 튀지 않음

## 4. Ready Mode Overlay

목표:
- ready 상태에 들어가면 배경 월드가 살짝 달라진 느낌이 난다
- loop 자체가 완전히 바뀌는 게 아니라, 같은 세계가 활성화된 느낌

방향:
- passive loop
- active/ready overlay
- crossfade 또는 layer add

확인할 것:
- ready 전/후 차이가 너무 약하지 않은지
- 반대로 너무 다른 곡처럼 느껴지지 않는지

성공 기준:
- "관람 모드"와 "인터랙션 준비 모드"의 분위기 차이가 은근하지만 분명함

## 5. Statue FX

목표:
- 조각상의 녹슬기/회복이 짧은 질감 사운드로 들린다

필요 파일:
- `statue_decay.wav`
- `statue_recover.wav`

확인할 것:
- decay는 거칠고 마른 재질 변화처럼 들리는지
- recover는 너무 판타지스럽지 않고 회복/정리 느낌인지

성공 기준:
- 짧지만 세계관에 맞는 재질 변화 사운드가 준비됨

## 6. Pd Integration

목표:
- Python/interaction 상태를 Pd에 연결해 실제 작품 사운드 흐름으로 이어진다

연결 대상:
- background loop
- grow start/loop/end
- ready on/off
- ready overlay
- statue decay/recover

확인할 것:
- Pd는 playback/mix 중심으로 단순하게 유지되는지
- 상태 판단은 가능하면 Python이 담당하는지

성공 기준:
- 루프 + 효과음 + 상태 전환이 한 패치 흐름으로 통합됨

## 권장 진행 순서

1. Background Loop Locked
2. Flower Sound State Machine
3. Ready On / Ready Off
4. Ready Mode Overlay
5. Statue FX
6. Pd Integration

## 다음에 "사운드 체크리스트 알려줘"라고 하면

아래 순서로 다시 안내:

1. Background Loop Locked
2. Flower Sound State Machine
3. Ready On / Ready Off
4. Ready Mode Overlay
5. Statue FX
6. Pd Integration
