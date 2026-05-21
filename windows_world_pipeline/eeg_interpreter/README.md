# EEG Interpreter

`eeg_interpreter`는 EEG/VAD JSON을 world instruction JSON으로 바꾸는 모듈입니다.

현재는 `windows_world_pipeline/run_pipeline.py`에서 내부 모듈로 호출합니다.

## 단독 테스트

```powershell
python windows_world_pipeline\eeg_interpreter\generate_world_instruction.py --config windows_world_pipeline\eeg_interpreter\config.example.json --eeg ..\..\eeg_json\person_001_eeg.json
```

## LLM 설정

기본값은 `dry_run: true`입니다.

```json
{
  "llm": {
    "dry_run": true
  }
}
```

실제 Gemini API를 쓰려면 `dry_run`을 `false`로 바꾸고 `GEMINI_API_KEY` 환경변수를 설정합니다.

## WebSocket

`listen_eeg_websocket.py`는 나중에 실제 EEG WebSocket을 직접 받을 때를 위한 후보 코드입니다.

현재 주 파이프라인은 WebSocket 대신 `eeg_json/` 폴더 watch 방식을 사용합니다.
