# Mac Mini Project: EEG to World Instruction

이 프로젝트는 `eeg.json`을 읽고 감정 상태를 해석한 뒤, Windows 생성 머신이 사용할 월드 생성 지시 JSON만 공유 폴더에 저장합니다. 이미지나 3D 에셋은 만들지 않습니다.

## 실행

```powershell
cd mac_eeg_llm
python -m pip install -r requirements.txt
python generate_world_instruction.py --config config.example.json
```

기본 설정은 `dry_run: true`라서 Gemini API 없이도 예시 JSON을 생성합니다. 실제 Gemini를 쓰려면 `config.example.json`을 복사해 `dry_run`을 `false`로 바꾸고 `GEMINI_API_KEY` 환경 변수를 설정하세요.

## 출력 JSON

공유 폴더에는 다음 형태의 파일이 생성됩니다.

```json
{
  "world_id": "world_0001",
  "world_number": 1,
  "world_concept": "quiet inner weather held inside a sealed vertical shrine",
  "asset_prompt": "single centered sealed observatory...",
  "negative_prompt": "castle, city...",
  "archetype": "sealed_observatory",
  "atmosphere": ["dreamlike", "misty", "soft", "sacred"]
}
```

프롬프트는 단일 고립 오브젝트, 닫힌 실루엣, 외부 시점, 깨끗한 바닥, 모호한 기능을 강하게 유지하도록 설계되어 있습니다.

전체 두 머신 파이프라인과 남은 작업은 [`../PIPELINE_CHECKLIST.md`](../PIPELINE_CHECKLIST.md)를 기준으로 관리합니다.
