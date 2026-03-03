# 🎥 Webcam Motion Capture

웹캠을 이용한 실시간 인체 포즈 추정 및 모션 캡처 프로젝트입니다.  
OpenCV + MediaPipe를 기반으로 33개의 인체 키포인트를 실시간으로 추출하고, 동작 데이터를 JSON/CSV로 저장합니다.

---

## 🛠 설치 방법

### 1. 가상환경 생성 및 활성화

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

---

## 🚀 실행

```bash
python main.py
```

---

## ⌨️ 단축키

| 키 | 동작 |
|---|---|
| `R` | 녹화 시작 / 정지 |
| `S` | 녹화 데이터 저장 (JSON + CSV) |
| `Q` | 프로그램 종료 |

---

## 📁 프로젝트 구조

```
interaction/
├── main.py             # 메인 실행 파일
├── requirements.txt    # 의존성
├── README.md
└── src/
    ├── capture.py      # 웹캠 캡처 (OpenCV)
    ├── pose.py         # 포즈 추정 (MediaPipe)
    ├── renderer.py     # 스켈레톤 렌더링 + HUD
    └── recorder.py     # 모션 데이터 녹화 및 저장
```

---

## 📦 출력 파일

녹화 후 `S`를 누르면 `recordings/` 디렉토리에 저장됩니다:

- `motion_YYYYMMDD_HHMMSS.json` — 전체 프레임 × 33 랜드마크 데이터
- `motion_YYYYMMDD_HHMMSS.csv` — 동일 데이터를 행 단위 CSV로 정리

---

## 📐 랜드마크 데이터 구조 (JSON)

```json
{
  "frames": [
    {
      "frame": 0,
      "timestamp": 0.0,
      "landmarks": [
        { "name": "NOSE", "x": 640.2, "y": 320.5, "z": -0.12, "visibility": 0.99 },
        ...
      ]
    }
  ]
}
```

---

## 📚 Tech Stack

- [OpenCV](https://opencv.org/) `>=4.8`
- [MediaPipe](https://developers.google.com/mediapipe) `>=0.10`
- [NumPy](https://numpy.org/) `>=1.24`
