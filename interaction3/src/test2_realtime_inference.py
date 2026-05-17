from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    from feature_engineering import normalize_landmarks
    from hand_extractor import HandExtractor
    from pd_bridge import PDBridge
    from pose_extractor import PoseExtractor
    from realtime_inference import draw_skeleton, draw_vad_meter
    from ue_bridge import UEBridge
    from vad_mapper import compute_unreal_payload
else:
    from .feature_engineering import normalize_landmarks
    from .hand_extractor import HandExtractor
    from .pd_bridge import PDBridge
    from .pose_extractor import PoseExtractor
    from .realtime_inference import draw_skeleton, draw_vad_meter
    from .ue_bridge import UEBridge
    from .vad_mapper import compute_unreal_payload


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ── 애니메이션 상태 ──────────────────────────────────────────────
class AnimationState:
    """현재 활성 애니메이션을 관리한다."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.mode: str | None = None       # rise / open / prayer / breath
        self.direction: str = "up"         # up / down (rise, open 용)
        self.duration: float = 0.0         # prayer, breath 전체 시간(초)
        self.started_at: float = 0.0       # 시작 시각
        self.amount: float = 0.0           # 현재 보간 값 0~1
        self.running: bool = False

    # ── 명령어 파싱 ──────────────────────────────────────────────
    def parse_command(self, raw: str) -> str:
        """명령어 문자열을 파싱해서 상태를 설정한다. 결과 메시지 반환."""
        parts = raw.strip().lower().split()
        if not parts:
            return ""

        cmd = parts[0]

        # ── stop / off ──
        if cmd in ("stop", "off", "0"):
            with self.lock:
                self.mode = None
                self.running = False
                self.amount = 0.0
            return "[STOP] 전송 중지"

        # ── rise up / rise down ──
        if cmd == "rise":
            direction = parts[1] if len(parts) > 1 else "up"
            if direction not in ("up", "down"):
                return f"[ERROR] rise up / rise down 만 가능 (입력: {direction})"
            dur = self._parse_duration(parts[2]) if len(parts) > 2 else 3.0
            with self.lock:
                self.mode = "rise"
                self.direction = direction
                self.duration = dur
                self.started_at = time.time()
                self.amount = 0.0
                self.running = True
            return f"[RISE {direction.upper()}] {dur:.1f}초"

        # ── open / close ──
        if cmd in ("open", "close"):
            direction = "up" if cmd == "open" else "down"
            dur = self._parse_duration(parts[1]) if len(parts) > 1 else 3.0
            with self.lock:
                self.mode = "open"
                self.direction = direction
                self.duration = dur
                self.started_at = time.time()
                self.amount = 0.0
                self.running = True
            label = "OPEN" if cmd == "open" else "CLOSE"
            return f"[{label}] {dur:.1f}초"

        # ── prayer Nsec ──
        if cmd == "prayer":
            dur = self._parse_duration(parts[1]) if len(parts) > 1 else 4.0
            with self.lock:
                self.mode = "prayer"
                self.direction = "up"
                self.duration = dur
                self.started_at = time.time()
                self.amount = 0.0
                self.running = True
            return f"[PRAYER] {dur:.1f}초"

        # ── breath Nsec ──
        if cmd == "breath":
            dur = self._parse_duration(parts[1]) if len(parts) > 1 else 5.0
            with self.lock:
                self.mode = "breath"
                self.direction = "up"
                self.duration = dur
                self.started_at = time.time()
                self.amount = 0.0
                self.running = True
            return f"[BREATH] {dur:.1f}초"

        return f"[ERROR] 알 수 없는 명령: {cmd}"

    @staticmethod
    def _parse_duration(s: str) -> float:
        """'4sec', '4s', '4' 같은 문자열에서 초 단위 숫자를 꺼낸다."""
        s = s.replace("sec", "").replace("s", "")
        try:
            return max(0.5, float(s))
        except ValueError:
            return 3.0

    # ── 매 프레임 호출: 현재 VAD 반환 ────────────────────────────
    def tick(self) -> tuple[float, float, float]:
        """현재 시각 기준 VAD(V, A, D) 반환. 없으면 (0,0,0)."""
        with self.lock:
            if not self.running or self.mode is None:
                return (0.0, 0.0, 0.0)

            elapsed = time.time() - self.started_at
            progress = clamp(elapsed / self.duration, 0.0, 1.0)

            if self.direction == "down":
                progress = 1.0 - progress

            self.amount = progress

            # 애니메이션 끝나면 멈춤
            if elapsed >= self.duration:
                self.running = False

            return self._vad_for_mode(self.mode, progress)

    @staticmethod
    def _vad_for_mode(mode: str, amount: float) -> tuple[float, float, float]:
        if mode == "rise":
            return (0.2 * amount, 0.8 * amount, 0.2 * amount)
        if mode == "open":
            return (0.7 * amount, 0.1 * amount, 0.7 * amount)
        if mode == "prayer":
            return (0.2 * amount, -0.2 * amount, 0.2 * amount)
        if mode == "breath":
            return (0.0, -0.6 * amount, 0.0)
        return (0.0, 0.0, 0.0)

    def status_text(self) -> tuple[str, tuple[int, int, int]]:
        with self.lock:
            if not self.mode:
                return "OFF", (180, 180, 180)
            colors = {
                "rise": (80, 220, 255),
                "open": (80, 255, 120),
                "prayer": (255, 180, 80),
                "breath": (200, 160, 255),
            }
            running_mark = " ▶" if self.running else " ■"
            return f"{self.mode.upper()} {self.direction} {self.amount:.2f}{running_mark}", colors.get(self.mode, (200, 200, 200))


# ── 터미널 입력 스레드 ───────────────────────────────────────────
def input_thread(anim: AnimationState, quit_event: threading.Event) -> None:
    HELP = (
        "\n──── 사용 가능한 명령어 ────\n"
        "  rise up [초]      rise 올라감 (기본 3초)\n"
        "  rise down [초]    rise 내려감 (기본 3초)\n"
        "  open [초]         open 열림 (기본 3초)\n"
        "  close [초]        close 닫힘 (기본 3초)\n"
        "  prayer [초]       prayer 진행 (기본 4초)\n"
        "  breath [초]       breath 진행 (기본 5초)\n"
        "  stop / off / 0    전송 중지\n"
        "  help              이 도움말\n"
        "  quit / q          종료\n"
        "────────────────────────────\n"
    )
    print(HELP)
    while not quit_event.is_set():
        try:
            raw = input(">>> ")
        except EOFError:
            break
        raw = raw.strip()
        if raw.lower() in ("quit", "q"):
            quit_event.set()
            break
        if raw.lower() == "help":
            print(HELP)
            continue
        if raw:
            msg = anim.parse_command(raw)
            if msg:
                print(msg)


WINDOW_NAME = "interaction3 command test"


def main() -> None:
    parser = argparse.ArgumentParser(description="Command-driven UE test")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--send", action="store_true", help="Enable sending to UE")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    ue = UEBridge(enabled=args.send)
    pd = PDBridge(enabled=args.send)
    ue.start()
    pd.start()

    if not cap.isOpened():
        print(f"[ERROR] camera {args.camera} could not be opened")
        ue.stop()
        pd.stop()
        sys.exit(1)

    extractor = PoseExtractor()
    hand_extractor = HandExtractor()

    anim = AnimationState()
    quit_event = threading.Event()
    prev_decay_value = 0.0
    recover_level = 0.0

    t = threading.Thread(target=input_thread, args=(anim, quit_event), daemon=True)
    t.start()

    try:
        while not quit_event.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = extractor.process(frame_rgb)
            hand_results = hand_extractor.process(frame_rgb)
            landmarks = extractor.extract_upper_body(pose_results)

            if landmarks is not None:
                landmarks.update(hand_extractor.extract_keypoints(hand_results))
                draw_skeleton(frame, landmarks)

            # VAD 계산 & 전송
            vad = anim.tick()
            raw_payload = compute_unreal_payload(*vad)
            payload = raw_payload.as_dict()

            # PD 연속 값 계산
            current_flower = clamp((float(raw_payload.density) - 0.4) / 0.6, 0.0, 1.0) if vad != (0.0, 0.0, 0.0) else 0.0
            current_decay = clamp(float(raw_payload.decay) / 1.2, 0.0, 1.0) if vad != (0.0, 0.0, 0.0) else 0.0
            fountain_value = clamp(((float(raw_payload.min_speed) + float(raw_payload.max_speed)) * 0.5) / 600.0, 0.0, 1.0)

            # DECAY: raw 값 직접 전송 (값이 양수인 동안 PD 사운드가 계속 재생됨)
            decay_send = current_decay

            # RECOVER: decay가 줄어드는 동안 지속적으로 켜짐
            if current_decay < prev_decay_value - 0.001:
                # decay가 감소 중 = 회복 중 → 레벨 유지
                recover_level = 1.0
            elif current_decay > prev_decay_value + 0.001:
                # decay가 증가 중 → recover 즉시 끔
                recover_level = 0.0
            else:
                # 안정 상태 → 서서히 페이드아웃
                recover_level = max(0.0, recover_level - 0.02)
            prev_decay_value = current_decay

            pd_values = {
                "READY_MODE": 1.0,
                "W": clamp(float(raw_payload.a), -1.0, 1.0),
                "F": clamp(fountain_value * 2.0 - 1.0, -1.0, 1.0),
                "FLOWER": current_flower,
                "DECAY": decay_send,
                "RECOVER": recover_level,
            }

            if args.send:
                pd.send(raw_payload)
                for symbol, value in pd_values.items():
                    pd.send_value(symbol, value)
                if vad != (0.0, 0.0, 0.0):
                    ue.send(raw_payload)

            # OSD
            status, color = anim.status_text()
            cv2.putText(frame, "COMMAND TEST", (20, 35),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"MODE: {status}", (20, 80),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2, cv2.LINE_AA)
            cv2.putText(frame,
                        f"VAD  V={payload['V']:+.2f}  A={payload['A']:+.2f}  D={payload['D']:+.2f}",
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame,
                        f"UE flower={payload['density']:.2f} decay={payload['decay']:.2f} "
                        f"fountain={payload['min_speed']:.0f}-{payload['max_speed']:.0f}",
                        (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2, cv2.LINE_AA)

            send_status = "SENDING" if args.send else "NOT SENDING"
            cv2.putText(frame, f"[{send_status}]", (20, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if args.send else (0, 0, 255), 2, cv2.LINE_AA)

            meter_x = max(20, frame.shape[1] - 310)
            draw_vad_meter(frame, "V", payload["V"], meter_x, 42, (80, 220, 255))
            draw_vad_meter(frame, "A", payload["A"], meter_x, 88, (80, 255, 120))
            draw_vad_meter(frame, "D", payload["D"], meter_x, 134, (255, 180, 80))

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                quit_event.set()
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                quit_event.set()
                break
    finally:
        ue.stop()
        pd.stop()
        cap.release()
        extractor.close()
        hand_extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
