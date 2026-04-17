import time
import msvcrt
from pathlib import Path

from src.capture import WebcamCapture
from src.eeg_profile import EmotionProfile
from src.gesture import InteractionEngine, InteractionSignals
from src.hand import HandEstimator
from src.pd_bridge import PDBridge
from src.pose import PoseEstimator
from src.session import SessionState
from src.ue_bridge import UEBridge
from src.vad_mapper import compute_unreal_payload


OPEN_READY_GRACE = 1.2
RISE_READY_GRACE = 0.7
GATHER_HOLD_GRACE = 0.18


def build_signals(dt: float, toggles: dict[str, float | bool | str], now: float) -> InteractionSignals:
    open_ready = now <= float(toggles["open_ready_until"])
    rise_ready = now <= float(toggles["rise_ready_until"])
    gather_ready = now <= float(toggles["gather_ready_until"])
    breath_ready = bool(toggles["breath_ready"])

    return InteractionSignals(
        dt=dt,
        open_ready=open_ready,
        open_branch=str(toggles["open_branch"]),
        open_spread=float(toggles["open_spread"]),
        open_release=bool(toggles["open_release"]),
        rise_ready=rise_ready,
        rise_height=float(toggles["rise_height"]),
        rise_release=bool(toggles["rise_release"]),
        gather_ready=gather_ready,
        gather_release=bool(toggles["gather_release"]),
        breath_ready=breath_ready,
        breath_release=bool(toggles["breath_release"]),
    )


def handle_keypress(
    toggles: dict[str, float | bool | str],
    key: bytes,
    now: float,
) -> tuple[bool, str]:
    if key in (b"q", b"Q"):
        return False, "quit"

    if key == b"1":
        toggles["open_branch"] = "open"
        return True, "branch=open"
    elif key == b"2":
        toggles["open_branch"] = "gather"
        return True, "branch=gather"
    elif key == b"3":
        toggles["open_branch"] = "missing"
        return True, "branch=missing"
    elif key in (b"b", b"B"):
        toggles["breath_ready"] = not bool(toggles["breath_ready"])
        return True, f"breath_ready={toggles['breath_ready']}"
    elif key in (b"n", b"N"):
        toggles["breath_release"] = True
        return True, "breath_release"
    elif key in (b"a", b"A", b"j", b"J"):
        toggles["open_spread"] = max(-1.0, float(toggles["open_spread"]) - 0.1)
        toggles["open_branch"] = "open"
        toggles["open_ready_until"] = now + OPEN_READY_GRACE
        toggles["open_release"] = False
        return True, f"open_spread={toggles['open_spread']:.2f}"
    elif key in (b"d", b"D", b"l", b"L"):
        toggles["open_spread"] = min(1.0, float(toggles["open_spread"]) + 0.1)
        toggles["open_branch"] = "open"
        toggles["open_ready_until"] = now + OPEN_READY_GRACE
        toggles["open_release"] = False
        return True, f"open_spread={toggles['open_spread']:.2f}"
    elif key in (b"p", b"P"):
        toggles["open_release"] = True
        return True, "open_release"
    elif key in (b"w", b"W", b"i", b"I"):
        toggles["rise_height"] = min(1.0, float(toggles["rise_height"]) + 0.1)
        toggles["rise_ready_until"] = now + RISE_READY_GRACE
        toggles["rise_release"] = False
        return True, f"rise_height={toggles['rise_height']:.2f}"
    elif key in (b"s", b"S", b"k", b"K"):
        toggles["rise_height"] = max(-1.0, float(toggles["rise_height"]) - 0.1)
        toggles["rise_ready_until"] = now + RISE_READY_GRACE
        toggles["rise_release"] = False
        return True, f"rise_height={toggles['rise_height']:.2f}"
    elif key in (b"-", b"_"):
        toggles["rise_height"] = max(-1.0, float(toggles["rise_height"]) - 0.1)
        toggles["rise_ready_until"] = now + RISE_READY_GRACE
        toggles["rise_release"] = False
        return True, f"rise_height={toggles['rise_height']:.2f}"
    elif key in (b"=", b"+"):
        toggles["rise_height"] = min(1.0, float(toggles["rise_height"]) + 0.1)
        toggles["rise_ready_until"] = now + RISE_READY_GRACE
        toggles["rise_release"] = False
        return True, f"rise_height={toggles['rise_height']:.2f}"
    elif key in (b"f", b"F"):
        toggles["rise_release"] = True
        return True, "rise_release"
    elif key == b" ":
        toggles["gather_ready_until"] = now + GATHER_HOLD_GRACE
        toggles["gather_release"] = False
        return True, "gather_hold"
    elif key in (b"c", b"C"):
        toggles["open_spread"] = 0.0
        toggles["rise_height"] = 0.0
        toggles["open_release"] = False
        toggles["rise_release"] = False
        toggles["gather_release"] = False
        toggles["breath_release"] = False
        toggles["open_branch"] = "missing"
        toggles["open_ready_until"] = 0.0
        toggles["rise_ready_until"] = 0.0
        toggles["gather_ready_until"] = 0.0
        toggles["breath_ready"] = False
        return True, "clear"

    return True, f"unmapped={key!r}"


def print_header() -> None:
    print("[interaction2] keyboard state-machine debug")
    print("A/D or J/L or Left/Right: open spread down/up (-1~1) | P: open end")
    print("W/S or I/K or Up/Down or -/=: rise down/up (-1~1) | F: rise end | Space hold: gather")
    print("1: open branch | 2: gather branch | 3: missing branch | B: breath toggle | N: breath release | C: clear | Q: quit")
    print("Logs are appended only when a keyboard input is received.")
    print()


def main() -> None:
    capture = WebcamCapture(camera_index=0, width=1280, height=720)
    pose_estimator = PoseEstimator()
    hand_estimator = HandEstimator()
    session = SessionState()
    engine = InteractionEngine()
    engine.open_hold_duration = 0.0
    engine.rise_hold_duration = 0.0
    ue_bridge = UEBridge()
    pd_bridge = PDBridge()
    ue_bridge.start()
    pd_bridge.start()

    toggles: dict[str, float | bool | str] = {
        "open_branch": "missing",
        "open_spread": 0.0,
        "open_ready_until": 0.0,
        "open_release": False,
        "rise_height": 0.0,
        "rise_ready_until": 0.0,
        "rise_release": False,
        "gather_ready_until": 0.0,
        "gather_release": False,
        "breath_ready": False,
        "breath_release": False,
    }

    _ = (capture, pose_estimator, hand_estimator, session)

    profile_path = Path(__file__).with_name("eeg_profile.json")
    if profile_path.exists():
        engine.set_profile(EmotionProfile.from_json(profile_path))
        print(f"[interaction2] loaded eeg profile: {profile_path}")
    else:
        print("[interaction2] using default emotion profile")

    running = True
    previous = time.time()
    print_header()

    while running:
        now = time.time()
        dt = max(0.001, now - previous)
        previous = now
        key_pressed = False

        while msvcrt.kbhit():
            key = msvcrt.getch()
            if key in (b"\xe0", b"\x00"):
                ext_key = msvcrt.getch()
                key_pressed = True
                if ext_key == b"K":
                    toggles["open_spread"] = max(-1.0, float(toggles["open_spread"]) - 0.1)
                    toggles["open_ready_until"] = now + OPEN_READY_GRACE
                    toggles["open_release"] = False
                elif ext_key == b"M":
                    toggles["open_spread"] = min(1.0, float(toggles["open_spread"]) + 0.1)
                    toggles["open_ready_until"] = now + OPEN_READY_GRACE
                    toggles["open_release"] = False
                elif ext_key == b"H":
                    toggles["rise_height"] = min(1.0, float(toggles["rise_height"]) + 0.1)
                    toggles["rise_ready_until"] = now + RISE_READY_GRACE
                    toggles["rise_release"] = False
                elif ext_key == b"P":
                    toggles["rise_height"] = max(-1.0, float(toggles["rise_height"]) - 0.1)
                    toggles["rise_ready_until"] = now + RISE_READY_GRACE
                    toggles["rise_release"] = False
                print(f"[key] extended={ext_key!r} open={toggles['open_spread']:.2f} rise={toggles['rise_height']:.2f}")
                continue
            key_pressed = True
            running, key_info = handle_keypress(toggles, key, now)
            print(f"[key] raw={key!r} {key_info}")
            if not running:
                break

        if now > float(toggles["gather_ready_until"]) and engine.interaction.gather.phase == "active":
            toggles["gather_release"] = True

        signals = build_signals(dt, toggles, now)
        events = engine.update(signals)
        payload_obj = compute_unreal_payload(engine.emotion)
        payload = payload_obj.as_dict()

        if engine.is_interacting:
            ue_bridge.send(payload_obj)
            pd_bridge.send(payload_obj)

        if key_pressed:
            print(
                "[emotion] "
                f"V={engine.emotion.valence:.3f} "
                f"A={engine.emotion.arousal:.3f} "
                f"D={engine.emotion.dominance:.3f} | "
                f"open={engine.interaction.open.amount:.2f} "
                f"rise={engine.interaction.rise.amount:.2f} | "
                f"interacting={engine.is_interacting} | "
                f"events={events.triggered if events.triggered else []} | "
                f"payload={payload}"
            )

        toggles["open_release"] = False
        toggles["rise_release"] = False
        toggles["gather_release"] = False
        toggles["breath_release"] = False

        time.sleep(0.05)

    ue_bridge.stop()
    pd_bridge.stop()


if __name__ == "__main__":
    main()
