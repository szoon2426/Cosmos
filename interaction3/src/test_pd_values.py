from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    from pd_bridge import PDBridge
else:
    from .pd_bridge import PDBridge


def _sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def send_value(pd: PDBridge, symbol: str, value: float, hold: float) -> None:
    print(f"[PD TEST] {symbol} -> {value:.3f}")
    pd.send_value(symbol, value)
    _sleep(hold)


def send_trigger(pd: PDBridge, symbol: str, hold: float) -> None:
    print(f"[PD TEST] {symbol}")
    pd.send_trigger(symbol)
    _sleep(hold)


def run_sequence(pd: PDBridge, hold: float) -> None:
    print("[PD TEST] sequence start")

    send_trigger(pd, "READY_ON", hold)
    send_value(pd, "READY_MODE", 1.0, hold)

    # Wind crossfade check: low -> mid -> high -> back low
    send_value(pd, "W", -1.0, hold)
    send_value(pd, "W", 0.0, hold)
    send_value(pd, "W", 1.0, hold)
    send_value(pd, "W", -1.0, hold)

    # Fountain crossfade check: low -> mid -> high -> back low
    send_value(pd, "F", -1.0, hold)
    send_value(pd, "F", 0.0, hold)
    send_value(pd, "F", 1.0, hold)
    send_value(pd, "F", -1.0, hold)

    # Ambient loop tone check
    send_value(pd, "V", -1.0, hold)
    send_value(pd, "V", 1.0, hold)
    send_value(pd, "A", -1.0, hold)
    send_value(pd, "A", 1.0, hold)
    send_value(pd, "D", -1.0, hold)
    send_value(pd, "D", 1.0, hold)
    send_value(pd, "V", 0.0, hold)
    send_value(pd, "A", 0.0, hold)
    send_value(pd, "D", 0.0, hold)

    # Ready off check
    send_trigger(pd, "READY_OFF", hold)
    send_value(pd, "READY_MODE", 0.0, hold)

    print("[PD TEST] sequence done")


def run_loop(pd: PDBridge, hold: float) -> None:
    try:
        while True:
            run_sequence(pd, hold)
            _sleep(max(hold, 0.5))
    except KeyboardInterrupt:
        print("\n[PD TEST] stopped by user")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send PD test values to cosmos_sound2.pd one by one (READY/VAD/W/F only)"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30025)
    parser.add_argument("--hold", type=float, default=2.0, help="Seconds to wait after each sent value")
    parser.add_argument("--loop", action="store_true", help="Repeat the full sequence until interrupted")
    args = parser.parse_args()

    pd = PDBridge(host=args.host, port=args.port, enabled=True)
    pd.start()
    try:
        if args.loop:
            run_loop(pd, args.hold)
        else:
            run_sequence(pd, args.hold)
    finally:
        pd.stop()


if __name__ == "__main__":
    main()
