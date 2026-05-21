from __future__ import annotations

import socket
from dataclasses import dataclass

if __package__ in (None, ""):
    from final_mapper import FinalInteractionPayload
else:
    from .final_mapper import FinalInteractionPayload


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(slots=True)
class FinalPDBridge:
    host: str = "127.0.0.1"
    port: int = 30025
    enabled: bool = True
    sock: socket.socket | None = None

    def start(self) -> None:
        print(f"[final_interaction] PDBridge ready -> tcp://{self.host}:{self.port}")

    def stop(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None
        print("[final_interaction] PDBridge stopped")

    def send_payload(self, payload: FinalInteractionPayload) -> None:
        if not self.enabled:
            return

        # Keep the existing cosmos_sound world controls alive while final interaction runs.
        fountain_value = clamp(((payload.min_speed + payload.max_speed) * 0.5) / 600.0, 0.0, 1.0)
        self.send_value("V", payload.target_v)
        self.send_value("A", payload.target_a)
        self.send_value("D", payload.target_d)
        self.send_value("W", payload.target_a)
        self.send_value("F", fountain_value * 2.0 - 1.0)

    def send_trigger(self, symbol: str) -> None:
        if not self.enabled:
            return
        self._send_message(f"{symbol} bang;")

    def send_value(self, symbol: str, value: float) -> None:
        if not self.enabled:
            return
        self._send_message(f"{symbol} {value:.6f};")

    def set_world_mode(self) -> None:
        self.send_value("WORLD_MODE", 1.0)

    def set_space_mode(self) -> None:
        self.send_value("SPACE_MODE", 1.0)

    def trigger_ready_on(self) -> None:
        self.send_trigger("READY_ON")

    def trigger_ready_off(self) -> None:
        self.send_trigger("READY_OFF")

    def _connect(self) -> socket.socket:
        if self.sock is None:
            self.sock = socket.create_connection((self.host, self.port), timeout=1.0)
        return self.sock

    def _send_message(self, message: str) -> None:
        try:
            self._connect().sendall(message.encode("utf-8"))
        except OSError as exc:
            if self.sock is not None:
                self.sock.close()
                self.sock = None
            print(f"[final_interaction] PD send error -> {message.strip()}: {exc}")
