from __future__ import annotations

import socket
from dataclasses import dataclass, field

from src.vad_mapper import UnrealPayload


@dataclass
class PDBridge:
    host: str = "127.0.0.1"
    port: int = 30021
    enabled: bool = True
    sock: socket.socket = field(default_factory=lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM))

    def start(self) -> None:
        print(f"[interaction2] PDBridge ready -> udp://{self.host}:{self.port}")

    def stop(self) -> None:
        self.sock.close()
        print("[interaction2] PDBridge stopped")

    def send(self, payload: UnrealPayload) -> None:
        if not self.enabled:
            return
        self._send_value("V", payload.v)
        self._send_value("A", payload.a)
        self._send_value("D", payload.d)
        self._send_value("Fountain", max(0.0, min(1.0, payload.max_speed / 600.0)))

    def _send_value(self, symbol: str, value: float) -> None:
        message = f"{symbol} {value:.6f};".encode("utf-8")
        try:
            self.sock.sendto(message, (self.host, self.port))
        except OSError as exc:
            print(f"[interaction2] PD send error -> {symbol}: {exc}")
