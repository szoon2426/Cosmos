from __future__ import annotations

import socket
from dataclasses import dataclass

if __package__ in (None, ""):
    from vad_mapper import UnrealPayload
else:
    from .vad_mapper import UnrealPayload


@dataclass
class PDBridge:
    host: str = "127.0.0.1"
    port: int = 30025
    enabled: bool = True
    sock: socket.socket | None = None

    def start(self) -> None:
        print(f"[interaction3] PDBridge ready -> tcp://{self.host}:{self.port}")

    def stop(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None
        print("[interaction3] PDBridge stopped")

    def send(self, payload: UnrealPayload) -> None:
        if not self.enabled:
            return
        self.send_value("V", payload.v)
        self.send_value("A", payload.a)
        self.send_value("D", payload.d)
        
    def send_trigger(self, symbol: str) -> None:
        if not self.enabled:
            return
        self._send_message(f"{symbol} bang;")

    def send_value(self, symbol: str, value: float) -> None:
        if not self.enabled:
            return
        self._send_value(symbol, value)

    def _send_value(self, symbol: str, value: float) -> None:
        self._send_message(f"{symbol} {value:.6f};")

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
            print(f"[interaction3] PD send error -> {message.strip()}: {exc}")
