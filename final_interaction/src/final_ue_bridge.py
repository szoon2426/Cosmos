from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import requests

if __package__ in (None, ""):
    from final_mapper import FinalInteractionPayload
else:
    from .final_mapper import FinalInteractionPayload


BASE_URL = "http://localhost:30010"
PRESET_NAME = "NewRemoteControlPreset"
PROPERTY_URL_TEMPLATE = f"{BASE_URL}/remote/preset/{PRESET_NAME}/property/{{property_name}}"


@dataclass
class FinalUEBridge:
    enabled: bool = True
    timeout_sec: float = 0.5
    session: requests.Session = field(default_factory=requests.Session)
    executor: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(max_workers=1))
    lock: threading.Lock = field(default_factory=threading.Lock)
    pending_payload: FinalInteractionPayload | None = None
    worker_running: bool = False

    def start(self) -> None:
        print(f"[interaction3-final] UEBridge ready -> {BASE_URL}")
        print(f"[interaction3-final] UE preset -> {PRESET_NAME}")

    def stop(self) -> None:
        self.executor.shutdown(wait=False)
        self.session.close()
        print("[interaction3-final] UEBridge stopped")

    def send(self, payload: FinalInteractionPayload) -> None:
        if not self.enabled:
            return

        with self.lock:
            self.pending_payload = payload
            if self.worker_running:
                return
            self.worker_running = True

        self.executor.submit(self._drain_worker)

    def _drain_worker(self) -> None:
        while True:
            with self.lock:
                payload = self.pending_payload
                self.pending_payload = None
                if payload is None:
                    self.worker_running = False
                    return

            for property_name, value in payload.as_preset_properties().items():
                self._set_property(property_name, value)

    def _set_property(self, property_name: str, value: float) -> None:
        url = PROPERTY_URL_TEMPLATE.format(property_name=property_name)
        try:
            response = self.session.put(url, json={"PropertyValue": value}, timeout=self.timeout_sec)
            if response.status_code != 200:
                print(
                    f"[interaction3-final] UE ERR {property_name}: "
                    f"{response.status_code} {response.text[:120]}"
                )
        except requests.exceptions.Timeout:
            print(f"[interaction3-final] UE timeout -> {property_name}")
        except requests.exceptions.ConnectionError:
            pass
