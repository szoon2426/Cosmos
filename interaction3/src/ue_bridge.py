import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import requests

if __package__ in (None, ""):
    from vad_mapper import UnrealPayload
else:
    from .vad_mapper import UnrealPayload


BASE_URL = "http://localhost:30010"
PRESET_NAME = "NewRemoteControlPreset"
PROPERTY_URL_TEMPLATE = f"{BASE_URL}/remote/preset/{PRESET_NAME}/property/{{property_name}}"


@dataclass
class UEBridge:
    enabled: bool = True
    timeout_sec: float = 0.5
    session: requests.Session = field(default_factory=requests.Session)
    executor: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(max_workers=1))

    def start(self) -> None:
        print(f"[interaction3] UEBridge ready -> {BASE_URL}")
        print(f"[interaction3] UE preset -> {PRESET_NAME}")

    def stop(self) -> None:
        self.executor.shutdown(wait=False)
        self.session.close()
        print("[interaction3] UEBridge stopped")

    def send(self, payload: UnrealPayload) -> None:
        if not self.enabled:
            print(f"[interaction3] UE payload -> {payload.as_dict()}")
            return

        # 백그라운드 스레드에서 HTTP 요청 처리 (카메라 화면 렉 방지)
        def _worker():
            for property_name, value in payload.as_preset_properties().items():
                self._set_property(property_name, value)
                
        self.executor.submit(_worker)

    def _set_property(self, property_name: str, value: float) -> None:
        url = PROPERTY_URL_TEMPLATE.format(property_name=property_name)
        body = {"PropertyValue": value}
        try:
            response = self.session.put(url, json=body, timeout=self.timeout_sec)
            if response.status_code != 200:
                print(f"[interaction3] UE ERR {property_name}: {response.status_code} {response.text[:120]}")
        except requests.exceptions.Timeout:
            print(f"[interaction3] UE timeout -> {property_name}")
        except requests.exceptions.ConnectionError:
            pass # 언리얼 안 켜져있을 때 에러 도배 방지


