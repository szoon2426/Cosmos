from dataclasses import dataclass, field

import requests

from src.vad_mapper import UnrealPayload


BASE_URL = "http://localhost:30010"
PRESET_NAME = "NewRemoteControlPreset"
PROPERTY_URL_TEMPLATE = f"{BASE_URL}/remote/preset/{PRESET_NAME}/property/{{property_name}}"


@dataclass
class UEBridge:
    enabled: bool = True
    timeout_sec: float = 0.5
    session: requests.Session = field(default_factory=requests.Session)

    def start(self) -> None:
        print(f"[interaction2] UEBridge ready -> {BASE_URL}")
        print(f"[interaction2] UE preset -> {PRESET_NAME}")

    def stop(self) -> None:
        self.session.close()
        print("[interaction2] UEBridge stopped")

    def send(self, payload: UnrealPayload) -> None:
        if not self.enabled:
            print(f"[interaction2] UE payload -> {payload.as_dict()}")
            return

        for property_name, value in payload.as_preset_properties().items():
            self._set_property(property_name, value)

    def _set_property(self, property_name: str, value: float) -> None:
        url = PROPERTY_URL_TEMPLATE.format(property_name=property_name)
        body = {"PropertyValue": value}

        try:
            response = self.session.put(url, json=body, timeout=self.timeout_sec)
            if response.status_code != 200:
                print(f"[interaction2] UE ERR {property_name}: {response.status_code} {response.text[:120]}")
        except requests.exceptions.Timeout:
            print(f"[interaction2] UE timeout -> {property_name}")
        except requests.exceptions.ConnectionError:
            print(f"[interaction2] UE connection error -> {property_name}")
