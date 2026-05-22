from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import requests

if __package__ in (None, ""):
    from final_mapper import FinalInteractionPayload
else:
    from .final_mapper import FinalInteractionPayload


BASE_URL = "http://localhost:30010"
PRESET_NAME = "RCP_WorldVariable"
PROPERTY_URL_TEMPLATE = f"{BASE_URL}/remote/preset/{PRESET_NAME}/property/{{property_name}}"
SWITCH_WORLD_ID = "galaxy"
POINTER_PROPERTIES = {
    "L_Grip",
    "R_Grip",
    "L_Y Location",
    "L_ Z Location",
    "R_Y Location",
    "R_ Z Location",
}
PROPERTY_PRIORITY = (
    "L_Y Location",
    "L_ Z Location",
    "R_Y Location",
    "R_ Z Location",
    "L_Grip",
    "R_Grip",
    "Target V",
    "Target A",
    "Target D",
    "Flower Density",
    "decay",
    "Min Speed",
    "Max Speed",
)


@dataclass
class FinalUEBridge:
    enabled: bool = True
    timeout_sec: float = 0.5
    session: requests.Session = field(default_factory=requests.Session)
    executor: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(max_workers=1))
    lock: threading.Lock = field(default_factory=threading.Lock)
    pending_payload: FinalInteractionPayload | None = None
    pending_switch_world_id: str | None = None
    worker_running: bool = False
    switch_function_cache: tuple[str, str] | None = None
    last_sent_values: dict[str, float] = field(default_factory=dict)

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
            if payload.switch_to_camera > 0.5:
                self.pending_switch_world_id = self._payload_switch_world_id(payload)
            if self.worker_running:
                return
            self.worker_running = True

        self.executor.submit(self._drain_worker)

    def _drain_worker(self) -> None:
        while True:
            with self.lock:
                payload = self.pending_payload
                switch_world_id = self.pending_switch_world_id
                self.pending_payload = None
                self.pending_switch_world_id = None
                if payload is None and switch_world_id is None:
                    self.worker_running = False
                    return

            if payload is not None:
                for property_name, value in self._ordered_properties(payload):
                    if not self._set_property(property_name, value):
                        break

            if switch_world_id is not None:
                self.switch_to_world(switch_world_id)

    @staticmethod
    def _payload_switch_world_id(payload: FinalInteractionPayload) -> str:
        world_id = (payload.switch_world_id or "").strip()
        return world_id or SWITCH_WORLD_ID

    def _set_property(self, property_name: str, value: float) -> bool:
        previous = self.last_sent_values.get(property_name)
        if previous is not None and abs(previous - value) < self._change_threshold(property_name):
            return True

        url = PROPERTY_URL_TEMPLATE.format(property_name=quote(property_name, safe=""))
        try:
            response = self.session.put(url, json={"PropertyValue": value}, timeout=self.timeout_sec)
            if response.status_code != 200:
                print(
                    f"[interaction3-final] UE ERR {property_name}: "
                    f"{response.status_code} {response.text[:120]}"
                )
            else:
                self.last_sent_values[property_name] = value
            return True
        except requests.exceptions.Timeout:
            print(f"[interaction3-final] UE timeout -> {property_name}")
            return False
        except requests.exceptions.ConnectionError:
            return False

    def _ordered_properties(self, payload: FinalInteractionPayload) -> list[tuple[str, float]]:
        properties = payload.as_preset_properties()
        ordered: list[tuple[str, float]] = []
        for name in PROPERTY_PRIORITY:
            if name in properties:
                ordered.append((name, properties.pop(name)))
        ordered.extend(properties.items())
        return ordered

    @staticmethod
    def _change_threshold(property_name: str) -> float:
        if property_name in POINTER_PROPERTIES:
            return 0.15
        return 0.002

    def get_property(self, property_name: str) -> Any | None:
        if not self.enabled:
            return None

        url = PROPERTY_URL_TEMPLATE.format(property_name=quote(property_name, safe=""))
        try:
            response = self.session.get(url, timeout=self.timeout_sec)
            if response.status_code != 200:
                return None
            return self._extract_property_value(response.json())
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, ValueError):
            return None

    def get_current_world_id(self) -> str | None:
        for property_name in (
            "CurrentWorldId",
            "Current World Id",
            "Current World ID",
            "CurrentWorldID",
            "World ID",
            "WorldID",
        ):
            value = self.get_property(property_name)
            if value is None:
                continue
            world_id = str(value).strip()
            if world_id:
                return world_id
        return None

    def switch_to_world(self, world_id: str) -> bool:
        resolved = self._resolve_switch_function()
        if resolved is None:
            print("[interaction3-final] UE switch function not found")
            return False

        object_path, function_name = resolved
        body = {
            "objectPath": object_path,
            "functionName": function_name,
            "parameters": {"WorldID": world_id},
            "generateTransaction": True,
        }

        try:
            response = self.session.put(f"{BASE_URL}/remote/object/call", json=body, timeout=self.timeout_sec)
            if response.status_code == 404:
                response = self.session.post(f"{BASE_URL}/remote/object/call", json=body, timeout=self.timeout_sec)
            if response.status_code != 200:
                print(
                    f"[interaction3-final] UE switch failed: "
                    f"{response.status_code} {response.text[:160]}"
                )
                return False
            print(f"[interaction3-final] UE switch -> {world_id}")
            return True
        except requests.exceptions.Timeout:
            print("[interaction3-final] UE switch timeout")
        except requests.exceptions.ConnectionError:
            pass
        return False

    def _resolve_switch_function(self) -> tuple[str, str] | None:
        if self.switch_function_cache is not None:
            return self.switch_function_cache

        names = {"Switch to World", "SwitchToWorld"}
        try:
            response = self.session.get(f"{BASE_URL}/remote/preset/{PRESET_NAME}", timeout=self.timeout_sec)
            if response.status_code != 200:
                return None
            preset = response.json().get("Preset", {})
        except (requests.exceptions.RequestException, ValueError):
            return None

        for group in preset.get("Groups", []):
            for function in group.get("ExposedFunctions", []):
                display_name = function.get("DisplayName", "")
                underlying = function.get("UnderlyingFunction", {})
                underlying_name = underlying.get("Name", "")
                if display_name not in names and underlying_name not in names:
                    continue
                owners = function.get("OwnerObjects", [])
                if not owners:
                    return None
                self.switch_function_cache = (owners[0]["Path"], underlying_name)
                print(f"[interaction3-final] UE switch resolved: {display_name} -> {underlying_name}")
                return self.switch_function_cache

        return None

    @staticmethod
    def _extract_property_value(data: Any) -> Any | None:
        if not isinstance(data, dict):
            return data

        for key in ("PropertyValue", "propertyValue", "Value"):
            if key in data:
                return data[key]

        property_values = data.get("PropertyValues")
        if isinstance(property_values, list) and property_values:
            first_value = property_values[0]
            if isinstance(first_value, dict):
                return FinalUEBridge._extract_property_value(first_value)
            return first_value

        value = data.get("value")
        if isinstance(value, dict):
            return FinalUEBridge._extract_property_value(value)
        if value is not None:
            return value

        return None
