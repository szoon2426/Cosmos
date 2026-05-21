from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from windows_world_pipeline.eeg_interpreter.generate_world_instruction import generate_instruction_from_eeg, load_json as load_eeg_json
from windows_world_pipeline.world_spawn.build_world_spawn import build_planet_spawn, build_world_spawn, ensure_vad_footprint, load_json, resolve_path, upsert_by_key, update_person_world_map, write_json


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_config_path(config_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (config_dir / path).resolve()


def file_is_stable(path: Path, stable_wait_sec: float) -> bool:
    try:
        size_a = path.stat().st_size
        time.sleep(stable_wait_sec)
        size_b = path.stat().st_size
    except FileNotFoundError:
        return False
    return size_a == size_b


def load_state(path: Path) -> dict[str, Any]:
    return read_json(path, {"processed": {}, "errors": {}}) or {"processed": {}, "errors": {}}


def save_state(path: Path, state: dict[str, Any]) -> None:
    write_json(path, state)


def file_signature(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def already_processed(state: dict[str, Any], path: Path) -> bool:
    record = state.get("processed", {}).get(str(path.resolve()))
    return bool(record and record.get("signature") == file_signature(path))


def mark_processed(
    state: dict[str, Any],
    path: Path,
    world_id: str,
    instruction_path: Path,
    world_path: Path,
    planet_path: Path,
    vad_footprint_path: Path,
) -> None:
    state.setdefault("processed", {})[str(path.resolve())] = {
        "signature": file_signature(path),
        "world_id": world_id,
        "instruction_path": str(instruction_path),
        "world_path": str(world_path),
        "planet_path": str(planet_path),
        "vad_footprint_path": str(vad_footprint_path),
        "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    state.get("errors", {}).pop(str(path.resolve()), None)


def mark_skipped_existing(state: dict[str, Any], path: Path) -> None:
    state.setdefault("processed", {})[str(path.resolve())] = {
        "signature": file_signature(path),
        "world_id": "",
        "instruction_path": "",
        "world_path": "",
        "planet_path": "",
        "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "note": "existing file skipped on watcher start",
    }


def mark_error(state: dict[str, Any], path: Path, error: Exception) -> None:
    state.setdefault("errors", {})[str(path.resolve())] = {
        "signature": file_signature(path) if path.exists() else "",
        "error": str(error),
        "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


class UnrealRemote:
    def __init__(self, config: dict[str, Any]) -> None:
        self.enabled = bool(config.get("enabled", True))
        self.base_url = str(config.get("base_url", "http://localhost:30010")).rstrip("/")
        self.preset_name = str(config.get("preset_name", "RCP_WorldVariable"))
        self.function_display_name = str(config.get("function_display_name", "Spawn Planet"))
        self.fallback_function_names = list(config.get("fallback_function_names", ["SpawnPlanet"]))
        self.parameters = dict(config.get("parameters", {}))
        self.timeout_sec = float(config.get("timeout_sec", 8.0))
        self.lookup_retries = int(config.get("lookup_retries", 3))
        self.retry_wait_sec = float(config.get("retry_wait_sec", 0.4))
        self.session = requests.Session()
        self._function_cache: tuple[str, str] | None = None

    def close(self) -> None:
        self.session.close()

    def trigger_spawn_planet(self) -> bool:
        if not self.enabled:
            print("[world_pipeline] remote disabled; skipped SpawnPlanet")
            return True

        resolved = self._resolve_function()
        if resolved is None:
            print("[world_pipeline] remote function not found: Spawn Planet / SpawnPlanet")
            return False

        object_path, function_name = resolved
        body = {
            "objectPath": object_path,
            "functionName": function_name,
            "parameters": self.parameters,
            "generateTransaction": True,
        }
        try:
            response = self.session.put(f"{self.base_url}/remote/object/call", json=body, timeout=self.timeout_sec)
            if response.status_code == 404:
                response = self.session.post(f"{self.base_url}/remote/object/call", json=body, timeout=self.timeout_sec)
            if response.status_code != 200:
                print(f"[world_pipeline] remote SpawnPlanet failed: {response.status_code} {response.text[:200]}")
                return False
            print("[world_pipeline] remote SpawnPlanet triggered")
            return True
        except requests.exceptions.RequestException as exc:
            print(f"[world_pipeline] remote connection failed: {exc}")
            return False

    def _resolve_function(self) -> tuple[str, str] | None:
        if self._function_cache is not None:
            return self._function_cache

        names = {self.function_display_name, *self.fallback_function_names}
        preset: dict[str, Any] = {}
        for attempt in range(1, self.lookup_retries + 1):
            try:
                response = self.session.get(f"{self.base_url}/remote/preset/{self.preset_name}", timeout=self.timeout_sec)
                if response.status_code != 200:
                    print(f"[world_pipeline] remote preset lookup failed: {response.status_code} {response.text[:200]}")
                else:
                    preset = response.json().get("Preset", {})
                    break
            except requests.exceptions.RequestException as exc:
                print(f"[world_pipeline] remote preset lookup error {attempt}/{self.lookup_retries}: {exc}")

            if attempt < self.lookup_retries:
                time.sleep(self.retry_wait_sec)

        if not preset:
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
                self._function_cache = (owners[0]["Path"], underlying_name)
                print(f"[world_pipeline] remote function resolved: {display_name} -> {underlying_name}")
                return self._function_cache
        return None


def build_outputs(instruction: dict[str, Any], world_config_path: Path) -> tuple[Path, Path, Path, Path]:
    world_config_dir = world_config_path.parent
    world_config = read_json(world_config_path, {}) or {}
    world_id = str(instruction["world_id"])

    world_spawn_dir = resolve_path(world_config_dir, world_config.get("world_spawn_dir", "../world_spawn_json"))
    planet_spawn_dir = resolve_path(world_config_dir, world_config.get("planet_spawn_dir", "../planet_spawn"))
    person_map_path = resolve_path(world_config_dir, world_config.get("person_world_map_path", "../person_world_map.json"))
    vad_footprint_dir = resolve_path(world_config_dir, world_config.get("vad_footprint_dir", "../vad_footprint"))
    min_planet_distance = float(world_config.get("planet_min_distance", 900.0))

    world = build_world_spawn(instruction, int(world_config.get("world_space", 2500)))
    world_path = world_spawn_dir / f"{world_id}.json"
    vad_footprint_path = vad_footprint_dir / f"{world_id}.json"
    planet_layout_path = planet_spawn_dir / "planet_layout.json"
    planets_layout_path = planet_spawn_dir / "planets_layout.json"

    planets_doc = load_json(planets_layout_path, {"planets": []}) or {"planets": []}
    existing_planets = list(planets_doc.get("planets", []))
    planet = build_planet_spawn(instruction, existing_planets, min_planet_distance)
    planets_doc["planets"] = upsert_by_key(existing_planets, "planet_id", world_id, planet)

    write_json(world_path, world)
    write_json(planet_layout_path, planet)
    write_json(planets_layout_path, planets_doc)
    ensure_vad_footprint(vad_footprint_path, world)
    update_person_world_map(person_map_path, world.get("person_name", "unknown"), world_id)

    return world_path, planet_layout_path, planets_layout_path, vad_footprint_path


def process_eeg_file(path: Path, config: dict[str, Any], config_dir: Path, remote: UnrealRemote) -> tuple[str, Path, Path, Path]:
    mac_config_path = resolve_config_path(config_dir, config.get("eeg_interpreter_config", config.get("mac_config", "./eeg_interpreter/config.example.json")))
    world_config_path = resolve_config_path(config_dir, config["world_spawn_config"])
    instruction_output_dir = str(resolve_config_path(config_dir, config["instruction_output_dir"]))

    mac_config = read_json(mac_config_path, {}) or {}
    eeg = load_eeg_json(path)
    instruction, instruction_path = generate_instruction_from_eeg(
        mac_config,
        mac_config_path.parent,
        eeg,
        instruction_output_dir,
    )
    world_path, planet_path, _planets_path, vad_footprint_path = build_outputs(instruction, world_config_path)
    remote.trigger_spawn_planet()
    return str(instruction["world_id"]), instruction_path, world_path, vad_footprint_path


def scan_once(config: dict[str, Any], config_dir: Path, state: dict[str, Any], remote: UnrealRemote) -> bool:
    watch_dir = resolve_config_path(config_dir, config["eeg_watch_dir"])
    stable_wait_sec = float(config.get("stable_wait_sec", 0.5))
    found_work = False

    watch_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(watch_dir.glob("*.json")):
        if path.name.startswith("."):
            continue
        if already_processed(state, path):
            continue
        if not file_is_stable(path, stable_wait_sec):
            continue

        found_work = True
        print(f"[world_pipeline] processing EEG -> {path}")
        try:
            world_id, instruction_path, world_path, vad_footprint_path = process_eeg_file(path, config, config_dir, remote)
            planet_path = resolve_config_path(config_dir, "../planet_spawn/planet_layout.json")
            mark_processed(state, path, world_id, instruction_path, world_path, planet_path, vad_footprint_path)
            print(f"[world_pipeline] done {path.name} -> {world_id}")
        except Exception as exc:
            mark_error(state, path, exc)
            print(f"[world_pipeline] error {path.name}: {exc}")

    return found_work


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch eeg_json and automatically build Cosmos world/planet spawn JSON.")
    parser.add_argument("--config", default="windows_world_pipeline/config.example.json")
    parser.add_argument("--once", action="store_true", help="Scan once and exit.")
    parser.add_argument("--reset-state", action="store_true", help="Forget processed EEG files before scanning.")
    parser.add_argument("--process-existing", action="store_true", help="Process JSON files that already exist in eeg_json.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config_dir = config_path.parent
    config = read_json(config_path, {}) or {}
    state_path = resolve_config_path(config_dir, config["state_path"])
    state = {"processed": {}, "errors": {}} if args.reset_state else load_state(state_path)
    remote = UnrealRemote(config.get("remote_control", {}))
    watch_dir = resolve_config_path(config_dir, config["eeg_watch_dir"])

    process_existing = args.process_existing or bool(config.get("process_existing_on_start", False))
    if not process_existing and not state_path.exists() and not args.once:
        watch_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(watch_dir.glob("*.json")):
            mark_skipped_existing(state, path)
        save_state(state_path, state)
        print("[world_pipeline] existing EEG files marked as already seen")

    print(f"[world_pipeline] watching -> {watch_dir}")
    print(f"[world_pipeline] state -> {state_path}")
    print("[world_pipeline] Ctrl+C to stop")

    try:
        while True:
            scan_once(config, config_dir, state, remote)
            save_state(state_path, state)
            if args.once:
                break
            time.sleep(float(config.get("poll_interval_sec", 2.0)))
    except KeyboardInterrupt:
        print("\n[world_pipeline] stopped")
    finally:
        save_state(state_path, state)
        remote.close()


if __name__ == "__main__":
    main()
