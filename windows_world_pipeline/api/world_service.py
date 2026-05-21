from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from windows_world_pipeline.api.models import EEGEmotionsPayload, GenerateWorldResponse
from windows_world_pipeline.eeg_interpreter.generate_world_instruction import (
    build_llm_prompt,
    call_gemini,
    dry_run_interpretation,
    normalize_eeg_payload,
    normalize_instruction,
)
from windows_world_pipeline.run_pipeline import UnrealRemote, build_outputs, read_json, resolve_config_path
from windows_world_pipeline.world_spawn.build_world_spawn import resolve_path as resolve_world_path


class DuplicateWorldError(RuntimeError):
    pass


class WorldService:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path.resolve()
        self.config_dir = self.config_path.parent
        self.config = read_json(self.config_path, {}) or {}
        self.eeg_config_path = resolve_config_path(
            self.config_dir,
            self.config.get("eeg_interpreter_config", self.config.get("mac_config", "./eeg_interpreter/config.example.json")),
        )
        self.world_config_path = resolve_config_path(self.config_dir, self.config["world_spawn_config"])
        self.eeg_config = read_json(self.eeg_config_path, {}) or {}
        self.world_config = read_json(self.world_config_path, {}) or {}
        self.instruction_output_dir = resolve_config_path(self.config_dir, self.config["instruction_output_dir"])
        self.world_spawn_dir = resolve_world_path(
            self.world_config_path.parent,
            self.world_config.get("world_spawn_dir", "../world_spawn_json"),
        )
        self.planet_spawn_dir = resolve_world_path(
            self.world_config_path.parent,
            self.world_config.get("planet_spawn_dir", "../planet_spawn"),
        )
        self.vad_footprint_dir = resolve_world_path(
            self.world_config_path.parent,
            self.world_config.get("vad_footprint_dir", "../vad_footprint"),
        )
        self.remote = UnrealRemote(self.config.get("remote_control", {}))

    def close(self) -> None:
        self.remote.close()

    def output_paths_for(self, world_id: str) -> dict[str, Path]:
        return {
            "instruction_path": self.instruction_output_dir / f"{world_id}.json",
            "world_spawn_path": self.world_spawn_dir / f"{world_id}.json",
            "vad_footprint_path": self.vad_footprint_dir / f"{world_id}.json",
        }

    def world_exists(self, world_id: str) -> bool:
        if any(path.exists() for path in self.output_paths_for(world_id).values()):
            return True
        planets_doc = read_json(self.planet_spawn_dir / "planets_layout.json", {"planets": []}) or {"planets": []}
        return any(planet.get("planet_id") == world_id for planet in planets_doc.get("planets", []))

    def validate_payload(self, payload: EEGEmotionsPayload) -> None:
        if payload.status != "succeeded":
            raise ValueError("Only succeeded result payloads can generate worlds")
        if payload.result is None:
            raise ValueError("Succeeded payloads must include result")

    def generate(self, world_id: str, payload: EEGEmotionsPayload) -> GenerateWorldResponse:
        self.validate_payload(payload)
        if self.world_exists(world_id):
            raise DuplicateWorldError(f"World '{world_id}' already exists")

        eeg = normalize_eeg_payload(payload_to_eeg_dict(payload))
        interpreted = self._interpret(eeg)
        world_number = deterministic_world_number(world_id)
        instruction = normalize_instruction(interpreted, world_number, eeg)
        instruction["world_id"] = world_id
        instruction["world_number"] = world_number

        paths = self.output_paths_for(world_id)
        if self.world_exists(world_id):
            raise DuplicateWorldError(f"World '{world_id}' already exists")

        write_new_json(paths["instruction_path"], instruction)
        world_path, planet_path, planets_path, vad_footprint_path = build_outputs(instruction, self.world_config_path)
        self.remote.trigger_spawn_planet()

        return GenerateWorldResponse(
            world_id=world_id,
            instruction_path=str(paths["instruction_path"]),
            world_spawn_path=str(world_path),
            planet_layout_path=str(planet_path),
            planets_layout_path=str(planets_path),
            vad_footprint_path=str(vad_footprint_path),
            created_at=str(instruction.get("created_at") or datetime.now(timezone.utc).isoformat()),
        )

    def _interpret(self, eeg: dict[str, Any]) -> dict[str, Any]:
        if self.eeg_config.get("llm", {}).get("dry_run", True):
            return dry_run_interpretation(eeg)
        return call_gemini(self.eeg_config, build_llm_prompt(eeg))


def deterministic_world_number(world_id: str) -> int:
    digest = hashlib.sha256(world_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 9999 + 1


def payload_to_eeg_dict(payload: EEGEmotionsPayload) -> dict[str, Any]:
    data = payload.model_dump(mode="json")
    data["recording"] = data.get("recording") or {}
    result = data.get("result") or {}
    result["resting_state_features"] = result.get("resting_state_features") or {}
    data["result"] = result
    return data


def write_new_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except FileExistsError as exc:
        raise DuplicateWorldError(f"World '{path.stem}' already exists") from exc
