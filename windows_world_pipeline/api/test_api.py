from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from windows_world_pipeline.api.main import app
from windows_world_pipeline.eeg_interpreter.generate_world_instruction import next_world_number


def sample_payload() -> dict[str, Any]:
    return {
        "job_id": "job-001",
        "status": "succeeded",
        "created_at": "2026-05-21T10:00:00+00:00",
        "updated_at": "2026-05-21T10:01:00+00:00",
        "started_at": "2026-05-21T10:00:00+00:00",
        "finished_at": "2026-05-21T10:01:00+00:00",
        "recording": {
            "duration_sec": 30,
            "sample_rate": 128,
            "channels": ["AF3", "AF4", "T7", "T8"],
            "source": "test",
            "model_name": "vad-test",
            "reference_name": "reference-test",
            "created_at": "2026-05-21T10:00:00+00:00",
            "quality_mean": 0.9,
        },
        "result": {
            "vad_raw": [0.6, 0.3, 0.4],
            "vad_ref_percentile": {"valence": 0.6, "arousal": 0.3, "dominance": 0.4},
            "low_confidence": False,
            "valid_window_ratio": 0.95,
            "resting_state_features": {
                "raw_metrics": {
                    "band_relative_power_global": {
                        "alpha": 0.58,
                        "beta": 0.31,
                        "theta": 0.44,
                        "gamma": 0.22,
                    }
                },
                "traits": {
                    "activation_edge": {"score": 0.41, "value": 0.1},
                    "inner_calm": {"score": 0.73, "value": 0.2},
                    "cognitive_tempo": {"score": 0.45, "value": 0.3},
                    "rhythm_clarity": {"score": 0.55, "value": 0.4},
                    "network_bridges": {"score": 0.52, "value": 0.5},
                    "inward_drift": {"score": 0.47, "value": 0.6},
                    "affective_tilt": {"score": 0.49, "value": -0.2},
                    "hemispheric_balance": {"score": 2.03, "value": 0.73},
                    "neural_texture": {"score": 0.61, "value": 0.7},
                    "temporal_weather": {"score": 0.57, "value": 0.8},
                },
            },
            "inference_manifest": None,
        },
        "error": None,
        "metadata": {"expected_recording_sec": 30, "subject_name": "백인호"},
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_test_config(tmp_path: Path) -> Path:
    eeg_config_path = tmp_path / "eeg_interpreter" / "config.json"
    world_config_path = tmp_path / "world_spawn" / "config.json"
    pipeline_config_path = tmp_path / "config.json"

    write_json(
        eeg_config_path,
        {
            "instruction_output_dir": str(tmp_path / "world_instructions"),
            "world_counter_path": str(tmp_path / "world_counter.txt"),
            "llm": {"provider": "gemini", "dry_run": True},
        },
    )
    write_json(
        world_config_path,
        {
            "world_spawn_dir": str(tmp_path / "world_spawn_json"),
            "planet_spawn_dir": str(tmp_path / "planet_spawn"),
            "vad_footprint_dir": str(tmp_path / "vad_footprint"),
            "person_world_map_path": str(tmp_path / "person_world_map.json"),
            "world_space": 2500,
            "planet_min_distance": 900.0,
        },
    )
    write_json(
        pipeline_config_path,
        {
            "instruction_output_dir": str(tmp_path / "world_instructions"),
            "eeg_interpreter_config": str(eeg_config_path),
            "world_spawn_config": str(world_config_path),
            "remote_control": {"enabled": False},
        },
    )
    return pipeline_config_path


def test_generate_world_creates_outputs(monkeypatch, tmp_path):
    monkeypatch.setenv("PIPELINE_CONFIG", str(write_test_config(tmp_path)))

    with TestClient(app) as client:
        response = client.post("/generate", json=sample_payload())

    assert response.status_code == 201
    body = response.json()
    assert body["world_id"] == "world_0001"
    assert Path(body["instruction_path"]).exists()
    assert Path(body["world_spawn_path"]).exists()
    assert Path(body["planet_layout_path"]).exists()
    assert Path(body["planets_layout_path"]).exists()
    assert Path(body["vad_footprint_path"]).exists()

    instruction = json.loads(Path(body["instruction_path"]).read_text(encoding="utf-8"))
    assert instruction["world_id"] == "world_0001"
    assert instruction["person_name"] == "백인호"
    assert instruction["world_number"] == 1
    assert instruction["source_eeg"]["vad"] == {"valence": 0.6, "arousal": 0.3, "dominance": 0.4}
    assert instruction["source_eeg"]["statue_traits"]["network_bridges"] == {"score": 0.52, "value": 0.5}
    assert instruction["source_eeg"]["placement_traits"]["hemispheric_balance"] == {"score": 2.03, "value": 0.73}

    expected_unreal_vad = {"valence": 0.2, "arousal": -0.4, "dominance": -0.2}
    world_spawn = json.loads(Path(body["world_spawn_path"]).read_text(encoding="utf-8"))
    vad_footprint = json.loads(Path(body["vad_footprint_path"]).read_text(encoding="utf-8"))
    assert world_spawn["base_vad"] == expected_unreal_vad
    assert world_spawn["flower_color_group"] == "red_yellow"
    assert next(asset["mesh"] for asset in world_spawn["assets"] if asset["asset_key"] == "fountain") == "basic"
    assert next(asset["mesh"] for asset in world_spawn["assets"] if asset["asset_key"] == "statue") == "network_bridges"
    assert len([asset for asset in world_spawn["assets"] if asset["asset_key"] == "tree"]) == 2
    assert vad_footprint["base_vad"] == expected_unreal_vad
    assert vad_footprint["current_vad"] == expected_unreal_vad


def test_generate_world_rejects_duplicate_id(monkeypatch, tmp_path):
    monkeypatch.setenv("PIPELINE_CONFIG", str(write_test_config(tmp_path)))

    with TestClient(app) as client:
        first = client.post("/generate", json=sample_payload())
        second = client.post("/generate/world_0001", json=sample_payload())

    assert first.status_code == 201
    assert second.status_code == 201
    assert second.json()["world_id"] == "world_0002"


def test_generate_world_increments_world_number(monkeypatch, tmp_path):
    monkeypatch.setenv("PIPELINE_CONFIG", str(write_test_config(tmp_path)))

    with TestClient(app) as client:
        first = client.post("/generate", json=sample_payload())
        second = client.post("/generate", json=sample_payload())

    assert first.status_code == 201
    assert second.status_code == 201

    first_instruction = json.loads(Path(first.json()["instruction_path"]).read_text(encoding="utf-8"))
    second_instruction = json.loads(Path(second.json()["instruction_path"]).read_text(encoding="utf-8"))
    assert first_instruction["world_number"] == 1
    assert second_instruction["world_number"] == 2


def test_next_world_number_wraps_after_slot_limit(tmp_path):
    counter_path = tmp_path / "world_counter.txt"
    worlds_dir = tmp_path / "worlds"

    counter_path.write_text("1", encoding="utf-8")
    assert next_world_number(counter_path, [worlds_dir]) == 2
    assert counter_path.read_text(encoding="utf-8") == "2"

    counter_path.write_text("29", encoding="utf-8")
    assert next_world_number(counter_path, [worlds_dir]) == 30
    assert counter_path.read_text(encoding="utf-8") == "30"

    counter_path.write_text("30", encoding="utf-8")
    assert next_world_number(counter_path, [worlds_dir]) == 1
    assert counter_path.read_text(encoding="utf-8") == "1"

    counter_path.write_text("35", encoding="utf-8")
    assert next_world_number(counter_path, [worlds_dir]) == 6
    assert counter_path.read_text(encoding="utf-8") == "6"


def test_next_world_number_wraps_when_existing_worlds_exceed_limit(tmp_path):
    counter_path = tmp_path / "world_counter.txt"
    worlds_dir = tmp_path / "worlds"
    write_json(worlds_dir / "world_0035.json", {"world_id": "world_0035", "world_number": 35})

    counter_path.write_text("1", encoding="utf-8")
    assert next_world_number(counter_path, [worlds_dir]) == 2
    assert counter_path.read_text(encoding="utf-8") == "2"


def test_next_world_number_bootstraps_from_existing_slots_only(tmp_path):
    counter_path = tmp_path / "world_counter.txt"
    worlds_dir = tmp_path / "worlds"
    write_json(worlds_dir / "world_0029.json", {"world_id": "world_0029", "world_number": 29})
    write_json(worlds_dir / "world_0037.json", {"world_id": "world_0037", "world_number": 37})

    assert next_world_number(counter_path, [worlds_dir]) == 30
    assert counter_path.read_text(encoding="utf-8") == "30"


def test_generate_world_overwrites_reused_slot(monkeypatch, tmp_path):
    config_path = write_test_config(tmp_path)
    monkeypatch.setenv("PIPELINE_CONFIG", str(config_path))

    (tmp_path / "world_counter.txt").write_text("30", encoding="utf-8")
    write_json(tmp_path / "world_instructions" / "world_0001.json", {"world_id": "old"})
    write_json(tmp_path / "world_spawn_json" / "world_0001.json", {"world_id": "old"})
    write_json(
        tmp_path / "vad_footprint" / "world_0001.json",
        {
            "world_id": "world_0001",
            "base_vad": {"valence": -1.0, "arousal": -1.0, "dominance": -1.0},
            "current_vad": {"valence": 1.0, "arousal": 1.0, "dominance": 1.0},
            "vad_footprints": [{"reason": "old interaction"}],
        },
    )
    write_json(
        tmp_path / "planet_spawn" / "planets_layout.json",
        {
            "planets": [
                {"planet_id": "world_0001", "location": {"x": 0, "y": 0, "z": 0}},
                {"planet_id": "world_0002", "location": {"x": 1, "y": 1, "z": 1}},
            ]
        },
    )
    write_json(
        tmp_path / "person_world_map.json",
        {"Old Person": "world_0001", "Other Person": "world_0002"},
    )

    payload = sample_payload()
    payload["metadata"]["subject_name"] = "New Person"
    with TestClient(app) as client:
        response = client.post("/generate", json=payload)

    assert response.status_code == 201
    assert response.json()["world_id"] == "world_0001"

    instruction = json.loads((tmp_path / "world_instructions" / "world_0001.json").read_text(encoding="utf-8"))
    world_spawn = json.loads((tmp_path / "world_spawn_json" / "world_0001.json").read_text(encoding="utf-8"))
    vad_footprint = json.loads((tmp_path / "vad_footprint" / "world_0001.json").read_text(encoding="utf-8"))
    planets = json.loads((tmp_path / "planet_spawn" / "planets_layout.json").read_text(encoding="utf-8"))
    person_map = json.loads((tmp_path / "person_world_map.json").read_text(encoding="utf-8"))

    assert instruction["person_name"] == "New Person"
    assert world_spawn["person_name"] == "New Person"
    assert vad_footprint["base_vad"] == world_spawn["base_vad"]
    assert vad_footprint["current_vad"] == world_spawn["base_vad"]
    assert vad_footprint["vad_footprints"] == []
    assert [planet["planet_id"] for planet in planets["planets"]].count("world_0001") == 1
    assert person_map == {"Other Person": "world_0002", "New Person": "world_0001"}


def test_generate_world_rejects_failed_payload(monkeypatch, tmp_path):
    monkeypatch.setenv("PIPELINE_CONFIG", str(write_test_config(tmp_path)))
    payload = sample_payload()
    payload["status"] = "failed"
    payload["result"] = None
    payload["error"] = {"code": "failed", "message": "inference failed", "stage": "inferring"}

    with TestClient(app) as client:
        response = client.post("/generate", json=payload)

    assert response.status_code == 422


def test_generate_world_rejects_missing_result(monkeypatch, tmp_path):
    monkeypatch.setenv("PIPELINE_CONFIG", str(write_test_config(tmp_path)))
    payload = sample_payload()
    payload["result"] = None

    with TestClient(app) as client:
        response = client.post("/generate", json=payload)

    assert response.status_code == 422


def test_generate_world_rejects_invalid_vad_raw(monkeypatch, tmp_path):
    monkeypatch.setenv("PIPELINE_CONFIG", str(write_test_config(tmp_path)))
    payload = sample_payload()
    payload["result"]["vad_raw"] = [0.1, 0.2]

    with TestClient(app) as client:
        response = client.post("/generate", json=payload)

    assert response.status_code == 422


def test_generate_world_deprecated_path_ignores_supplied_world_id(monkeypatch, tmp_path):
    monkeypatch.setenv("PIPELINE_CONFIG", str(write_test_config(tmp_path)))

    with TestClient(app) as client:
        response = client.post("/generate/test.world", json=sample_payload())

    assert response.status_code == 201
    assert response.json()["world_id"] == "world_0001"
