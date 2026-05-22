from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from windows_world_pipeline.api.main import app


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
