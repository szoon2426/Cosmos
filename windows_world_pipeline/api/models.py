from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

JobStatus = Literal["queued", "recording", "inferring", "succeeded", "failed"]
VAD_KEYS = {"valence", "arousal", "dominance"}


class FlexibleModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class JobErrorPayload(FlexibleModel):
    code: str
    message: str
    stage: str


class RecordingSummaryPayload(FlexibleModel):
    duration_sec: int
    sample_rate: int
    channels: list[str]
    source: str
    model_name: str
    reference_name: str
    created_at: str
    signal_path: str | None = None
    capture_path: str | None = None
    battery_last: int | None = None
    quality_last: float | None = None
    quality_mean: float | None = None
    quality_min: float | None = None
    incomplete: bool = False


class InferenceManifestPayload(FlexibleModel):
    model_name: str
    reference_name: str
    model_path: str
    config_path: str
    reference_path: str
    config_hash: str
    ran_at: str
    preprocess_params: dict[str, Any] = Field(default_factory=dict)


class ResultSummaryPayload(FlexibleModel):
    vad_raw: list[float] = Field(min_length=3, max_length=3)
    vad_ref_percentile: dict[str, float]
    low_confidence: bool
    valid_window_ratio: float
    resting_state_features: dict[str, Any] | None = None
    inference_manifest: InferenceManifestPayload | None = None

    @field_validator("vad_ref_percentile")
    @classmethod
    def validate_vad_ref_percentile(cls, value: dict[str, float]) -> dict[str, float]:
        missing = VAD_KEYS.difference(value)
        if missing:
            missing_keys = ", ".join(sorted(missing))
            raise ValueError(f"vad_ref_percentile is missing: {missing_keys}")
        return value


class EEGEmotionsPayload(FlexibleModel):
    job_id: str
    status: JobStatus
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    recording: RecordingSummaryPayload | None = None
    result: ResultSummaryPayload | None = None
    error: JobErrorPayload | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_generate_payload(self) -> EEGEmotionsPayload:
        if self.status != "succeeded":
            raise ValueError("Only succeeded result payloads can generate worlds")
        if self.result is None:
            raise ValueError("Succeeded payloads must include result")
        return self


class GenerateWorldResponse(BaseModel):
    world_id: str
    instruction_path: str
    world_spawn_path: str
    planet_layout_path: str
    planets_layout_path: str
    vad_footprint_path: str
    created_at: str
