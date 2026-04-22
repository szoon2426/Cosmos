from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_MIN_VALID_WINDOW_RATIO = 0.70


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def default_jobs_root() -> Path:
    cosmos_root = Path(__file__).resolve().parents[2]
    return cosmos_root / "eeg-emotions" / "artifacts" / "jobs"


def remap_raw_vad(value: float) -> float:
    return clamp(float(value) / 50.0 - 1.0, -1.0, 1.0)


@dataclass(slots=True)
class EEGProfile:
    source_path: str
    job_id: str | None
    base_v: float
    base_a: float
    base_d: float
    vad_raw: list[float]
    vad_ref_percentile: dict[str, float]
    low_confidence: bool
    valid_window_ratio: float
    min_valid_window_ratio: float

    @property
    def usable(self) -> bool:
        return (not self.low_confidence) and self.valid_window_ratio >= self.min_valid_window_ratio

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["usable"] = self.usable
        return payload


@dataclass(slots=True)
class BaseVADState:
    v: float = 0.0
    a: float = 0.0
    d: float = 0.0
    source_path: str | None = None
    job_id: str | None = None
    updated: bool = False
    usable: bool = False
    low_confidence: bool = True
    valid_window_ratio: float = 0.0
    reason: str = "uninitialized"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_result_block(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("result")
    if isinstance(result, dict):
        return result
    return payload


def _extract_job_id(payload: dict[str, Any], path: Path) -> str | None:
    job_id = payload.get("job_id")
    if isinstance(job_id, str) and job_id:
        return job_id
    parent = path.parent.name
    return parent or None


def find_latest_result_path(jobs_root: str | Path | None = None) -> Path | None:
    root = Path(jobs_root) if jobs_root is not None else default_jobs_root()
    if not root.exists():
        return None

    result_paths = [path for path in root.glob("*/result.json") if path.is_file()]
    if not result_paths:
        return None

    return max(result_paths, key=lambda item: item.stat().st_mtime)


def resolve_result_path(
    result_path: str | Path | None = None,
    jobs_root: str | Path | None = None,
) -> Path | None:
    if result_path is not None:
        path = Path(result_path)
        if path.is_dir():
            candidate = path / "result.json"
            return candidate if candidate.exists() else None
        return path if path.exists() else None
    return find_latest_result_path(jobs_root)


def load_eeg_profile(
    result_path: str | Path | None = None,
    jobs_root: str | Path | None = None,
    min_valid_window_ratio: float = DEFAULT_MIN_VALID_WINDOW_RATIO,
) -> EEGProfile | None:
    resolved = resolve_result_path(result_path=result_path, jobs_root=jobs_root)
    if resolved is None:
        return None

    payload = _read_json(resolved)
    result = _extract_result_block(payload)

    vad_raw = result.get("vad_raw")
    if not isinstance(vad_raw, list) or len(vad_raw) < 3:
        raise ValueError(f"Invalid vad_raw in EEG result: {resolved}")

    vad_ref_percentile = result.get("vad_ref_percentile")
    if not isinstance(vad_ref_percentile, dict):
        vad_ref_percentile = {}

    low_confidence = bool(result.get("low_confidence", True))
    valid_window_ratio = float(result.get("valid_window_ratio", 0.0))

    return EEGProfile(
        source_path=str(resolved),
        job_id=_extract_job_id(payload, resolved),
        base_v=remap_raw_vad(float(vad_raw[0])),
        base_a=remap_raw_vad(float(vad_raw[1])),
        base_d=remap_raw_vad(float(vad_raw[2])),
        vad_raw=[float(vad_raw[0]), float(vad_raw[1]), float(vad_raw[2])],
        vad_ref_percentile={str(key): float(value) for key, value in vad_ref_percentile.items()},
        low_confidence=low_confidence,
        valid_window_ratio=valid_window_ratio,
        min_valid_window_ratio=float(min_valid_window_ratio),
    )


def resolve_base_vad(
    profile: EEGProfile | None,
    previous: BaseVADState | None = None,
) -> BaseVADState:
    previous_state = previous or BaseVADState()

    if profile is None:
        return BaseVADState(
            v=previous_state.v,
            a=previous_state.a,
            d=previous_state.d,
            source_path=previous_state.source_path,
            job_id=previous_state.job_id,
            updated=False,
            usable=False,
            low_confidence=True,
            valid_window_ratio=0.0,
            reason="no_result",
        )

    if not profile.usable:
        reason = "low_confidence" if profile.low_confidence else "low_valid_window_ratio"
        return BaseVADState(
            v=previous_state.v,
            a=previous_state.a,
            d=previous_state.d,
            source_path=profile.source_path,
            job_id=profile.job_id,
            updated=False,
            usable=False,
            low_confidence=profile.low_confidence,
            valid_window_ratio=profile.valid_window_ratio,
            reason=reason,
        )

    return BaseVADState(
        v=profile.base_v,
        a=profile.base_a,
        d=profile.base_d,
        source_path=profile.source_path,
        job_id=profile.job_id,
        updated=True,
        usable=True,
        low_confidence=profile.low_confidence,
        valid_window_ratio=profile.valid_window_ratio,
        reason="updated_from_result",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Load EEG V3 result.json and print base VAD summary")
    parser.add_argument("--result", help="Path to result.json or a job directory that contains result.json")
    parser.add_argument(
        "--jobs-root",
        default=str(default_jobs_root()),
        help="Root directory that contains EEG job folders",
    )
    parser.add_argument(
        "--min-valid-window-ratio",
        type=float,
        default=DEFAULT_MIN_VALID_WINDOW_RATIO,
        help="Minimum valid window ratio required to mark the profile usable",
    )
    parser.add_argument("--prev-v", type=float, default=0.0, help="Previous base V to keep when EEG result is unusable")
    parser.add_argument("--prev-a", type=float, default=0.0, help="Previous base A to keep when EEG result is unusable")
    parser.add_argument("--prev-d", type=float, default=0.0, help="Previous base D to keep when EEG result is unusable")
    args = parser.parse_args()

    profile = load_eeg_profile(
        result_path=args.result,
        jobs_root=args.jobs_root,
        min_valid_window_ratio=args.min_valid_window_ratio,
    )
    resolved = resolve_base_vad(
        profile,
        previous=BaseVADState(v=args.prev_v, a=args.prev_a, d=args.prev_d),
    )
    payload = {
        "profile": profile.as_dict() if profile is not None else None,
        "resolved_base_vad": resolved.as_dict(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
