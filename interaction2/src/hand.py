from dataclasses import dataclass


@dataclass
class HandEstimator:
    model_path: str | None = None
