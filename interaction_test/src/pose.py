from dataclasses import dataclass


@dataclass
class PoseEstimator:
    model_path: str | None = None
