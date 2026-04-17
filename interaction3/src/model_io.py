from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

try:
    import joblib as _joblib
except ImportError:
    _joblib = None


def save_model(model: Any, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if _joblib is not None:
        _joblib.dump(model, output_path)
        return

    with output_path.open("wb") as fp:
        pickle.dump(model, fp)


def load_model(path: str | Path) -> Any:
    model_path = Path(path)
    if _joblib is not None:
        return _joblib.load(model_path)

    with model_path.open("rb") as fp:
        return pickle.load(fp)
