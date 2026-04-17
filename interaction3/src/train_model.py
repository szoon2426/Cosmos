from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    from model_io import save_model
    from simple_knn import SimpleKNNClassifier
else:
    from .model_io import save_model
    from .simple_knn import SimpleKNNClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Train kNN model for interaction3")
    parser.add_argument("--dataset", default="interaction3/data/dataset.jsonl")
    parser.add_argument("--output", default="interaction3/models/knn_model.joblib")
    parser.add_argument("--neighbors", type=int, default=3)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    X = []
    y = []
    feature_dim: int | None = None
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row_dim = len(row["features"])
        if feature_dim is None:
            feature_dim = row_dim
        elif row_dim != feature_dim:
            raise ValueError(
                f"Inconsistent feature dimension in dataset: expected {feature_dim}, got {row_dim}. "
                "Please rebuild the dataset after feature changes."
            )
        X.append(row["features"])
        y.append(row["label"])

    X_np = np.asarray(X, dtype=np.float32)
    y_np = np.asarray(y)

    model = SimpleKNNClassifier(n_neighbors=args.neighbors)
    model.fit(X_np, y_np)

    output_path = Path(args.output)
    save_model(model, output_path)
    print(f"[train] saved model -> {output_path}")
    print(f"[train] samples={len(X_np)} classes={sorted(set(y_np.tolist()))}")


if __name__ == "__main__":
    main()
