from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimpleKNNClassifier:
    n_neighbors: int = 3

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleKNNClassifier":
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if len(X) != len(y):
            raise ValueError("X and y length mismatch")

        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ < 1e-6] = 1.0
        self.X_train_ = (X - self.mean_) / self.scale_
        self.y_train_ = y.astype(str)
        self.classes_ = np.asarray(sorted(set(self.y_train_.tolist())))
        self.n_features_in_ = X.shape[1]
        return self

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def _neighbor_info(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_scaled = self._transform(X)
        diff = self.X_train_[None, :, :] - X_scaled[:, None, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        k = min(self.n_neighbors, self.X_train_.shape[0])
        idx = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
        neighbor_dists = np.take_along_axis(dists, idx, axis=1)
        order = np.argsort(neighbor_dists, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        neighbor_dists = np.take_along_axis(neighbor_dists, order, axis=1)
        return idx, neighbor_dists

    def predict(self, X: np.ndarray) -> np.ndarray:
        idx, _ = self._neighbor_info(X)
        preds: list[str] = []
        for row in idx:
            labels = self.y_train_[row]
            values, counts = np.unique(labels, return_counts=True)
            best = values[np.argmax(counts)]
            preds.append(str(best))
        return np.asarray(preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        idx, neighbor_dists = self._neighbor_info(X)
        probs = np.zeros((X.shape[0], len(self.classes_)), dtype=np.float32)
        for row_i, (row_idx, row_dist) in enumerate(zip(idx, neighbor_dists)):
            labels = self.y_train_[row_idx]
            weights = 1.0 / (row_dist + 1e-6)
            for label, weight in zip(labels, weights):
                class_i = int(np.where(self.classes_ == label)[0][0])
                probs[row_i, class_i] += float(weight)
            total = probs[row_i].sum()
            if total > 0:
                probs[row_i] /= total
        return probs
