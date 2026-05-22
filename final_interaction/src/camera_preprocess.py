from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class PreprocessConfig:
    mode: str = "auto"
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8
    alpha: float = 1.12
    beta: int = 8
    auto_luma_mean: float = 92.0
    auto_luma_std: float = 38.0


@dataclass(slots=True)
class PreprocessResult:
    frame_bgr: np.ndarray
    luma_mean: float
    luma_std: float
    applied: bool


class FramePreprocessor:
    def __init__(self, config: PreprocessConfig) -> None:
        self.config = config
        tile = max(2, int(config.clahe_tile_grid))
        self._clahe = cv2.createCLAHE(
            clipLimit=max(0.1, float(config.clahe_clip_limit)),
            tileGridSize=(tile, tile),
        )

    def process(self, frame_bgr: np.ndarray) -> PreprocessResult:
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:, :, 0]
        luma_mean = float(np.mean(y_channel))
        luma_std = float(np.std(y_channel))

        mode = self.config.mode
        should_apply = mode == "lowlight" or (
            mode == "auto"
            and (luma_mean <= self.config.auto_luma_mean or luma_std <= self.config.auto_luma_std)
        )
        if mode == "off" or not should_apply:
            return PreprocessResult(
                frame_bgr=frame_bgr,
                luma_mean=luma_mean,
                luma_std=luma_std,
                applied=False,
            )

        enhanced_ycrcb = ycrcb.copy()
        enhanced_ycrcb[:, :, 0] = self._clahe.apply(y_channel)
        enhanced = cv2.cvtColor(enhanced_ycrcb, cv2.COLOR_YCrCb2BGR)
        enhanced = cv2.convertScaleAbs(
            enhanced,
            alpha=max(0.1, float(self.config.alpha)),
            beta=int(self.config.beta),
        )
        return PreprocessResult(
            frame_bgr=enhanced,
            luma_mean=luma_mean,
            luma_std=luma_std,
            applied=True,
        )
