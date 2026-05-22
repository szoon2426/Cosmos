from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.camera_preprocess import FramePreprocessor, PreprocessConfig
from src.final_hand_features import FinalHandFeatures, actual_both_open
from src.hand_roi_rescue import CropRect, map_crop_landmarks_to_frame


class LowlightTrackingTests(unittest.TestCase):
    def test_auto_preprocess_enhances_dark_frame(self) -> None:
        frame = np.full((48, 64, 3), 24, dtype=np.uint8)
        preprocessor = FramePreprocessor(PreprocessConfig(mode="auto"))

        result = preprocessor.process(frame)

        self.assertTrue(result.applied)
        self.assertGreater(float(np.mean(result.frame_bgr)), float(np.mean(frame)))

    def test_auto_preprocess_leaves_bright_frame_unapplied(self) -> None:
        gradient = np.tile(np.linspace(80, 255, 64, dtype=np.uint8), (48, 1))
        frame = np.dstack((gradient, gradient, gradient))
        preprocessor = FramePreprocessor(PreprocessConfig(mode="auto"))

        result = preprocessor.process(frame)

        self.assertFalse(result.applied)
        self.assertIs(result.frame_bgr, frame)

    def test_crop_landmark_coordinates_map_back_to_frame(self) -> None:
        landmarks = [
            SimpleNamespace(x=0.0, y=0.0, z=0.0),
            SimpleNamespace(x=0.5, y=0.5, z=-0.2),
            SimpleNamespace(x=1.0, y=1.0, z=0.2),
        ]
        rect = CropRect(x=100, y=50, width=200, height=100)

        mapped = map_crop_landmarks_to_frame(landmarks, rect, (1000, 500))

        self.assertAlmostEqual(mapped[0].x, 0.1)
        self.assertAlmostEqual(mapped[0].y, 0.1)
        self.assertAlmostEqual(mapped[1].x, 0.2)
        self.assertAlmostEqual(mapped[1].y, 0.2)
        self.assertAlmostEqual(mapped[2].x, 0.3)
        self.assertAlmostEqual(mapped[2].y, 0.3)
        self.assertAlmostEqual(mapped[1].z, -0.04)

    def test_pose_fallback_cannot_start_open_interaction(self) -> None:
        fallback_left = FinalHandFeatures(
            visible=True,
            hand_visible=False,
            pose_fallback=True,
            open_strength=1.0,
        )
        real_right = FinalHandFeatures(
            visible=True,
            hand_visible=True,
            open_strength=1.0,
        )

        self.assertFalse(actual_both_open(real_right, fallback_left))


if __name__ == "__main__":
    unittest.main()
