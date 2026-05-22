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
from src.realtime_inference_final import (
    GRAB_RECOVERY_SECONDS,
    InteractionSession,
    activate_grab_session,
    can_recover_grab,
    end_interaction,
)


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

    def test_lost_open_session_opens_grab_recovery_window(self) -> None:
        remembered = []
        store = SimpleNamespace(remember_pending=lambda vad: remembered.append(vad))
        session = InteractionSession(active=True, mode="open")

        end_interaction(
            session,
            store,
            (0.1, 0.2, 0.3),
            100.0,
            allow_grab_recovery=True,
        )

        self.assertFalse(session.active)
        self.assertEqual(session.mode, "idle")
        self.assertAlmostEqual(session.grab_recover_until, 100.0 + GRAB_RECOVERY_SECONDS)
        self.assertTrue(can_recover_grab(session, 104.9))
        self.assertFalse(can_recover_grab(session, 105.1))
        self.assertEqual(remembered, [(0.1, 0.2, 0.3)])

    def test_activate_grab_session_enters_grab_without_open_anchor(self) -> None:
        session = InteractionSession(active=False, mode="idle", grab_recover_until=105.0)

        activate_grab_session(
            session,
            pointer_xy=(0.4, 0.6),
            avg_z=0.2,
            span_x=0.12,
            span_y=0.08,
            world_vad=(0.1, -0.2, 0.3),
            now=101.0,
        )

        self.assertTrue(session.active)
        self.assertEqual(session.mode, "grab")
        self.assertTrue(session.grab_locked)
        self.assertIsNone(session.grab_recover_until)
        self.assertEqual((session.grab_anchor_x, session.grab_anchor_y, session.grab_anchor_z), (0.4, 0.6, 0.2))
        self.assertEqual((session.grab_anchor_v, session.grab_anchor_a, session.grab_anchor_d), (0.1, -0.2, 0.3))


if __name__ == "__main__":
    unittest.main()
