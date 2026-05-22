from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.camera_preprocess import FramePreprocessor, PreprocessConfig
from src.final_hand_features import FinalHandFeatures, actual_both_open
from src.final_hud_state import hud_frame_state
from src.final_mapper import compute_final_payload
from src.hand_roi_rescue import CropRect, map_crop_landmarks_to_frame
from src.realtime_inference_final import (
    BaseMemoryState,
    GRAB_RECOVERY_SECONDS,
    InteractionSession,
    LEFT_SOLO_GRAB_SWIPE_DELTA_X,
    PointerRuntimeState,
    VADFootprintStore,
    WorldVADState,
    activate_grab_session,
    can_recover_grab,
    end_interaction,
    left_solo_galaxy_debug_state,
    recover_world_vad_after_release,
    reset_interaction_session,
    update_left_solo_galaxy_gesture,
)


class LowlightTrackingTests(unittest.TestCase):
    def _left_solo_hud_state(self, session: InteractionSession, now: float):
        payload = compute_final_payload(
            interaction_active=False,
            pointer_x=-1.0,
            pointer_y=-1.0,
            target_v=0.0,
            target_a=0.0,
            target_d=0.0,
            grab_active=False,
            open_strength=0.0,
            grab_strength=0.0,
            switch_to_camera=False,
        )
        return hud_frame_state(
            timestamp=now,
            frame_index=0,
            mode=session.mode,
            world_active=True,
            world_id="world_a",
            world_number=1,
            both_visible=False,
            both_open=False,
            both_grab=False,
            left_features=FinalHandFeatures(),
            right_features=FinalHandFeatures(),
            left_pointer_active=False,
            right_pointer_active=False,
            payload=payload,
            **left_solo_galaxy_debug_state(session, now),
        )

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
        recorded = []
        store = SimpleNamespace(record_interaction=lambda vad, reason: recorded.append((vad, reason)) or True)
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
        self.assertEqual(recorded, [((0.1, 0.2, 0.3), "interaction_settled")])

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

    def test_vad_footprint_loads_base_when_no_memory_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = VADFootprintStore(Path(tmp_dir))
            base_vad = (0.2, -0.4, 0.6)

            loaded = store.load_world("world_a", base_vad)

            self.assertEqual(loaded, base_vad)
            data = json.loads((Path(tmp_dir) / "world_a.json").read_text(encoding="utf-8"))
            self.assertEqual(data["current_vad"], {"valence": 0.2, "arousal": -0.4, "dominance": 0.6})

    def test_vad_footprint_blends_last_memory_at_fifteen_percent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "world_a.json"
            path.write_text(
                json.dumps(
                    {
                        "world_id": "world_a",
                        "base_vad": {"valence": 0.0, "arousal": 0.5, "dominance": -0.5},
                        "current_vad": {"valence": 1.0, "arousal": -0.5, "dominance": 0.5},
                        "vad_footprints": [],
                    }
                ),
                encoding="utf-8",
            )
            store = VADFootprintStore(Path(tmp_dir))

            loaded = store.load_world("world_a", (0.0, 0.5, -0.5))

            self.assertAlmostEqual(loaded[0], 0.15)
            self.assertAlmostEqual(loaded[1], 0.35)
            self.assertAlmostEqual(loaded[2], -0.35)

    def test_vad_footprint_ignores_invalid_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "world_a.json"
            path.write_text(
                json.dumps(
                    {
                        "world_id": "world_a",
                        "base_vad": {"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
                        "current_vad": {"valence": "bad"},
                        "vad_footprints": [{"reason": "old"}],
                    }
                ),
                encoding="utf-8",
            )
            store = VADFootprintStore(Path(tmp_dir))

            loaded = store.load_world("world_a", (0.4, -0.2, 0.8))

            self.assertEqual(loaded, (0.4, -0.2, 0.8))
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["current_vad"], {"valence": 0.4, "arousal": -0.2, "dominance": 0.8})

    def test_end_interaction_records_current_vad_without_changing_runtime_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = VADFootprintStore(Path(tmp_dir))
            runtime_base = store.load_world("world_a", (0.0, 0.2, 0.4))
            memory = BaseMemoryState(*runtime_base)
            session = InteractionSession(active=True, mode="grab")

            end_interaction(session, store, (0.7, -0.3, 0.1), 100.0)

            self.assertEqual(memory.current_base(), runtime_base)
            data = json.loads((Path(tmp_dir) / "world_a.json").read_text(encoding="utf-8"))
            self.assertEqual(data["current_vad"], {"valence": 0.7, "arousal": -0.3, "dominance": 0.1})
            self.assertEqual(data["vad_footprints"][-1]["reason"], "interaction_settled")

    def test_released_interaction_holds_then_recovers_to_runtime_base(self) -> None:
        memory = BaseMemoryState(0.0, 0.5, -0.5)
        session = InteractionSession(active=False, released_at=10.0)
        world_vad = (1.0, 1.0, 1.0)

        held = recover_world_vad_after_release(session, memory, world_vad, 14.9)
        recovered = recover_world_vad_after_release(session, memory, world_vad, 15.0)

        self.assertEqual(held, world_vad)
        self.assertLess(recovered[0], world_vad[0])
        self.assertLess(recovered[1], world_vad[1])
        self.assertLess(recovered[2], world_vad[2])

    def test_left_only_grab_under_five_seconds_does_not_lock_vad(self) -> None:
        session = InteractionSession()
        world_vad = (0.8, 0.1, -0.2)
        base_vad = (-0.3, 0.4, 0.5)
        left = FinalHandFeatures(hand_visible=True, grab_active=True, x=0.4)
        right = FinalHandFeatures(hand_visible=False, grab_active=False)

        update_left_solo_galaxy_gesture(
            session,
            pointer_world_active=True,
            left_features=left,
            right_features=right,
            world_vad=world_vad,
            world_base_vad=base_vad,
            now=100.0,
        )
        vad, switch = update_left_solo_galaxy_gesture(
            session,
            pointer_world_active=True,
            left_features=FinalHandFeatures(hand_visible=True, grab_active=True, x=0.41),
            right_features=right,
            world_vad=world_vad,
            world_base_vad=base_vad,
            now=104.9,
        )

        self.assertEqual(vad, world_vad)
        self.assertFalse(switch)

    def test_left_only_grab_at_five_seconds_locks_vad_to_world_base(self) -> None:
        session = InteractionSession()
        world_vad = (0.8, 0.1, -0.2)
        base_vad = (-0.3, 0.4, 0.5)
        right = FinalHandFeatures(hand_visible=False, grab_active=False)

        update_left_solo_galaxy_gesture(
            session,
            pointer_world_active=True,
            left_features=FinalHandFeatures(hand_visible=True, grab_active=True, x=0.4),
            right_features=right,
            world_vad=world_vad,
            world_base_vad=base_vad,
            now=100.0,
        )
        vad, switch = update_left_solo_galaxy_gesture(
            session,
            pointer_world_active=True,
            left_features=FinalHandFeatures(hand_visible=True, grab_active=True, x=0.41),
            right_features=right,
            world_vad=world_vad,
            world_base_vad=base_vad,
            now=105.0,
        )

        self.assertEqual(vad, base_vad)
        self.assertFalse(switch)

    def test_right_grab_cancels_left_only_galaxy_gesture(self) -> None:
        session = InteractionSession()
        world_vad = (0.8, 0.1, -0.2)
        base_vad = (-0.3, 0.4, 0.5)
        left = FinalHandFeatures(hand_visible=True, grab_active=True, x=0.4)

        update_left_solo_galaxy_gesture(
            session,
            pointer_world_active=True,
            left_features=left,
            right_features=FinalHandFeatures(hand_visible=False, grab_active=False),
            world_vad=world_vad,
            world_base_vad=base_vad,
            now=100.0,
        )
        update_left_solo_galaxy_gesture(
            session,
            pointer_world_active=True,
            left_features=left,
            right_features=FinalHandFeatures(hand_visible=True, grab_active=True),
            world_vad=world_vad,
            world_base_vad=base_vad,
            now=102.0,
        )
        self.assertIsNone(session.left_solo_grab_started_at)

        vad, switch = update_left_solo_galaxy_gesture(
            session,
            pointer_world_active=True,
            left_features=left,
            right_features=FinalHandFeatures(hand_visible=False, grab_active=False),
            world_vad=world_vad,
            world_base_vad=base_vad,
            now=106.9,
        )

        self.assertEqual(session.left_solo_grab_started_at, 106.9)
        self.assertEqual(vad, world_vad)
        self.assertFalse(switch)

    def test_rightward_swipe_after_left_only_hold_triggers_galaxy(self) -> None:
        session = InteractionSession()
        world_vad = (0.8, 0.1, -0.2)
        base_vad = (-0.3, 0.4, 0.5)
        right = FinalHandFeatures(hand_visible=False, grab_active=False)

        update_left_solo_galaxy_gesture(
            session,
            pointer_world_active=True,
            left_features=FinalHandFeatures(hand_visible=True, grab_active=True, x=0.4),
            right_features=right,
            world_vad=world_vad,
            world_base_vad=base_vad,
            now=100.0,
        )
        update_left_solo_galaxy_gesture(
            session,
            pointer_world_active=True,
            left_features=FinalHandFeatures(hand_visible=True, grab_active=True, x=0.41),
            right_features=right,
            world_vad=world_vad,
            world_base_vad=base_vad,
            now=105.0,
        )
        vad, switch = update_left_solo_galaxy_gesture(
            session,
            pointer_world_active=True,
            left_features=FinalHandFeatures(hand_visible=True, grab_active=True, x=0.64),
            right_features=right,
            world_vad=world_vad,
            world_base_vad=base_vad,
            now=105.1,
        )

        self.assertEqual(vad, base_vad)
        self.assertTrue(switch)

    def test_rightward_swipe_before_left_only_hold_does_not_trigger_galaxy(self) -> None:
        session = InteractionSession()
        world_vad = (0.8, 0.1, -0.2)
        base_vad = (-0.3, 0.4, 0.5)
        right = FinalHandFeatures(hand_visible=False, grab_active=False)

        update_left_solo_galaxy_gesture(
            session,
            pointer_world_active=True,
            left_features=FinalHandFeatures(hand_visible=True, grab_active=True, x=0.4),
            right_features=right,
            world_vad=world_vad,
            world_base_vad=base_vad,
            now=100.0,
        )
        update_left_solo_galaxy_gesture(
            session,
            pointer_world_active=True,
            left_features=FinalHandFeatures(hand_visible=True, grab_active=True, x=0.41),
            right_features=right,
            world_vad=world_vad,
            world_base_vad=base_vad,
            now=104.8,
        )
        vad, switch = update_left_solo_galaxy_gesture(
            session,
            pointer_world_active=True,
            left_features=FinalHandFeatures(hand_visible=True, grab_active=True, x=0.64),
            right_features=right,
            world_vad=world_vad,
            world_base_vad=base_vad,
            now=104.9,
        )

        self.assertEqual(vad, world_vad)
        self.assertFalse(switch)

    def test_arm_fallback_swipe_after_left_only_hold_triggers_galaxy(self) -> None:
        session = InteractionSession()
        world_vad = (0.8, 0.1, -0.2)
        base_vad = (-0.3, 0.4, 0.5)
        right = FinalHandFeatures(hand_visible=False, grab_active=False)

        update_left_solo_galaxy_gesture(
            session,
            pointer_world_active=True,
            left_features=FinalHandFeatures(hand_visible=True, grab_active=True, x=0.4),
            right_features=right,
            world_vad=world_vad,
            world_base_vad=base_vad,
            now=100.0,
        )
        update_left_solo_galaxy_gesture(
            session,
            pointer_world_active=True,
            left_features=FinalHandFeatures(visible=True, pose_fallback=True, x=0.41),
            right_features=right,
            world_vad=world_vad,
            world_base_vad=base_vad,
            now=105.0,
        )
        vad, switch = update_left_solo_galaxy_gesture(
            session,
            pointer_world_active=True,
            left_features=FinalHandFeatures(visible=True, pose_fallback=True, x=0.64),
            right_features=right,
            world_vad=world_vad,
            world_base_vad=base_vad,
            now=105.1,
        )

        self.assertEqual(vad, base_vad)
        self.assertTrue(switch)

    def test_arm_fallback_cannot_start_left_only_galaxy_gesture(self) -> None:
        session = InteractionSession()
        world_vad = (0.8, 0.1, -0.2)
        base_vad = (-0.3, 0.4, 0.5)

        vad, switch = update_left_solo_galaxy_gesture(
            session,
            pointer_world_active=True,
            left_features=FinalHandFeatures(visible=True, pose_fallback=True, x=0.64),
            right_features=FinalHandFeatures(hand_visible=False, grab_active=False),
            world_vad=world_vad,
            world_base_vad=base_vad,
            now=105.1,
        )

        self.assertIsNone(session.left_solo_grab_started_at)
        self.assertEqual(vad, world_vad)
        self.assertFalse(switch)

    def test_right_grab_cancels_arm_fallback_galaxy_swipe(self) -> None:
        session = InteractionSession()
        world_vad = (0.8, 0.1, -0.2)
        base_vad = (-0.3, 0.4, 0.5)

        update_left_solo_galaxy_gesture(
            session,
            pointer_world_active=True,
            left_features=FinalHandFeatures(hand_visible=True, grab_active=True, x=0.4),
            right_features=FinalHandFeatures(hand_visible=False, grab_active=False),
            world_vad=world_vad,
            world_base_vad=base_vad,
            now=100.0,
        )
        vad, switch = update_left_solo_galaxy_gesture(
            session,
            pointer_world_active=True,
            left_features=FinalHandFeatures(visible=True, pose_fallback=True, x=0.64),
            right_features=FinalHandFeatures(hand_visible=True, grab_active=True),
            world_vad=world_vad,
            world_base_vad=base_vad,
            now=105.1,
        )

        self.assertIsNone(session.left_solo_grab_started_at)
        self.assertEqual(vad, world_vad)
        self.assertFalse(switch)

    def test_left_solo_hud_debug_inactive_defaults_to_zero(self) -> None:
        state = self._left_solo_hud_state(InteractionSession(), 100.0)

        self.assertFalse(state.left_solo_grab_active)
        self.assertEqual(state.left_solo_grab_elapsed, 0.0)
        self.assertEqual(state.left_solo_grab_hold_progress, 0.0)
        self.assertFalse(state.left_solo_vad_restore_active)
        self.assertEqual(state.left_solo_swipe_delta_x, 0.0)
        self.assertEqual(state.left_solo_swipe_velocity_x, 0.0)
        self.assertEqual(state.left_solo_swipe_progress, 0.0)
        self.assertFalse(state.left_solo_swipe_velocity_ready)
        self.assertFalse(state.left_solo_world_move_ready)
        self.assertFalse(state.left_solo_world_move_fired)

    def test_left_solo_hud_debug_tracks_partial_hold(self) -> None:
        session = InteractionSession()
        world_vad = (0.8, 0.1, -0.2)
        base_vad = (-0.3, 0.4, 0.5)
        right = FinalHandFeatures(hand_visible=False, grab_active=False)

        update_left_solo_galaxy_gesture(
            session,
            pointer_world_active=True,
            left_features=FinalHandFeatures(hand_visible=True, grab_active=True, x=0.4),
            right_features=right,
            world_vad=world_vad,
            world_base_vad=base_vad,
            now=100.0,
        )
        update_left_solo_galaxy_gesture(
            session,
            pointer_world_active=True,
            left_features=FinalHandFeatures(hand_visible=True, grab_active=True, x=0.5),
            right_features=right,
            world_vad=world_vad,
            world_base_vad=base_vad,
            now=102.5,
        )

        state = self._left_solo_hud_state(session, 102.5)

        self.assertTrue(state.left_solo_grab_active)
        self.assertAlmostEqual(state.left_solo_grab_elapsed, 2.5)
        self.assertAlmostEqual(state.left_solo_grab_hold_progress, 0.5)
        self.assertFalse(state.left_solo_vad_restore_active)
        self.assertAlmostEqual(state.left_solo_swipe_delta_x, 0.1)
        self.assertAlmostEqual(state.left_solo_swipe_velocity_x, 0.04)
        self.assertAlmostEqual(state.left_solo_swipe_progress, 0.1 / LEFT_SOLO_GRAB_SWIPE_DELTA_X)
        self.assertFalse(state.left_solo_swipe_velocity_ready)
        self.assertFalse(state.left_solo_world_move_ready)
        self.assertFalse(state.left_solo_world_move_fired)

    def test_left_solo_hud_debug_reports_vad_restore_after_hold(self) -> None:
        session = InteractionSession()
        world_vad = (0.8, 0.1, -0.2)
        base_vad = (-0.3, 0.4, 0.5)
        right = FinalHandFeatures(hand_visible=False, grab_active=False)

        update_left_solo_galaxy_gesture(
            session,
            pointer_world_active=True,
            left_features=FinalHandFeatures(hand_visible=True, grab_active=True, x=0.4),
            right_features=right,
            world_vad=world_vad,
            world_base_vad=base_vad,
            now=100.0,
        )
        update_left_solo_galaxy_gesture(
            session,
            pointer_world_active=True,
            left_features=FinalHandFeatures(hand_visible=True, grab_active=True, x=0.41),
            right_features=right,
            world_vad=world_vad,
            world_base_vad=base_vad,
            now=105.0,
        )

        state = self._left_solo_hud_state(session, 105.0)

        self.assertTrue(state.left_solo_grab_active)
        self.assertAlmostEqual(state.left_solo_grab_hold_progress, 1.0)
        self.assertTrue(state.left_solo_vad_restore_active)
        self.assertGreater(state.left_solo_swipe_progress, 0.0)
        self.assertFalse(state.left_solo_swipe_velocity_ready)
        self.assertTrue(state.left_solo_world_move_ready)
        self.assertFalse(state.left_solo_world_move_fired)

    def test_left_solo_hud_debug_reports_world_move_fired(self) -> None:
        session = InteractionSession()
        world_vad = (0.8, 0.1, -0.2)
        base_vad = (-0.3, 0.4, 0.5)
        right = FinalHandFeatures(hand_visible=False, grab_active=False)

        update_left_solo_galaxy_gesture(
            session,
            pointer_world_active=True,
            left_features=FinalHandFeatures(hand_visible=True, grab_active=True, x=0.4),
            right_features=right,
            world_vad=world_vad,
            world_base_vad=base_vad,
            now=100.0,
        )
        update_left_solo_galaxy_gesture(
            session,
            pointer_world_active=True,
            left_features=FinalHandFeatures(hand_visible=True, grab_active=True, x=0.41),
            right_features=right,
            world_vad=world_vad,
            world_base_vad=base_vad,
            now=105.0,
        )
        update_left_solo_galaxy_gesture(
            session,
            pointer_world_active=True,
            left_features=FinalHandFeatures(hand_visible=True, grab_active=True, x=0.64),
            right_features=right,
            world_vad=world_vad,
            world_base_vad=base_vad,
            now=105.1,
        )

        state = self._left_solo_hud_state(session, 105.1)

        self.assertTrue(state.left_solo_vad_restore_active)
        self.assertAlmostEqual(state.left_solo_swipe_delta_x, 0.24)
        self.assertGreaterEqual(state.left_solo_swipe_velocity_x, 1.2)
        self.assertEqual(state.left_solo_swipe_progress, 1.0)
        self.assertTrue(state.left_solo_swipe_velocity_ready)
        self.assertFalse(state.left_solo_world_move_ready)
        self.assertTrue(state.left_solo_world_move_fired)

    def test_world_change_loads_blended_vad_and_resets_interaction_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            world_dir = root / "worlds"
            footprint_dir = root / "footprints"
            world_dir.mkdir()
            footprint_dir.mkdir()
            (world_dir / "world_a.json").write_text(
                json.dumps(
                    {
                        "world_id": "world_a",
                        "world_number": 7,
                        "base_vad": {"valence": 0.0, "arousal": 0.5, "dominance": -0.5},
                    }
                ),
                encoding="utf-8",
            )
            (footprint_dir / "world_a.json").write_text(
                json.dumps(
                    {
                        "world_id": "world_a",
                        "base_vad": {"valence": 0.0, "arousal": 0.5, "dominance": -0.5},
                        "current_vad": {"valence": 1.0, "arousal": -0.5, "dominance": 0.5},
                        "vad_footprints": [],
                    }
                ),
                encoding="utf-8",
            )
            store = VADFootprintStore(footprint_dir)
            state = WorldVADState(world_json_dir=world_dir, footprint_store=store)
            memory = BaseMemoryState(0.0, 0.0, 0.0)
            session = InteractionSession(active=True, mode="grab", released_at=1.0, left_pointer_locked=True)
            pointer_state = PointerRuntimeState(world_number=99, left_y=999.0)

            try:
                changed = state._apply_world_id("world_a", memory)
                if changed:
                    reset_interaction_session(session)
                    pointer_state.reset(state.current_world_number)
            finally:
                state.close()

            self.assertTrue(changed)
            self.assertAlmostEqual(memory.current_base()[0], 0.15)
            self.assertFalse(session.active)
            self.assertIsNone(session.released_at)
            self.assertFalse(session.left_pointer_locked)
            self.assertEqual(pointer_state.world_number, 7)


if __name__ == "__main__":
    unittest.main()
