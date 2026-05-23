from __future__ import annotations

from windows_world_pipeline.world_spawn.build_world_spawn import (
    build_trees,
    infer_flower_group,
    pick_fountain_mesh,
    pick_flowers,
    pick_statue_mesh,
    tree_count_from_hemispheric_balance,
)


def test_pick_statue_mesh_forces_network_bridges_at_threshold() -> None:
    source_eeg = {
        "statue_traits": {
            "inner_calm": {"score": 0.95, "value": 0.2},
            "cognitive_tempo": {"score": 0.2, "value": 0.3},
            "inward_drift": {"score": 0.2, "value": 0.4},
            "activation_edge": {"score": 0.2, "value": 0.5},
            "affective_tilt": {"score": 0.2, "value": 0.6},
            "rhythm_clarity": {"score": 0.2, "value": 0.7},
            "network_bridges": {"score": 0.5, "value": 0.1},
        }
    }

    assert pick_statue_mesh(source_eeg) == "network_bridges"


def test_pick_statue_mesh_breaks_score_tie_by_value() -> None:
    source_eeg = {
        "statue_traits": {
            "inner_calm": {"score": 0.7, "value": 0.2},
            "cognitive_tempo": {"score": 0.7, "value": 10.0},
            "inward_drift": {"score": 0.4, "value": 0.4},
            "activation_edge": {"score": 0.4, "value": 0.5},
            "affective_tilt": {"score": 0.4, "value": 0.6},
            "rhythm_clarity": {"score": 0.7, "value": 1.8},
            "network_bridges": {"score": 0.49, "value": 0.9},
        }
    }

    assert pick_statue_mesh(source_eeg) == "neural_tempo"


def test_pick_fountain_mesh_uses_lowest_statue_trait_score() -> None:
    source_eeg = {
        "statue_traits": {
            "inner_calm": {"score": 0.7, "value": 0.2},
            "cognitive_tempo": {"score": 0.05, "value": 7.75},
            "inward_drift": {"score": 1.0, "value": 2.3},
            "activation_edge": {"score": 0.84, "value": 1.2},
            "affective_tilt": {"score": 0.51, "value": 0.02},
            "rhythm_clarity": {"score": 0.21, "value": 1.6},
            "network_bridges": {"score": 0.85, "value": 0.8},
        }
    }

    assert pick_fountain_mesh(source_eeg) == "basic"


def test_pick_fountain_mesh_breaks_low_score_tie_by_lowest_value() -> None:
    source_eeg = {
        "statue_traits": {
            "inner_calm": {"score": 0.4, "value": 0.3},
            "cognitive_tempo": {"score": 0.4, "value": 7.75},
            "inward_drift": {"score": 0.6, "value": 2.3},
            "activation_edge": {"score": 0.6, "value": 1.2},
            "affective_tilt": {"score": 0.4, "value": -0.02},
            "rhythm_clarity": {"score": 0.8, "value": 1.6},
            "network_bridges": {"score": 0.8, "value": 0.8},
        }
    }

    assert pick_fountain_mesh(source_eeg) == "rock_octagon"


def test_infer_flower_group_uses_vad_quadrants() -> None:
    assert infer_flower_group({"source_eeg": {"vad": {"valence": 0.7, "arousal": 0.7}}}, []) == "red"
    assert infer_flower_group({"source_eeg": {"vad": {"valence": 0.3, "arousal": 0.7}}}, []) == "purple"
    assert infer_flower_group({"source_eeg": {"vad": {"valence": 0.3, "arousal": 0.3}}}, []) == "yellow"
    assert infer_flower_group({"source_eeg": {"vad": {"valence": 0.7, "arousal": 0.3}}}, []) == "red_yellow"


def test_pick_flowers_can_mix_red_and_yellow_for_fourth_quadrant() -> None:
    class FirstThenLastRandom:
        def __init__(self) -> None:
            self.calls = 0

        def randint(self, _start: int, _end: int) -> int:
            return 2

        def choice(self, values: list[str]) -> str:
            self.calls += 1
            return values[0] if self.calls == 1 else values[-1]

    flowers = pick_flowers(
        FirstThenLastRandom(),
        {"source_eeg": {"vad": {"valence": 0.7, "arousal": 0.3}}},
        [],
    )

    assert flowers == ["silver_downy_2", "windflower_2"]


def test_tree_count_uses_ceil_of_hemispheric_balance_score_bucket_plus_one() -> None:
    source_eeg = {
        "placement_traits": {
            "hemispheric_balance": {"score": 2.03, "value": 0.73},
        }
    }

    assert tree_count_from_hemispheric_balance(source_eeg) == 2


def test_build_trees_creates_requested_number_of_tree_assets() -> None:
    class FixedRandom:
        def uniform(self, start: float, end: float) -> float:
            return (start + end) / 2.0

    trees = build_trees(FixedRandom(), 4)

    assert len(trees) == 4
    assert all(tree["asset_key"] == "tree" for tree in trees)


def test_tree_count_is_capped_at_eight() -> None:
    source_eeg = {
        "placement_traits": {
            "hemispheric_balance": {"score": 100.0, "value": 0.0},
        }
    }

    assert tree_count_from_hemispheric_balance(source_eeg) == 8
