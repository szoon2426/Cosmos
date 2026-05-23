from __future__ import annotations

import argparse
import json
import math
import random
from math import ceil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FLOWER_GROUPS = {
    "purple": ["silver_downy_1", "pentas_1", "pentas_2", "leadwort_1", "leadwort_2", "bigleaf"],
    "red": ["silver_downy_2", "bougainv_1", "bougainv_2", "dianthus_1", "dianthus_2", "daisy_1", "daisy_2"],
    "yellow": ["campion_1", "campion_2", "gazania_1", "gazania_2", "crownbeard_1", "crownbeard_2", "windflower_1", "windflower_2"],
}
FLOWER_GROUPS["red_yellow"] = FLOWER_GROUPS["red"] + FLOWER_GROUPS["yellow"]
FLOWER_TYPES = [flower for flowers in FLOWER_GROUPS.values() for flower in flowers]
PLANET_MESH_TYPES = ["basic_planet", "sharp_planet", "smooth_planet", "complicated_planet", "simple_planet"]
PLANET_MATERIAL_TYPES = ["yellow", "blue", "green", "orange", "purple", "gold", "red", "pink"]
FOUNTAIN_MESH_TYPES = ["rock_octagon", "plate_round", "pillar_round", "basic"]
FOUNTAIN_TRAIT_TO_MESH = {
    "inner_calm": "plate_round",
    "cognitive_tempo": "basic",
    "inward_drift": "basic",
    "activation_edge": "pillar_round",
    "affective_tilt": "rock_octagon",
    "rhythm_clarity": "plate_round",
    "network_bridges": "rock_octagon",
}
STATUE_MESH_TYPES = [
    "inner_quietness",
    "neural_tempo",
    "inward_drift",
    "activation_edge",
    "frontal_tilt",
    "resonance_clarity",
    "network_bridges",
]
STATUE_TRAIT_TO_MESH = {
    "inner_calm": "inner_quietness",
    "cognitive_tempo": "neural_tempo",
    "inward_drift": "inward_drift",
    "activation_edge": "activation_edge",
    "affective_tilt": "frontal_tilt",
    "rhythm_clarity": "resonance_clarity",
    "network_bridges": "network_bridges",
}
STATUE_TRAIT_ORDER = [
    "inner_calm",
    "cognitive_tempo",
    "inward_drift",
    "activation_edge",
    "affective_tilt",
    "rhythm_clarity",
    "network_bridges",
]
NETWORK_BRIDGES_FORCE_THRESHOLD = 0.5

FRONT_X = -2000.0
BACK_X = 300.0
FRONT_Y_LIMIT = 1550.0
BACK_Y_LIMIT = 3700.0
FRONT_Z_LIMIT = 900.0
BACK_Z_LIMIT = 2100.0
CAMERA_X = -900.0
CAMERA_Y = 40.0
STATUE_MIN_X = 30.0
STATUE_MAX_X = 310.0
STATUE_MIN_SCALE = 1.25
STATUE_MAX_SCALE = 1.35
TREE_COUNT_MAX = 8


def load_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def vad_to_dict(vad: dict[str, Any]) -> dict[str, float]:
    return {
        "valence": round(float(vad.get("valence", 0.0)), 4),
        "arousal": round(float(vad.get("arousal", 0.0)), 4),
        "dominance": round(float(vad.get("dominance", 0.0)), 4),
    }


def vad01_to_unreal(vad: dict[str, Any]) -> dict[str, float]:
    def convert(key: str) -> float:
        try:
            value = float(vad.get(key, 0.5))
        except (TypeError, ValueError):
            value = 0.5
        value = max(0.0, min(1.0, value))
        return round(value * 2.0 - 1.0, 4)

    return {
        "valence": convert("valence"),
        "arousal": convert("arousal"),
        "dominance": convert("dominance"),
    }


def ensure_vad_footprint(path: Path, world: dict[str, Any]) -> None:
    world_id = str(world["world_id"])
    base_vad = vad_to_dict(world.get("base_vad", {}))
    path.parent.mkdir(parents=True, exist_ok=True)

    write_json(
        path,
        {
            "world_id": world_id,
            "base_vad": base_vad,
            "current_vad": base_vad,
            "vad_footprints": [],
        },
    )


def resolve_path(config_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return config_dir / path


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def inverse_lerp(a: float, b: float, value: float) -> float:
    if a == b:
        return 0.0
    return max(0.0, min(1.0, (value - a) / (b - a)))


def distance_3d(a: dict[str, float], b: dict[str, float]) -> float:
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2)


def normalize_flower_types(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    flowers: list[str] = []
    for value in values:
        if isinstance(value, str) and value in FLOWER_TYPES and value not in flowers:
            flowers.append(value)
    return flowers[:5]


def infer_flower_group(instruction: dict[str, Any], atmosphere: list[str]) -> str:
    vad = instruction.get("source_eeg", {}).get("vad", {})
    if isinstance(vad, dict) and vad:
        valence = number_from(vad, "valence")
        arousal = number_from(vad, "arousal")
        if valence >= 0.5 and arousal >= 0.5:
            return "red"
        if valence < 0.5 and arousal >= 0.5:
            return "purple"
        if valence < 0.5 and arousal < 0.5:
            return "yellow"
        return "red_yellow"

    requested = str(instruction.get("flower_color_group", "")).strip()
    if requested in FLOWER_GROUPS:
        return requested

    material = str(instruction.get("planet_material_type", "")).strip()
    if material in ("purple", "red", "yellow"):
        return material
    if material in ("gold", "orange"):
        return "yellow"
    if material == "pink":
        return "red"
    if any(keyword in atmosphere for keyword in ("dreamlike", "misty", "soft", "quiet", "pale")):
        return "purple"
    if any(keyword in atmosphere for keyword in ("tense", "charged", "luminous")):
        return "red"
    return "purple"


def pick_flowers(rng: random.Random, instruction: dict[str, Any], atmosphere: list[str]) -> list[str]:
    group = infer_flower_group(instruction, atmosphere)
    vad = instruction.get("source_eeg", {}).get("vad", {})
    explicit_flowers = normalize_flower_types(instruction.get("flower_types"))
    if not (isinstance(vad, dict) and vad) and len(explicit_flowers) >= 2:
        return explicit_flowers

    count = rng.randint(2, 5)
    weighted = list(FLOWER_GROUPS[group])

    flowers: list[str] = []
    while len(flowers) < count:
        flower = rng.choice(weighted)
        if flower not in flowers:
            flowers.append(flower)
    return flowers


def asset(asset_key: str, location: list[float], scale: list[float], yaw: float = 0.0, **extra: Any) -> dict[str, Any]:
    item = {
        "asset_key": asset_key,
        "location": location,
        "rotation": [0.0, 0.0, yaw],
        "scale": scale,
    }
    item.update(extra)
    return item


def yaw_toward_camera(location: list[float]) -> float:
    dx = float(location[0]) - CAMERA_X
    dy = float(location[1]) - CAMERA_Y
    yaw = math.degrees(math.atan2(dy, dx))
    return round(yaw % 360.0, 2)


def statue_scale_for_x(x: float) -> list[float]:
    t = inverse_lerp(STATUE_MIN_X, STATUE_MAX_X, x)
    scale = round(lerp(STATUE_MIN_SCALE, STATUE_MAX_SCALE, t), 3)
    return [scale, scale, scale]


def pick_statue_location(rng: random.Random, pond_y: float) -> list[float]:
    x = round(rng.uniform(STATUE_MIN_X, STATUE_MAX_X), 2)
    y = pond_y + rng.choice([-140, 140])
    return [x, y, 0]


def number_from(data: dict[str, Any], key: str, default: float = 0.5) -> float:
    try:
        return float(data.get(key, default))
    except (TypeError, ValueError):
        return default


def trait_number(traits: dict[str, Any], trait: str, field: str, default: float = -1.0) -> float:
    item = traits.get(trait, {})
    if not isinstance(item, dict):
        return default
    return number_from(item, field, default)


def tree_count_from_hemispheric_balance(source_eeg: dict[str, Any]) -> int:
    placement_traits = source_eeg.get("placement_traits", {})
    score = trait_number(placement_traits, "hemispheric_balance", "score", 1.0)
    return max(1, min(TREE_COUNT_MAX, ceil(score / 10.0) + 1))


def build_trees(rng: random.Random, count: int) -> list[dict[str, Any]]:
    trees: list[dict[str, Any]] = []
    for index in range(count):
        side = -1 if index % 2 == 0 else 1
        x = round(rng.uniform(560.0, 760.0), 2)
        y = round(side * rng.uniform(280.0, 620.0), 2)
        scale_value = round(rng.uniform(1.12, 1.34), 3)
        yaw = round(rng.uniform(-18.0, 18.0), 2)
        trees.append(asset("tree", [x, y, 0], [scale_value, scale_value, scale_value], yaw))
    return trees


def pick_statue_mesh(source_eeg: dict[str, Any]) -> str:
    statue_traits = source_eeg.get("statue_traits", {})
    if isinstance(statue_traits, dict) and statue_traits:
        network_score = trait_number(statue_traits, "network_bridges", "score")
        if network_score >= NETWORK_BRIDGES_FORCE_THRESHOLD:
            return STATUE_TRAIT_TO_MESH["network_bridges"]

        selected_trait = max(
            STATUE_TRAIT_ORDER,
            key=lambda trait: (
                trait_number(statue_traits, trait, "score"),
                trait_number(statue_traits, trait, "value"),
            ),
        )
        return STATUE_TRAIT_TO_MESH[selected_trait]

    world_style = source_eeg.get("world_style", {})
    if isinstance(world_style, dict) and world_style:
        scores = {
            "inner_quietness": number_from(world_style, "quietness", -1.0),
            "neural_tempo": number_from(world_style, "tempo", -1.0),
            "inward_drift": number_from(world_style, "drift", -1.0),
            "activation_edge": number_from(source_eeg.get("features", {}), "engagement", -1.0),
            "frontal_tilt": abs(number_from(world_style, "frontal_tilt", 0.0)),
            "resonance_clarity": number_from(world_style, "clarity", -1.0),
            "network_bridges": number_from(world_style, "bandwidth", -1.0),
        }
        if scores["network_bridges"] >= NETWORK_BRIDGES_FORCE_THRESHOLD:
            return "network_bridges"
        return max(scores, key=scores.get)

    vad = source_eeg.get("vad", {})
    features = source_eeg.get("features", {})
    if number_from(vad, "arousal") > 0.66:
        return "activation_edge"
    if number_from(vad, "dominance") > 0.66:
        return "resonance_clarity"
    if number_from(features, "relaxation") > 0.66 or number_from(vad, "arousal") < 0.34:
        return "inner_quietness"
    if number_from(features, "engagement") > 0.62 or number_from(features, "beta") > 0.62:
        return "neural_tempo"
    return "frontal_tilt"


def pick_fountain_mesh(source_eeg: dict[str, Any]) -> str:
    statue_traits = source_eeg.get("statue_traits", {})
    if isinstance(statue_traits, dict) and statue_traits:
        selected_trait = min(
            STATUE_TRAIT_ORDER,
            key=lambda trait: (
                trait_number(statue_traits, trait, "score"),
                trait_number(statue_traits, trait, "value"),
            ),
        )
        return FOUNTAIN_TRAIT_TO_MESH[selected_trait]

    vad = source_eeg.get("vad", {})
    features = source_eeg.get("features", {})
    arousal = number_from(vad, "arousal")
    dominance = number_from(vad, "dominance")
    relaxation = number_from(features, "relaxation")
    engagement = number_from(features, "engagement")
    alpha = number_from(features, "alpha")
    beta = number_from(features, "beta")
    theta = number_from(features, "theta")
    gamma = number_from(features, "gamma")

    if relaxation >= 0.68 or alpha >= 0.62:
        return "plate_round"
    if arousal >= 0.68 or gamma >= 0.55:
        return "pillar_round"
    if dominance >= 0.62 or engagement >= 0.65 or beta >= 0.62:
        return "rock_octagon"
    if theta >= 0.56:
        return "basic"
    return "basic"


def build_world_spawn(instruction: dict[str, Any], world_space: int) -> dict[str, Any]:
    world_id = str(instruction["world_id"])
    world_number = int(instruction["world_number"])
    atmosphere = list(instruction.get("atmosphere", []))
    source_eeg = instruction.get("source_eeg", {})
    vad = source_eeg.get("vad", {})
    seed = random.Random(world_id).randint(1000, 999999)
    rng = random.Random(seed)
    fountain_mesh = pick_fountain_mesh(source_eeg)
    statue_mesh = pick_statue_mesh(source_eeg)
    tree_count = tree_count_from_hemispheric_balance(source_eeg)

    pond_y = rng.choice([-260, 260])
    statue_location = pick_statue_location(rng, pond_y)
    assets = [
        asset("fountain", [-430, -pond_y, 0], [1.05, 1.05, 1.05], 145.0, mesh=fountain_mesh),
        asset("pond", [-150, pond_y, 2], [1.35, 1.75, 1.8], 0.0),
        asset(
            "statue",
            statue_location,
            statue_scale_for_x(float(statue_location[0])),
            yaw_toward_camera(statue_location),
            mesh=statue_mesh,
        ),
        *build_trees(rng, tree_count),
        asset("rock_path", [-730, 40, 0], [0.9, 0.75, 1.0], 0.0),
        asset("rock_path", [-560, pond_y * 0.28, 0], [0.95, 0.75, 1.0], 5.0),
        asset("rock_path", [-390, pond_y * 0.55, 0], [0.9, 0.75, 1.0], -7.0),
        asset("rock_m1", [320, -360, 0], [1.02, 0.8, 1.0], 68.0),
        asset("rock_m2", [320, 360, 0], [0.98, 0.78, 1.0], 24.0),
        asset("glow_sphere", [-620, -130, 25], [1.0, 1.0, 1.0], 0.0),
        asset("glow_sphere", [-260, 310, 25], [1.0, 1.0, 1.0], 0.0),
    ]

    return {
        "schema_version": 2,
        "world_id": world_id,
        "world_number": world_number,
        "world_name": instruction.get("world_concept", world_id),
        "person_name": instruction.get("person_name", "unknown"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "world_space": world_space,
        "world_offset": world_number * world_space,
        "base_vad": vad01_to_unreal(vad),
        "flower_density": 0.74 if any(keyword in atmosphere for keyword in ("dreamlike", "misty", "soft")) else 0.62,
        "flower_color_group": infer_flower_group(instruction, atmosphere),
        "flower_type": pick_flowers(rng, instruction, atmosphere),
        "assets": assets,
    }


def normalize_planet_mesh_type(mesh_type: str) -> str:
    allowed = set(PLANET_MESH_TYPES)
    aliases = {
        "rock_planet_01": "basic_planet",
        "rock_planet_02": "complicated_planet",
        "cracked_planet_01": "sharp_planet",
        "smooth_planet_01": "smooth_planet",
    }
    value = mesh_type.strip()
    value = aliases.get(value, value)
    return value if value in allowed else ""


def planet_type_from_instruction(instruction: dict[str, Any], key: str, fallback_options: list[str], rng: random.Random) -> str:
    value = instruction.get(key) or instruction.get("planet", {}).get(key)
    if isinstance(value, str) and value.strip():
        if key == "planet_mesh_type":
            normalized = normalize_planet_mesh_type(value)
            if normalized:
                return normalized
        elif value.strip() in fallback_options:
            return value.strip()
    return rng.choice(fallback_options)


def sample_planet_location(rng: random.Random) -> dict[str, float]:
    x = rng.uniform(FRONT_X, BACK_X)
    t = (x - FRONT_X) / (BACK_X - FRONT_X)
    y_limit = lerp(FRONT_Y_LIMIT, BACK_Y_LIMIT, t)
    z_limit = lerp(FRONT_Z_LIMIT, BACK_Z_LIMIT, t)
    return {
        "x": round(x, 2),
        "y": round(rng.uniform(-y_limit, y_limit), 2),
        "z": round(rng.uniform(-z_limit, z_limit), 2),
    }


def location_is_far_enough(location: dict[str, float], existing_planets: list[dict[str, Any]], min_distance: float) -> bool:
    for planet in existing_planets:
        other = planet.get("location", {})
        if not all(axis in other for axis in ("x", "y", "z")):
            continue
        if distance_3d(location, other) < min_distance:
            return False
    return True


def build_planet_spawn(
    instruction: dict[str, Any],
    existing_planets: list[dict[str, Any]],
    min_distance: float,
) -> dict[str, Any]:
    world_id = str(instruction["world_id"])
    rng = random.Random(f"planet:{world_id}")
    location = sample_planet_location(rng)
    for _ in range(200):
        candidate = sample_planet_location(rng)
        if location_is_far_enough(candidate, existing_planets, min_distance):
            location = candidate
            break

    return {
        "planet_id": world_id,
        "location": location,
        "scale": round(rng.uniform(0.35, 0.55), 2),
        "mesh_type": planet_type_from_instruction(instruction, "planet_mesh_type", PLANET_MESH_TYPES, rng),
        "material_type": planet_type_from_instruction(instruction, "planet_material_type", PLANET_MATERIAL_TYPES, rng),
    }


def upsert_by_key(items: list[dict[str, Any]], key: str, value: str, new_item: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in items if item.get(key) != value] + [new_item]


def update_person_world_map(path: Path, person_name: str, world_id: str) -> None:
    try:
        mapping = load_json(path, {}) or {}
    except json.JSONDecodeError:
        mapping = {}
    if not isinstance(mapping, dict):
        mapping = {}
    mapping = {str(name): value for name, value in mapping.items() if value != world_id}
    mapping[str(person_name)] = world_id
    write_json(path, mapping)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Cosmos world and planet spawn JSON.")
    parser.add_argument("--config", default="config.example.json")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--person-name", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    config_dir = config_path.parent
    config = load_json(config_path, {})
    instruction = load_json(Path(args.instruction))
    if args.person_name:
        instruction["person_name"] = args.person_name

    world_id = str(instruction["world_id"])
    world = build_world_spawn(instruction, int(config.get("world_space", 2500)))

    world_spawn_dir = resolve_path(config_dir, config.get("world_spawn_dir", "../world_spawn_json"))
    planet_spawn_dir = resolve_path(config_dir, config.get("planet_spawn_dir", "../planet_spawn"))
    person_map_path = resolve_path(config_dir, config.get("person_world_map_path", "../person_world_map.json"))
    vad_footprint_dir = resolve_path(config_dir, config.get("vad_footprint_dir", "../vad_footprint"))
    min_planet_distance = float(config.get("planet_min_distance", 900.0))

    world_path = world_spawn_dir / f"{world_id}.json"
    vad_footprint_path = vad_footprint_dir / f"{world_id}.json"
    planet_layout_path = planet_spawn_dir / "planet_layout.json"
    planets_layout_path = planet_spawn_dir / "planets_layout.json"

    planets_doc = load_json(planets_layout_path, {"planets": []}) or {"planets": []}
    existing_planets = list(planets_doc.get("planets", []))
    planet = build_planet_spawn(instruction, existing_planets, min_planet_distance)
    planets_doc["planets"] = upsert_by_key(existing_planets, "planet_id", world_id, planet)

    write_json(world_path, world)
    write_json(planet_layout_path, planet)
    write_json(planets_layout_path, planets_doc)
    ensure_vad_footprint(vad_footprint_path, world)
    update_person_world_map(person_map_path, world.get("person_name", "unknown"), world_id)

    print(f"[world_spawn] wrote world -> {world_path}")
    print(f"[world_spawn] wrote latest planet -> {planet_layout_path}")
    print(f"[world_spawn] updated planets -> {planets_layout_path}")
    print(f"[world_spawn] ensured vad footprint -> {vad_footprint_path}")
    print(f"[world_spawn] updated person map -> {person_map_path}")


if __name__ == "__main__":
    main()
