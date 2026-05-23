from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

WORLD_ID_PATTERN = re.compile(r"^world_(\d{4,})$")
WORLD_SLOT_LIMIT = 30


SYSTEM_RULES = """
You interpret EEG-derived emotional features into a single symbolic asset generation brief.
Return only valid JSON.

3D asset prompt rule:
- The final asset must keep the same ancient stone pedestal artifact style every time.
- Only the symbolic object integrated into the top of the pedestal should change.
- Do not generate a free-form scene prompt.
- The symbolic object must feel emotionally symbolic, ambiguous, sacred, and minimal.
- The symbolic object must be physically connected to the pedestal.
- Prefer wording such as "integrated into the top", "carved into the pedestal",
  "mounted on the pedestal", or "emerging from the pedestal".
- Avoid independent loose objects, floating pieces, thin fragile parts, characters,
  landscapes, fantasy scenes, castles, cities, open architecture, and explicit weapons.

Good symbolic_object examples:
- a blade-like stone relic
- an abstract animal-shaped stone idol
- a sealed stone codex
- a cracked orb relic
- a hollow crescent stone marker
- a folded shell-like limestone idol
- a closed eye-shaped stone emblem

Flower mesh rules:
- Choose flower_color_group from purple, red, or yellow.
- Purple flowers: silver_downy_1, pentas_1, pentas_2, leadwort_1, leadwort_2, bigleaf.
- Red flowers: silver_downy_2, bougainv_1, bougainv_2, dianthus_1, dianthus_2, daisy_1, daisy_2.
- Yellow flowers: campion_1, campion_2, gazania_1, gazania_2, crownbeard_1, crownbeard_2, windflower_1, windflower_2.
- Return 2 to 5 flower_types and keep them inside the chosen color group.
"""


BASE_NEGATIVE_PROMPT = (
    "character, person, warrior, fantasy scene, landscape, environment, castle, "
    "sword fight, multiple objects, floating pieces, thin fragile parts, text, "
    "watermark, logo, blurry, low quality"
)

ASSET_PROMPT_TEMPLATE = (
    "a single ancient stone pedestal artifact with {symbolic_object} integrated into the top, "
    "weathered limestone texture, carved stone surface, heavy sculptural base, "
    "museum object showcase, single centered object, front view, clean white background, "
    "soft studio lighting, stylized realistic, large readable silhouette, "
    "isolated 3D game asset, minimal composition"
)


def build_asset_prompt(symbolic_object: str) -> str:
    symbolic_object = symbolic_object.strip().strip(".")
    if not symbolic_object:
        symbolic_object = "a sealed stone relic"
    return ASSET_PROMPT_TEMPLATE.format(symbolic_object=symbolic_object)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_config_path(config_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return config_dir / path


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def max_world_number_in_dirs(paths: list[Path] | None) -> int:
    if not paths:
        return 0

    max_number = 0
    for path in paths:
        if not path.exists() or not path.is_dir():
            continue
        for json_path in path.glob("world_*.json"):
            match = WORLD_ID_PATTERN.match(json_path.stem)
            if not match:
                continue
            number = int(match.group(1))
            if 1 <= number <= WORLD_SLOT_LIMIT:
                max_number = max(max_number, number)
        for json_path in path.glob("*.json"):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError):
                continue
            world_id = str(data.get("world_id", ""))
            match = WORLD_ID_PATTERN.match(world_id)
            if not match:
                continue
            number = int(match.group(1))
            if 1 <= number <= WORLD_SLOT_LIMIT:
                max_number = max(max_number, number)
    return max_number


def next_world_number(counter_path: Path, existing_dirs: list[Path] | None = None) -> int:
    counter_path.parent.mkdir(parents=True, exist_ok=True)
    current: int | None = None
    if counter_path.exists():
        raw = counter_path.read_text(encoding="utf-8").strip()
        if raw:
            try:
                current = int(raw)
            except ValueError:
                current = None
    if current is None or current <= 0:
        current = max_world_number_in_dirs(existing_dirs)
    value = (current % WORLD_SLOT_LIMIT) + 1
    counter_path.write_text(str(value), encoding="utf-8")
    return value


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def percentile(value: Any, default: float = 50.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    if numeric > 1.0:
        numeric = numeric / 100.0
    return round(clamp01(numeric), 4)


def trait_score(traits: dict[str, Any], name: str, default: float = 0.5) -> float:
    trait = traits.get(name, {})
    if not isinstance(trait, dict):
        return default
    return percentile(trait.get("score", default), default * 100.0)


def trait_value(traits: dict[str, Any], name: str, default: float = 0.0) -> float:
    trait = traits.get(name, {})
    if not isinstance(trait, dict):
        return default
    try:
        return float(trait.get("value", default))
    except (TypeError, ValueError):
        return default


def statue_trait(traits: dict[str, Any], name: str, default_score: float = 0.0) -> dict[str, float]:
    return {
        "score": trait_score(traits, name, default_score),
        "value": round(trait_value(traits, name), 4),
    }


def raw_trait_score(traits: dict[str, Any], name: str, default: float = 0.0) -> float:
    trait = traits.get(name, {})
    if not isinstance(trait, dict):
        return default
    try:
        return float(trait.get("score", default))
    except (TypeError, ValueError):
        return default


def normalize_eeg_payload(eeg: dict[str, Any]) -> dict[str, Any]:
    if "result" not in eeg:
        return eeg

    result = eeg.get("result", {})
    resting = result.get("resting_state_features", {})
    raw_metrics = resting.get("raw_metrics", {})
    traits = resting.get("traits", {})
    band_power = raw_metrics.get("band_relative_power_global", {})
    vad_percentile = result.get("vad_ref_percentile", {})

    normalized = {
        "timestamp": eeg.get("finished_at") or eeg.get("updated_at") or eeg.get("created_at"),
        "person_name": eeg.get("person_name") or eeg.get("participant") or eeg.get("job_id", "unknown"),
        "job_id": eeg.get("job_id"),
        "features": {
            "alpha": percentile(band_power.get("alpha")),
            "beta": percentile(band_power.get("beta")),
            "theta": percentile(band_power.get("theta")),
            "gamma": percentile(band_power.get("gamma")),
            "engagement": trait_score(traits, "activation_edge"),
            "relaxation": trait_score(traits, "inner_calm"),
        },
        "vad": {
            "valence": percentile(vad_percentile.get("valence")),
            "arousal": percentile(vad_percentile.get("arousal")),
            "dominance": percentile(vad_percentile.get("dominance")),
        },
        "world_style": {
            "quietness": trait_score(traits, "inner_calm"),
            "tempo": trait_score(traits, "cognitive_tempo"),
            "clarity": trait_score(traits, "rhythm_clarity"),
            "bandwidth": trait_score(traits, "network_bridges"),
            "drift": trait_score(traits, "inward_drift"),
            "frontal_tilt": round(trait_value(traits, "affective_tilt"), 4),
            "texture": trait_score(traits, "neural_texture"),
            "ecology": trait_score(traits, "temporal_weather"),
        },
        "statue_traits": {
            "inner_calm": statue_trait(traits, "inner_calm"),
            "cognitive_tempo": statue_trait(traits, "cognitive_tempo"),
            "inward_drift": statue_trait(traits, "inward_drift"),
            "activation_edge": statue_trait(traits, "activation_edge"),
            "affective_tilt": statue_trait(traits, "affective_tilt"),
            "rhythm_clarity": statue_trait(traits, "rhythm_clarity"),
            "network_bridges": statue_trait(traits, "network_bridges"),
        },
        "placement_traits": {
            "hemispheric_balance": {
                "score": round(raw_trait_score(traits, "hemispheric_balance"), 4),
                "value": round(trait_value(traits, "hemispheric_balance"), 4),
            },
        },
        "quality": {
            "low_confidence": bool(result.get("low_confidence", False)),
            "valid_window_ratio": result.get("valid_window_ratio"),
            "quality_mean": eeg.get("recording", {}).get("quality_mean"),
        },
    }
    return normalized


def build_llm_prompt(eeg: dict[str, Any]) -> str:
    return (
        f"{SYSTEM_RULES}\n\n"
        "EEG JSON:\n"
        f"{json.dumps(eeg, ensure_ascii=False, indent=2)}\n\n"
        "Return this exact JSON shape:\n"
        "{\n"
        '  "world_concept": "short poetic concept",\n'
        '  "symbolic_object": "short noun phrase for the object integrated into the pedestal top",\n'
        '  "asset_prompt": "must be generated from the fixed ancient stone pedestal template",\n'
        f'  "negative_prompt": "{BASE_NEGATIVE_PROMPT}",\n'
        '  "archetype": "sealed_observatory | emotional_tower | abstract_vertical_structure | symbolic_landmark",\n'
        '  "planet_mesh_type": "basic_planet | sharp_planet | smooth_planet | complicated_planet | simple_planet",\n'
        '  "planet_material_type": "yellow | blue | green | orange | purple | gold | red | pink",\n'
        '  "flower_color_group": "purple | red | yellow",\n'
        '  "flower_types": ["valid flower mesh key", "valid flower mesh key"],\n'
        '  "atmosphere": ["keyword", "keyword", "keyword"],\n'
        '  "emotional_state": {"valence": "low|mid|high", "arousal": "low|mid|high", "dominance": "low|mid|high"}\n'
        "}\n"
        "\n"
        "The asset_prompt must follow this exact structure:\n"
        f"{ASSET_PROMPT_TEMPLATE}\n"
    )


def classify(value: float) -> str:
    if value < 0.34:
        return "low"
    if value > 0.66:
        return "high"
    return "mid"


def dry_run_interpretation(eeg: dict[str, Any]) -> dict[str, Any]:
    vad = eeg.get("vad", {})
    valence = float(vad.get("valence", 0.5))
    arousal = float(vad.get("arousal", 0.5))
    dominance = float(vad.get("dominance", 0.5))

    if arousal < 0.38 and dominance < 0.45:
        archetype = "sealed_observatory"
        concept = "quiet inner weather held inside a sealed vertical shrine"
        atmosphere = ["dreamlike", "misty", "soft", "sacred"]
        symbolic_object = "a sealed stone codex"
        planet_mesh_type = "smooth_planet"
        planet_material_type = "blue"
        flower_color_group = "purple"
        flower_types = ["leadwort_1", "pentas_1", "silver_downy_1"]
    elif arousal > 0.62:
        archetype = "emotional_tower"
        concept = "compressed pulse rising into a solitary emotional tower"
        atmosphere = ["tense", "luminous", "focused", "charged"]
        symbolic_object = "a blade-like stone relic"
        planet_mesh_type = "sharp_planet"
        planet_material_type = "orange"
        flower_color_group = "red"
        flower_types = ["bougainv_1", "dianthus_1", "daisy_1"]
    elif dominance > 0.62:
        archetype = "symbolic_landmark"
        concept = "still confidence condensed into a closed symbolic landmark"
        atmosphere = ["clear", "solemn", "balanced", "monolithic"]
        symbolic_object = "a closed eye-shaped stone emblem"
        planet_mesh_type = "complicated_planet"
        planet_material_type = "gold"
        flower_color_group = "yellow"
        flower_types = ["gazania_1", "crownbeard_1", "windflower_1"]
    else:
        archetype = "abstract_vertical_structure"
        concept = "ambiguous feeling suspended as a minimal vertical object"
        atmosphere = ["uncertain", "quiet", "pale", "isolated"]
        symbolic_object = "a cracked orb relic"
        planet_mesh_type = "basic_planet"
        planet_material_type = "purple"
        flower_color_group = "purple"
        flower_types = ["silver_downy_1", "leadwort_2", "bigleaf"]

    return {
        "world_concept": concept,
        "symbolic_object": symbolic_object,
        "asset_prompt": build_asset_prompt(symbolic_object),
        "negative_prompt": BASE_NEGATIVE_PROMPT,
        "archetype": archetype,
        "planet_mesh_type": planet_mesh_type,
        "planet_material_type": planet_material_type,
        "flower_color_group": flower_color_group,
        "flower_types": flower_types,
        "atmosphere": atmosphere,
        "emotional_state": {
            "valence": classify(valence),
            "arousal": classify(arousal),
            "dominance": classify(dominance),
        },
    }


def extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def call_gemini(config: dict[str, Any], prompt: str) -> dict[str, Any]:
    llm = config["llm"]
    api_key = os.environ.get(llm.get("api_key_env", "GEMINI_API_KEY"), "")
    if not api_key:
        raise RuntimeError("Gemini API key is missing. Set the configured environment variable.")

    model = llm.get("model", "gemini-1.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.7},
    }
    response = requests.post(
        url,
        params={"key": api_key},
        json=body,
        timeout=float(llm.get("timeout_sec", 30)),
    )
    response.raise_for_status()
    data = response.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    return extract_json(text)


def normalize_instruction(raw: dict[str, Any], world_number: int, eeg: dict[str, Any]) -> dict[str, Any]:
    world_id = f"world_{world_number:04d}"
    symbolic_object = str(raw.get("symbolic_object", "")).strip()
    if not symbolic_object:
        symbolic_object = symbolic_object_from_archetype(str(raw.get("archetype", "sealed_observatory")))

    return {
        "schema_version": 1,
        "world_id": world_id,
        "world_number": world_number,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "person_name": str(eeg.get("person_name") or eeg.get("name") or eeg.get("participant") or "unknown"),
        "world_concept": str(raw.get("world_concept", "silent emotional landmark")),
        "symbolic_object": symbolic_object,
        "asset_prompt": build_asset_prompt(symbolic_object),
        "negative_prompt": BASE_NEGATIVE_PROMPT,
        "archetype": str(raw.get("archetype", "sealed_observatory")),
        "planet_mesh_type": normalize_planet_mesh_type(str(raw.get("planet_mesh_type", "basic_planet"))),
        "planet_material_type": normalize_planet_material_type(str(raw.get("planet_material_type", "blue"))),
        "flower_color_group": normalize_flower_color_group(str(raw.get("flower_color_group", "purple"))),
        "flower_types": normalize_flower_types(raw.get("flower_types", [])),
        "atmosphere": list(raw.get("atmosphere", ["dreamlike", "misty", "sacred"])),
        "emotional_state": raw.get("emotional_state", {}),
        "world_style": eeg.get("world_style", {}),
        "source_eeg": eeg,
    }


def symbolic_object_from_archetype(archetype: str) -> str:
    options = {
        "sealed_observatory": "a sealed stone codex",
        "emotional_tower": "a blade-like stone relic",
        "abstract_vertical_structure": "a cracked orb relic",
        "symbolic_landmark": "a closed eye-shaped stone emblem",
    }
    return options.get(archetype, "a sealed stone relic")


def normalize_planet_material_type(material_type: str) -> str:
    allowed = {"yellow", "blue", "green", "orange", "purple", "gold", "red", "pink"}
    aliases = {
        "yello": "yellow",
        "greed": "green",
        "warm_orange": "orange",
        "cool_blue": "blue",
        "mist_gray": "blue",
        "pale_gold": "gold",
        "deep_violet": "purple",
    }
    value = material_type.strip()
    value = aliases.get(value, value)
    return value if value in allowed else "blue"


def normalize_planet_mesh_type(mesh_type: str) -> str:
    allowed = {"basic_planet", "sharp_planet", "smooth_planet", "complicated_planet", "simple_planet"}
    aliases = {
        "rock_planet_01": "basic_planet",
        "rock_planet_02": "complicated_planet",
        "cracked_planet_01": "sharp_planet",
        "smooth_planet_01": "smooth_planet",
    }
    value = mesh_type.strip()
    value = aliases.get(value, value)
    return value if value in allowed else "basic_planet"


def normalize_flower_color_group(group: str) -> str:
    value = group.strip()
    return value if value in {"purple", "red", "yellow"} else "purple"


def normalize_flower_types(values: Any) -> list[str]:
    allowed = {
        "silver_downy_1", "pentas_1", "pentas_2", "leadwort_1", "leadwort_2", "bigleaf",
        "silver_downy_2", "bougainv_1", "bougainv_2", "dianthus_1", "dianthus_2", "daisy_1", "daisy_2",
        "campion_1", "campion_2", "gazania_1", "gazania_2", "crownbeard_1", "crownbeard_2", "windflower_1", "windflower_2",
    }
    if not isinstance(values, list):
        return []
    flowers: list[str] = []
    for value in values:
        if isinstance(value, str) and value in allowed and value not in flowers:
            flowers.append(value)
    return flowers[:5]


def generate_instruction_from_eeg(
    config: dict[str, Any],
    config_dir: Path,
    eeg: dict[str, Any],
    out_dir_override: str | None = None,
    existing_world_dirs: list[Path] | None = None,
) -> tuple[dict[str, Any], Path]:
    eeg = normalize_eeg_payload(eeg)
    output_dir = out_dir_override or config.get("instruction_output_dir") or config.get("shared_output_dir")
    if not output_dir:
        raise KeyError("instruction_output_dir")
    out_dir = resolve_config_path(config_dir, output_dir)
    counter_path = resolve_config_path(config_dir, config.get("world_counter_path", "world_counter.txt"))
    existing_dirs = [out_dir]
    if existing_world_dirs:
        existing_dirs.extend(existing_world_dirs)
    world_number = next_world_number(counter_path, existing_dirs)

    if config.get("llm", {}).get("dry_run", True):
        interpreted = dry_run_interpretation(eeg)
    else:
        interpreted = call_gemini(config, build_llm_prompt(eeg))

    instruction = normalize_instruction(interpreted, world_number, eeg)
    output_path = out_dir / f"{instruction['world_id']}.json"
    atomic_write_json(output_path, instruction)
    return instruction, output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate symbolic world instructions from EEG JSON.")
    parser.add_argument("--config", default="config.example.json")
    parser.add_argument("--eeg", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    config_dir = config_path.parent
    config = load_json(config_path)
    eeg_path = resolve_config_path(config_dir, args.eeg or config["eeg_path"])

    eeg = load_json(eeg_path)
    _instruction, output_path = generate_instruction_from_eeg(config, config_dir, eeg, args.out_dir)
    print(f"[eeg_interpreter] wrote {output_path}")


if __name__ == "__main__":
    main()
