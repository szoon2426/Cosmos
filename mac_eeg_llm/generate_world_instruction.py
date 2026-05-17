from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


SYSTEM_RULES = """
You interpret EEG-derived emotional features into a single symbolic asset generation brief.
Return only valid JSON.

Aesthetic rules:
- Structures should feel emotionally symbolic.
- Avoid explicit fantasy objects.
- Structures should have ambiguous function.
- Avoid open architectural scenes.
- Prefer closed silhouettes.
- Structures should feel isolated, sacred, dreamlike, minimal.

Strong visual constraints:
- isolated structure
- single object
- floating isolated object when emotionally appropriate
- clean bottom
- single centered structure
- viewed from outside
- object showcase
- closed silhouette
- ambiguous function

Preferred structure types:
- sealed observatory
- emotional tower
- abstract vertical structure
- symbolic landmark

Avoid:
- castles
- cities
- large environments
- interior shots
- wide cinematic scenes
- explicit fantasy storytelling
- stairs
- platforms
"""


BASE_NEGATIVE_PROMPT = (
    "castle, city, village, large environment, landscape, interior, open architecture, "
    "wide cinematic scene, fantasy character, dragon, weapon, stairs, platform, bridge, "
    "crowd, multiple buildings, clutter, text, watermark, logo, low quality"
)


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


def next_world_number(counter_path: Path) -> int:
    counter_path.parent.mkdir(parents=True, exist_ok=True)
    if counter_path.exists():
        raw = counter_path.read_text(encoding="utf-8").strip()
        current = int(raw or "0")
    else:
        current = 0
    value = current + 1
    counter_path.write_text(str(value), encoding="utf-8")
    return value


def build_llm_prompt(eeg: dict[str, Any]) -> str:
    return (
        f"{SYSTEM_RULES}\n\n"
        "EEG JSON:\n"
        f"{json.dumps(eeg, ensure_ascii=False, indent=2)}\n\n"
        "Return this exact JSON shape:\n"
        "{\n"
        '  "world_concept": "short poetic concept",\n'
        '  "asset_prompt": "image/3D asset prompt",\n'
        f'  "negative_prompt": "{BASE_NEGATIVE_PROMPT}",\n'
        '  "archetype": "sealed_observatory | emotional_tower | abstract_vertical_structure | symbolic_landmark",\n'
        '  "atmosphere": ["keyword", "keyword", "keyword"],\n'
        '  "emotional_state": {"valence": "low|mid|high", "arousal": "low|mid|high", "dominance": "low|mid|high"}\n'
        "}\n"
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
    elif arousal > 0.62:
        archetype = "emotional_tower"
        concept = "compressed pulse rising into a solitary emotional tower"
        atmosphere = ["tense", "luminous", "focused", "charged"]
    elif dominance > 0.62:
        archetype = "symbolic_landmark"
        concept = "still confidence condensed into a closed symbolic landmark"
        atmosphere = ["clear", "solemn", "balanced", "monolithic"]
    else:
        archetype = "abstract_vertical_structure"
        concept = "ambiguous feeling suspended as a minimal vertical object"
        atmosphere = ["uncertain", "quiet", "pale", "isolated"]

    brightness = "warm pale material" if valence >= 0.5 else "cool muted material"
    asset_prompt = (
        f"single centered {archetype.replace('_', ' ')}, isolated structure, closed silhouette, "
        f"ambiguous function, viewed from outside, object showcase, clean bottom, {brightness}, "
        "minimal sacred dreamlike form, readable game asset, no environment"
    )

    return {
        "world_concept": concept,
        "asset_prompt": asset_prompt,
        "negative_prompt": BASE_NEGATIVE_PROMPT,
        "archetype": archetype,
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
    negative = raw.get("negative_prompt") or BASE_NEGATIVE_PROMPT
    return {
        "schema_version": 1,
        "world_id": world_id,
        "world_number": world_number,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "world_concept": str(raw.get("world_concept", "silent emotional landmark")),
        "asset_prompt": str(raw.get("asset_prompt", "")).strip(),
        "negative_prompt": str(negative).strip(),
        "archetype": str(raw.get("archetype", "sealed_observatory")),
        "atmosphere": list(raw.get("atmosphere", ["dreamlike", "misty", "sacred"])),
        "emotional_state": raw.get("emotional_state", {}),
        "source_eeg": eeg,
    }


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
    out_dir = resolve_config_path(config_dir, args.out_dir or config["shared_output_dir"])
    counter_path = resolve_config_path(config_dir, config.get("world_counter_path", "world_counter.txt"))

    eeg = load_json(eeg_path)
    world_number = next_world_number(counter_path)

    if config.get("llm", {}).get("dry_run", True):
        interpreted = dry_run_interpretation(eeg)
    else:
        interpreted = call_gemini(config, build_llm_prompt(eeg))

    instruction = normalize_instruction(interpreted, world_number, eeg)
    output_path = out_dir / f"{instruction['world_id']}.json"
    atomic_write_json(output_path, instruction)
    print(f"[mac_eeg_llm] wrote {output_path}")


if __name__ == "__main__":
    main()
