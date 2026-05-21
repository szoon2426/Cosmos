from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import websockets

try:
    from .generate_world_instruction import generate_instruction_from_eeg, load_json
except ImportError:
    from generate_world_instruction import generate_instruction_from_eeg, load_json


def extract_eeg_payload(message: str) -> dict[str, Any] | None:
    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    for key in ("eeg", "payload", "data", "result"):
        nested = data.get(key)
        if isinstance(nested, dict):
            data = nested
            break

    if "vad" in data:
        return data

    vad = {}
    for source_key, target_key in (
        ("valence", "valence"),
        ("arousal", "arousal"),
        ("dominance", "dominance"),
        ("v", "valence"),
        ("a", "arousal"),
        ("d", "dominance"),
    ):
        if source_key in data:
            vad[target_key] = data[source_key]

    if vad:
        normalized = dict(data)
        normalized["vad"] = vad
        return normalized

    return data


async def listen(config_path: Path, uri: str, out_dir: str | None, once: bool, person_name: str | None) -> None:
    config_dir = config_path.parent
    config = load_json(config_path)
    print(f"[eeg_interpreter] websocket -> {uri}")

    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            eeg = extract_eeg_payload(str(message))
            if eeg is None:
                print("[eeg_interpreter] skipped non-json websocket message")
                continue

            if person_name:
                eeg["person_name"] = person_name

            instruction, output_path = generate_instruction_from_eeg(config, config_dir, eeg, out_dir)
            print(f"[eeg_interpreter] wrote {output_path} ({instruction['person_name']} -> {instruction['world_id']})")

            if once:
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="Receive EEG JSON over WebSocket and write world instruction JSON.")
    parser.add_argument("--config", default="config.example.json")
    parser.add_argument("--uri", required=True, help="Example: ws://192.168.0.10:8765")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--person-name", default=None)
    args = parser.parse_args()

    asyncio.run(listen(Path(args.config), args.uri, args.out_dir, args.once, args.person_name))


if __name__ == "__main__":
    main()
