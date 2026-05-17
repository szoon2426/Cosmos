from __future__ import annotations

import argparse
import base64
import copy
import json
import math
import os
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from asset_generator import AssetGenerationRequest, AssetGenerationResult, generate_asset


PLACEHOLDER_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


FRONT_LOW_DETAIL_ASSETS = ["rock_path", "rock_s", "glow_sphere"]
MIDDLE_ASSETS = ["rock_m1", "rock_m2", "pond"]
BACK_SIDE_ASSETS = ["tree", "rock_l"]
FLOWER_TYPES = {
    "dreamlike": ["sea_thrift", "leadwort", "elderberry"],
    "misty": ["sea_thrift", "desert_cotton", "leadwort"],
    "soft": ["poppy", "sea_thrift", "desert_cotton"],
    "sacred": ["elderberry", "leadwort", "desert_cotton"],
    "tense": ["poppy", "leadwort", "elderberry"],
    "luminous": ["sea_thrift", "poppy", "leadwort"],
}


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


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def replace_template(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        for key, replacement in variables.items():
            value = value.replace("{" + key + "}", replacement)
        return value
    if isinstance(value, list):
        return [replace_template(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: replace_template(item, variables) for key, item in value.items()}
    return value


def json_path(path: Path) -> str:
    return Path(os.path.normpath(str(path))).as_posix()


@dataclass
class GeneratedAssetPaths:
    image_png: Path
    transparent_png: Path
    glb_path: Path
    provider: str = "unknown"
    ok: bool = True
    error: str | None = None


class ComfyUIClient:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.base_url = config.get("base_url", "http://127.0.0.1:8188").rstrip("/")
        self.session = requests.Session()

    def run_text_to_image(self, prompt: str, negative_prompt: str, output_path: Path) -> Path:
        workflow_path = Path(self.config["text_to_image_workflow"])
        workflow = load_json(workflow_path)
        self._set_node_text(workflow, self.config.get("prompt_node_id"), prompt)
        self._set_node_text(workflow, self.config.get("negative_prompt_node_id"), negative_prompt)
        return self._run_workflow_and_download(workflow, output_path)

    def run_rmbg(self, input_image: Path, output_path: Path) -> Path:
        workflow_path = Path(self.config["rmbg_workflow"])
        workflow = load_json(workflow_path)
        image_name = self._upload_image(input_image)
        image_node_id = self.config.get("image_input_node_id")
        if image_node_id and image_node_id in workflow:
            workflow[image_node_id].setdefault("inputs", {})["image"] = image_name
        return self._run_workflow_and_download(workflow, output_path)

    def run_full_asset_workflow(
        self,
        prompt: str,
        negative_prompt: str,
        image_output_path: Path,
        transparent_output_path: Path,
        glb_output_path: Path,
    ) -> None:
        workflow_path = Path(self.config["full_asset_workflow"])
        workflow = load_json(workflow_path)
        self._set_node_text(workflow, self.config.get("prompt_node_id"), prompt)
        self._set_node_text(workflow, self.config.get("negative_prompt_node_id"), negative_prompt)
        history_item = self._run_workflow(workflow)
        image_info = self._find_first_output_file(history_item, {".png", ".jpg", ".jpeg", ".webp"})
        glb_info = self._find_first_output_file(history_item, {".glb", ".gltf"})
        if image_info:
            self._download_file(image_info, image_output_path)
            self._download_file(image_info, transparent_output_path)
        if glb_info:
            self._download_file(glb_info, glb_output_path)
        if not glb_info:
            glb_output_path.write_text("ComfyUI finished, but no GLB output was found in history.\n", encoding="utf-8")

    def _set_node_text(self, workflow: dict[str, Any], node_id: str | None, text: str) -> None:
        if not node_id or node_id not in workflow:
            return
        inputs = workflow[node_id].setdefault("inputs", {})
        if "text" in inputs:
            inputs["text"] = text
        elif "prompt" in inputs:
            inputs["prompt"] = text
        else:
            inputs["text"] = text

    def _upload_image(self, image_path: Path) -> str:
        with image_path.open("rb") as f:
            files = {"image": (image_path.name, f, "image/png")}
            response = self.session.post(f"{self.base_url}/upload/image", files=files, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("name", image_path.name)

    def _queue_prompt(self, workflow: dict[str, Any]) -> str:
        body = {
            "prompt": workflow,
            "client_id": self.config.get("client_id", "cosmos_world_pipeline"),
        }
        response = self.session.post(f"{self.base_url}/prompt", json=body, timeout=30)
        response.raise_for_status()
        return response.json()["prompt_id"]

    def _run_workflow_and_download(self, workflow: dict[str, Any], output_path: Path) -> Path:
        history_item = self._run_workflow(workflow)
        image_info = self._find_first_image(history_item)
        if image_info:
            return self._download_image(image_info, output_path)
        raise RuntimeError("ComfyUI workflow finished, but no image output was found.")

    def _run_workflow(self, workflow: dict[str, Any]) -> dict[str, Any]:
        prompt_id = self._queue_prompt(workflow)
        deadline = time.time() + 600
        while time.time() < deadline:
            history = self.session.get(f"{self.base_url}/history/{prompt_id}", timeout=30)
            history.raise_for_status()
            data = history.json()
            if prompt_id in data:
                return data[prompt_id]
            time.sleep(1)
        raise TimeoutError(f"ComfyUI workflow timed out: {prompt_id}")

    def _find_first_image(self, history_item: dict[str, Any]) -> dict[str, Any] | None:
        outputs = history_item.get("outputs", {})
        for output in outputs.values():
            images = output.get("images") or []
            if images:
                return images[0]
        return None

    def _find_first_output_file(self, history_item: dict[str, Any], extensions: set[str]) -> dict[str, Any] | None:
        outputs = history_item.get("outputs", {})
        for output in outputs.values():
            for value in output.values():
                if not isinstance(value, list):
                    continue
                for item in value:
                    if not isinstance(item, dict):
                        continue
                    filename = str(item.get("filename", ""))
                    if Path(filename).suffix.lower() in extensions:
                        return item
        return None

    def _download_image(self, image_info: dict[str, Any], output_path: Path) -> Path:
        return self._download_file(image_info, output_path)

    def _download_file(self, file_info: dict[str, Any], output_path: Path) -> Path:
        params = {
            "filename": file_info["filename"],
            "subfolder": file_info.get("subfolder", ""),
            "type": file_info.get("type", "output"),
        }
        response = self.session.get(f"{self.base_url}/view", params=params, timeout=60)
        response.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)
        return output_path


class ComfyUIProvider:
    name = "comfyui"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def generate(self, request: AssetGenerationRequest) -> AssetGenerationResult:
        image_png = request.output_dir / f"{request.world_id}_source.png"
        transparent_png = request.output_dir / f"{request.world_id}_transparent.png"
        glb_path = request.output_dir / f"{request.world_id}.glb"
        comfy = ComfyUIClient(self.config)
        if self.config.get("full_asset_workflow"):
            comfy.run_full_asset_workflow(
                request.prompt,
                request.negative_prompt,
                image_png,
                transparent_png,
                glb_path,
            )
        else:
            comfy.run_text_to_image(request.prompt, request.negative_prompt, image_png)
            comfy.run_rmbg(image_png, transparent_png)
        return AssetGenerationResult(self.name, image_png, transparent_png, glb_path, glb_path.exists())


def generate_assets(config: dict[str, Any], instruction: dict[str, Any], asset_dir: Path) -> GeneratedAssetPaths:
    image_png = asset_dir / f"{instruction['world_id']}_source.png"
    transparent_png = asset_dir / f"{instruction['world_id']}_transparent.png"
    glb_path = asset_dir / f"{instruction['world_id']}.glb"
    provider_name = str(config.get("asset_provider", "comfyui"))

    if config.get("dry_run", True):
        image_png.write_bytes(PLACEHOLDER_PNG)
        transparent_png.write_bytes(PLACEHOLDER_PNG)
        glb_path.write_text("dry-run placeholder for generated GLB\n", encoding="utf-8")
        return GeneratedAssetPaths(image_png, transparent_png, glb_path, provider=provider_name, ok=True)

    request = AssetGenerationRequest(
        world_id=str(instruction["world_id"]),
        prompt=str(instruction["asset_prompt"]),
        negative_prompt=str(instruction["negative_prompt"]),
        output_dir=asset_dir,
    )
    if provider_name == "comfyui":
        result = generate_asset(request, ComfyUIProvider(config["comfyui"]))
    else:
        raise RuntimeError(f"Unknown asset provider: {provider_name}")

    if not result.ok:
        fallback = config.get("fallback", {})
        fallback_glb = str(fallback.get("glb_path", ""))
        if fallback.get("enabled", True) and fallback_glb:
            fallback_path = resolve_config_path(Path(config["_config_dir"]), fallback_glb)
            glb_path.write_bytes(fallback_path.read_bytes())
            return GeneratedAssetPaths(
                result.image_png,
                result.transparent_png,
                glb_path,
                provider=f"{provider_name}:fallback",
                ok=True,
                error=result.error,
            )
        return GeneratedAssetPaths(
            result.image_png,
            result.transparent_png,
            result.glb_path,
            provider=provider_name,
            ok=False,
            error=result.error or "Asset generation failed.",
        )

    return GeneratedAssetPaths(
        result.image_png,
        result.transparent_png,
        result.glb_path,
        provider=provider_name,
        ok=True,
        error=result.error,
    )


def import_generated_asset(config: dict[str, Any], instruction: dict[str, Any], paths: GeneratedAssetPaths) -> str | None:
    unreal_import = config.get("unreal_import", {})
    world_id = str(instruction["world_id"])
    destination_path = str(unreal_import.get("destination_path", "/Game/Generated/{world_id}")).replace("{world_id}", world_id)
    asset_name = str(unreal_import.get("asset_name", "SM_{world_id}_structure")).replace("{world_id}", world_id)
    asset_object_path = (
        str(unreal_import.get("asset_object_path", f"{destination_path}/{asset_name}.{asset_name}"))
        .replace("{world_id}", world_id)
        .replace("{destination_path}", destination_path)
        .replace("{asset_name}", asset_name)
    )

    if not unreal_import.get("enabled", False):
        print(f"[windows_pipeline] Unreal import skipped -> {paths.glb_path}")
        return asset_object_path

    config_dir = Path(config["_config_dir"])
    script_path = resolve_config_path(config_dir, unreal_import["script_path"])
    variables = {
        "script_path": json_path(script_path),
        "glb_path": json_path(paths.glb_path),
        "world_id": world_id,
        "destination_path": destination_path,
        "asset_name": asset_name,
    }
    command = replace_template(unreal_import["command"], variables)
    subprocess.run(command, check=True)
    print(f"[windows_pipeline] Unreal import finished -> {asset_object_path}")
    return asset_object_path


def choose_flowers(atmosphere: list[str]) -> list[str]:
    flowers: list[str] = []
    for keyword in atmosphere:
        for flower in FLOWER_TYPES.get(keyword, []):
            if flower not in flowers:
                flowers.append(flower)
    fallback = ["poppy", "sea_thrift", "desert_cotton", "leadwort", "elderberry"]
    for flower in fallback:
        if len(flowers) >= 5:
            break
        if flower not in flowers:
            flowers.append(flower)
    return flowers[:5]


def asset(
    asset_key: str,
    location: list[float],
    scale: list[float],
    zone: str,
    yaw: float = 0,
) -> dict[str, Any]:
    return {
        "asset_key": asset_key,
        "location": location,
        "rotation": [0, 0, yaw],
        "scale": scale,
    }


def rock_l_yaw(rng: random.Random, y: float) -> int:
    if y > 0:
        return rng.randint(50, 180)
    return rng.randint(-10, 120)


def choose_world_density(atmosphere: list[str]) -> float:
    if any(keyword in atmosphere for keyword in ("misty", "soft", "dreamlike")):
        return 0.74
    if any(keyword in atmosphere for keyword in ("tense", "charged")):
        return 0.58
    return 0.68


POND_SPOTS = [
    {"name": "left_near", "x": (-300, -90), "y": (210, 430)},
    {"name": "right_near", "x": (-280, -60), "y": (-430, -210)},
    {"name": "center_mid", "x": (-120, 140), "y": (-90, 120)},
    {"name": "left_far", "x": (80, 260), "y": (240, 470)},
    {"name": "right_far", "x": (80, 260), "y": (-470, -240)},
    {"name": "back_center", "x": (120, 280), "y": (-110, 130)},
]


def choose_pond_spots(rng: random.Random, atmosphere: list[str], pond_count: int) -> list[dict[str, Any]]:
    spots = list(POND_SPOTS)
    if "misty" in atmosphere or "dreamlike" in atmosphere:
        preferred_names = ["left_far", "right_far", "back_center", "left_near", "right_near"]
        spots.sort(key=lambda spot: preferred_names.index(spot["name"]) if spot["name"] in preferred_names else 99)
    else:
        rng.shuffle(spots)

    if pond_count == 1:
        return [rng.choice(spots)]
    if pond_count == 2:
        pairs = [
            ("left_near", "right_far"),
            ("right_near", "left_far"),
            ("left_far", "right_far"),
            ("left_near", "right_near"),
        ]
        left_name, right_name = rng.choice(pairs)
        by_name = {spot["name"]: spot for spot in spots}
        return [by_name[left_name], by_name[right_name]]

    by_name = {spot["name"]: spot for spot in spots}
    triplets = [
        ["left_near", "right_near", "back_center"],
        ["left_near", "center_mid", "right_far"],
        ["right_near", "center_mid", "left_far"],
        ["left_far", "right_far", "center_mid"],
    ]
    return [by_name[name] for name in rng.choice(triplets)]


def choose_pond_location(rng: random.Random, spot: dict[str, Any]) -> list[float]:
    return [rng.randint(*spot["x"]), rng.randint(*spot["y"]), 2]


def distance_2d(a: list[float], b: list[float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def is_behind_blocker(target: list[float], blocker: list[float], lane_width: float) -> bool:
    # Camera is around X=-900 looking toward +X. Larger X is farther/back.
    return target[0] > blocker[0] and abs(target[1] - blocker[1]) < lane_width


def blocks_any_pond(location: list[float], ponds: list[dict[str, Any]], lane_width: float) -> bool:
    return any(is_behind_blocker(pond["location"], location, lane_width) for pond in ponds)


def overlaps(location: list[float], radius: float, occupied: list[tuple[list[float], float]]) -> bool:
    return any(distance_2d(location, other_location) < radius + other_radius for other_location, other_radius in occupied)


def add_occupied(occupied: list[tuple[list[float], float]], location: list[float], radius: float) -> None:
    occupied.append((location, radius))


def pick_non_overlapping(
    rng: random.Random,
    x_range: tuple[int, int],
    y_range: tuple[int, int],
    radius: float,
    occupied: list[tuple[list[float], float]],
    attempts: int = 80,
) -> list[float] | None:
    for _ in range(attempts):
        location = [rng.randint(*x_range), rng.randint(*y_range), 0]
        if not overlaps(location, radius, occupied):
            add_occupied(occupied, location, radius)
            return location

    step_x = max(int(radius), 40)
    step_y = max(int(radius), 40)
    candidates: list[list[float]] = []
    for x in range(x_range[0], x_range[1] + 1, step_x):
        for y in range(y_range[0], y_range[1] + 1, step_y):
            candidates.append([x, y, 0])
    rng.shuffle(candidates)
    for location in candidates:
        if not overlaps(location, radius, occupied):
            add_occupied(occupied, location, radius)
            return location
    return None


def generate_ponds(
    rng: random.Random,
    atmosphere: list[str],
    occupied: list[tuple[list[float], float]],
) -> list[dict[str, Any]]:
    pond_count = rng.randint(1, 3)
    ponds: list[dict[str, Any]] = []
    spots = choose_pond_spots(rng, atmosphere, pond_count)
    for spot in spots:
        scale_x = round(rng.uniform(1.05, 1.75), 2)
        scale_y = round(scale_x * rng.uniform(1.0, 1.45), 2)
        if rng.random() < 0.25:
            scale_x, scale_y = scale_y, scale_x
        pond_scale = [scale_x, scale_y, round(rng.uniform(1.45, 2.15), 2)]
        radius = pond_collision_radius(pond_scale)
        for _attempt in range(80):
            location = choose_pond_location(rng, spot)
            # Pond may live near lateral grass edges, but avoid the front-center path lane.
            in_front_center_lane = location[0] < -300 and -110 < location[1] < 130
            if not in_front_center_lane and not overlaps(location, radius, occupied):
                add_occupied(occupied, location, radius)
                ponds.append(asset("pond", location, pond_scale, "middle_center"))
                break
    if not ponds:
        location = [-120, 170, 2]
        pond_scale = [1.35, 1.35, 1.8]
        add_occupied(occupied, location, 130)
        ponds.append(asset("pond", location, pond_scale, "middle_left"))
    return ponds


def pond_visual_radii(pond_scale: list[float]) -> tuple[float, float]:
    return 90 * pond_scale[0], 72 * pond_scale[1]


def pond_collision_radius(pond_scale: list[float]) -> float:
    radius_x, radius_y = pond_visual_radii(pond_scale)
    return max(radius_x, radius_y) + 115


def generate_pond_border_rocks(rng: random.Random, pond: dict[str, Any]) -> list[dict[str, Any]]:
    center_x, center_y, _ = pond["location"]
    radius_x, radius_y = pond_visual_radii(pond["scale"])
    rock_count = rng.randint(5, 8)
    # Left-side ponds look cleaner when the stone accents sit mostly on the far/top edge.
    left_pond = center_y < -140
    if left_pond:
        angles = [rng.uniform(70, 155) for _ in range(rock_count)]
    else:
        start_angle = rng.uniform(0, 360)
        angles = [start_angle + index * (360 / rock_count) + rng.uniform(-10, 10) for index in range(rock_count)]
    rocks: list[dict[str, Any]] = []
    for angle in angles:
        radians = angle * 3.141592653589793 / 180
        ring_x = center_x + (radius_x + rng.uniform(22, 34)) * math.cos(radians)
        ring_y = center_y + (radius_y + rng.uniform(20, 32)) * math.sin(radians)
        scale_x = round(rng.uniform(0.42, 0.72), 2)
        scale_y = round(rng.uniform(0.34, 0.62), 2)
        rocks.append(
            asset(
                "rock_s",
                [round(ring_x), round(ring_y), 0],
                [scale_x, scale_y, 0.65],
                "pond_border",
                round(angle + rng.uniform(-25, 25)),
            )
        )
    return rocks


def place_fountain(
    rng: random.Random,
    ponds: list[dict[str, Any]],
    occupied: list[tuple[list[float], float]],
) -> dict[str, Any]:
    # Fountain is mandatory, but it must stay comfortably inside the grass plane.
    candidates = [
        [-500, -240, 0],
        [-500, 240, 0],
        [-420, -260, 0],
        [-420, 260, 0],
        [-300, -260, 0],
        [-300, 260, 0],
        [-180, -220, 0],
        [-180, 220, 0],
    ]
    rng.shuffle(candidates)
    for location in candidates:
        blocks_pond = blocks_any_pond(location, ponds, 130)
        if not blocks_pond and not overlaps(location, 95, occupied):
            add_occupied(occupied, location, 95)
            x = location[0]
            scale_value = round(1.0 + max(min(x, -150) - (-520), 0) / 370 * 0.3, 2)
            return asset("fountain", location, [scale_value, scale_value, scale_value], "middle_side", 145)

    # Mandatory fallback: still keep margin from the grass edge.
    location = [-500, 240, 0]
    add_occupied(occupied, location, 95)
    return asset("fountain", location, [1.0, 1.0, 1.0], "middle_side", 145)


def place_statue(
    rng: random.Random,
    ponds: list[dict[str, Any]],
    occupied: list[tuple[list[float], float]],
) -> dict[str, Any]:
    sorted_ponds = sorted(ponds, key=lambda pond: abs(pond["location"][1]))
    candidates: list[list[float]] = []
    for pond in sorted_ponds:
        pond_x, pond_y, _ = pond["location"]
        for dx in (220, 280, 340):
            for dy in (0, 110, -110, 170, -170):
                candidates.append([min(310, round(pond_x + dx)), round(pond_y + dy), 0])
    candidates.extend([[280, 220, 0], [260, -220, 0], [80, 300, 0], [80, -300, 0]])

    for location in candidates:
        if location[0] > 310:
            continue
        if not overlaps(location, 85, occupied):
            add_occupied(occupied, location, 85)
            statue_scale = round(1.25 + max(location[0] - 30, 0) / 280 * 0.10, 2)
            return asset("statue", location, [statue_scale, statue_scale, statue_scale], "back_side", -85)

    # Mandatory fallback: prefer a readable side position over omission.
    location = [300, 360, 0]
    add_occupied(occupied, location, 85)
    return asset("statue", location, [1.35, 1.35, 1.35], "back_side", -85)


def place_medium_rocks(
    rng: random.Random,
    ponds: list[dict[str, Any]],
    occupied: list[tuple[list[float], float]],
) -> list[dict[str, Any]]:
    candidates = [
        ("rock_m1", [-300, -360, 0], [1.08, 0.84, 1.0], -25),
        ("rock_m1", [-300, 360, 0], [1.04, 0.82, 1.0], 42),
        ("rock_m2", [320, -360, 0], [1.02, 0.8, 1.0], 68),
        ("rock_m2", [320, 360, 0], [0.98, 0.78, 1.0], 24),
        ("rock_m1", [460, -120, 0], [1.0, 0.82, 1.0], -18),
        ("rock_m1", [460, 160, 0], [1.02, 0.82, 1.0], 36),
    ]
    rng.shuffle(candidates)
    rocks: list[dict[str, Any]] = []
    for asset_key, location, scale, yaw in candidates:
        if len(rocks) >= 3:
            break
        if blocks_any_pond(location, ponds, 145):
            continue
        if overlaps(location, 65, occupied):
            continue
        add_occupied(occupied, location, 65)
        rocks.append(asset(asset_key, location, scale, "middle_side", yaw))
    return rocks


def generate_rock_path_to_pond(rng: random.Random, pond: dict[str, Any]) -> list[dict[str, Any]]:
    pond_location = pond["location"]
    pond_scale = pond["scale"]
    pond_x, pond_y, _ = pond_location
    pond_radius_x, _ = pond_visual_radii(pond_scale)
    stop_x = pond_x - pond_radius_x - 55
    start_x = -780
    if stop_x <= start_x + 120:
        stop_x = start_x + 120

    path_count = max(3, min(6, int((stop_x - start_x) / 110) + 1))
    assets: list[dict[str, Any]] = []
    for index in range(path_count):
        t = index / max(path_count - 1, 1)
        x = start_x + (stop_x - start_x) * t
        y = 40 + (pond_y - 40) * t
        x += rng.randint(-12, 12)
        y += rng.randint(-8, 8)
        scale_x = round(rng.uniform(0.92, 1.08), 2)
        assets.append(
            asset(
                "rock_path",
                [round(x), round(y), 0],
                [scale_x, 0.75, 1.0],
                "front_center",
                rng.randint(-8, 8),
            )
        )
    return assets


def generate_world_json(
    config: dict[str, Any],
    instruction: dict[str, Any],
    paths: GeneratedAssetPaths,
    generated_mesh_asset_path: str | None,
    world_dir: Path,
) -> Path:
    world_number = int(instruction["world_number"])
    world_space = int(config.get("world_space", 2500))
    world_offset = world_number * world_space
    seed = random.Random(instruction["world_id"]).randint(1000, 999999)
    rng = random.Random(seed)
    atmosphere = list(instruction.get("atmosphere", []))
    occupied: list[tuple[list[float], float]] = []
    ponds = generate_ponds(rng, atmosphere, occupied)
    primary_pond = min(ponds, key=lambda item: item["location"][0])

    # Asset locations are local to BP_World. Unreal places the whole world
    # with world_number * world_space, so do not add world_offset here.
    assets = generate_rock_path_to_pond(rng, primary_pond) + [
        # Front zones: small/low-detail only. Rock path leads toward pond and stops before it.
        asset("rock_s", [-500, 220, 0], [0.82, 0.66, 0.8], "front_left", rng.randint(-30, 60)),
        asset("rock_s", [80, -280, 0], [0.72, 0.62, 0.8], "front_right", rng.randint(-60, 30)),
        asset("glow_sphere", [-620, -120, 25], [1.0, 1.0, 1.0], "front_right"),
        asset("glow_sphere", [-260, 300, 25], [1.0, 1.0, 1.0], "front_left"),

        # Middle zones: ponds are generated procedurally and kept out of the front-center lane.
        *ponds,
    ]
    assets.extend(place_medium_rocks(rng, ponds, occupied))

    fountain = place_fountain(rng, ponds, occupied)
    assets.append(fountain)

    for pond in ponds:
        assets.extend(generate_pond_border_rocks(rng, pond))

    structure_location = pick_non_overlapping(rng, (520, 700), (-80, 110), 120, occupied)
    if structure_location:
        generated_structure = asset(
            "generated_symbolic_structure",
            structure_location,
            [1.0, 1.0, 1.0],
            "back_center",
        )
        if generated_mesh_asset_path:
            generated_structure["mesh_asset_path"] = generated_mesh_asset_path
        assets.append(generated_structure)

    tree_count = rng.randint(2, 5)
    for _ in range(tree_count):
        y_range = (-330, -170) if rng.random() < 0.5 else (170, 330)
        location = pick_non_overlapping(rng, (500, 700), y_range, 95, occupied)
        if location:
            scale_value = round(rng.uniform(1.08, 1.45), 2)
            assets.append(asset("tree", location, [scale_value, scale_value, scale_value], "side"))

    assets.append(place_statue(rng, ponds, occupied))

    for x_range, y_range in [((80, 260), (180, 330)), ((340, 540), (-330, -160))]:
        rock_location = None
        for _ in range(60):
            candidate = [rng.randint(*x_range), rng.randint(*y_range), 0]
            if blocks_any_pond(candidate, ponds, 170) or overlaps(candidate, 70, occupied):
                continue
            rock_location = candidate
            add_occupied(occupied, rock_location, 70)
            break
        if rock_location:
            assets.append(
                asset(
                    "rock_l",
                    rock_location,
                    [round(rng.uniform(1.05, 1.25), 2), round(rng.uniform(0.84, 0.96), 2), 1.0],
                    "side",
                    rock_l_yaw(rng, rock_location[1]),
                )
            )

    world = {
        "world_id": instruction["world_id"],
        "world_number": world_number,
        "world_name": instruction.get("world_concept", instruction["world_id"]),
        "seed": seed,
        "flower_density": choose_world_density(atmosphere),
        "flower_type": choose_flowers(atmosphere),
        "assets": assets,
    }

    manifest = {
        "schema_version": 1,
        "world_id": instruction["world_id"],
        "world_number": world_number,
        "world_offset": world_offset,
        "world_space": world_space,
        "seed": seed,
        "pcg_seed": seed,
        "flower_type": world["flower_type"],
        "flower_density": world["flower_density"],
        "atmosphere": atmosphere,
        "archetype": instruction.get("archetype"),
        "world_concept": instruction.get("world_concept"),
        "generated_assets": {
            "image_png": json_path(paths.image_png),
            "transparent_png": json_path(paths.transparent_png),
            "glb_mesh": json_path(paths.glb_path),
            "unreal_asset_path": generated_mesh_asset_path,
            "provider": paths.provider,
            "ok": paths.ok,
            "error": paths.error,
        },
        "zoning_rules": {
            "front": "low-detail objects only",
            "middle": "medium structures only",
            "back_side": "large silhouettes only",
            "large_structures_in_front": False,
        },
    }

    manifest_path = world_dir / f"{instruction['world_id']}_manifest.json"
    atomic_write_json(manifest_path, manifest)

    configured_path = config.get("world_json_path")
    if configured_path:
        path = resolve_config_path(Path(config["_config_dir"]), configured_path)
    else:
        path = world_dir / f"{instruction['world_id']}_world.json"
    atomic_write_json(path, world)
    return path


def call_unreal(config: dict[str, Any], world_json_path: Path) -> None:
    unreal = config.get("unreal", {})
    if not unreal.get("enabled", False):
        print(f"[windows_pipeline] Unreal call skipped -> {world_json_path}")
        return

    base_url = unreal.get("base_url", "http://localhost:30010").rstrip("/")
    endpoint = unreal.get("endpoint", "/remote/object/call")
    variables = {"world_json_path": json_path(world_json_path)}
    body = replace_template(copy.deepcopy(unreal.get("body_template", {})), variables)
    response = requests.post(
        f"{base_url}{endpoint}",
        json=body,
        timeout=float(unreal.get("timeout_sec", 5)),
    )
    response.raise_for_status()
    print(f"[windows_pipeline] Unreal SpawnWorld triggered -> {world_json_path}")


def process_instruction(config: dict[str, Any], instruction_path: Path) -> Path:
    instruction = load_json(instruction_path)
    output_root = ensure_dir(resolve_config_path(Path(config["_config_dir"]), config["output_dir"]))
    world_dir = ensure_dir(output_root / instruction["world_id"])
    asset_dir = ensure_dir(world_dir / "assets")

    print(f"[windows_pipeline] processing {instruction_path.name}")
    paths = generate_assets(config, instruction, asset_dir)
    generated_mesh_asset_path = import_generated_asset(config, instruction, paths)
    world_json_path = generate_world_json(config, instruction, paths, generated_mesh_asset_path, world_dir)
    call_unreal(config, world_json_path)
    return world_json_path


def load_processed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def mark_processed(path: Path, instruction_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(str(instruction_path.resolve()) + "\n")


def watch(config: dict[str, Any]) -> None:
    config_dir = Path(config["_config_dir"])
    input_dir = ensure_dir(resolve_config_path(config_dir, config["shared_input_dir"]))
    state_path = ensure_dir(resolve_config_path(config_dir, config["output_dir"])) / ".processed"
    processed = load_processed(state_path)
    poll_interval = float(config.get("poll_interval_sec", 2.0))
    print(f"[windows_pipeline] watching {input_dir.resolve()}")

    while True:
        for path in sorted(input_dir.glob("*.json")):
            key = str(path.resolve())
            if key in processed:
                continue
            try:
                process_instruction(config, path)
                mark_processed(state_path, path)
                processed.add(key)
            except Exception as exc:
                print(f"[windows_pipeline] error processing {path.name}: {exc}")
        time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch world instructions and generate Unreal world JSON.")
    parser.add_argument("--config", default="config.example.json")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--file", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_json(config_path)
    config["_config_dir"] = str(config_path.parent)
    if args.file:
        world_json_path = process_instruction(config, Path(args.file))
        print(f"[windows_pipeline] wrote {world_json_path}")
        return
    if args.once:
        input_dir = ensure_dir(resolve_config_path(config_path.parent, config["shared_input_dir"]))
        candidates = sorted(input_dir.glob("*.json"))
        if not candidates:
            print(f"[windows_pipeline] no instruction JSON found in {input_dir}")
            return
        world_json_path = process_instruction(config, candidates[-1])
        print(f"[windows_pipeline] wrote {world_json_path}")
        return
    watch(config)


if __name__ == "__main__":
    main()
