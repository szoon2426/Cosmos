from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class AssetGenerationRequest:
    world_id: str
    prompt: str
    negative_prompt: str
    output_dir: Path


@dataclass
class AssetGenerationResult:
    provider: str
    image_png: Path
    transparent_png: Path
    glb_path: Path
    ok: bool
    error: str | None = None


class AssetProvider(Protocol):
    name: str

    def generate(self, request: AssetGenerationRequest) -> AssetGenerationResult:
        ...


def generate_asset(request: AssetGenerationRequest, provider: AssetProvider) -> AssetGenerationResult:
    return provider.generate(request)


class NotConfiguredProvider:
    def __init__(self, name: str) -> None:
        self.name = name

    def generate(self, request: AssetGenerationRequest) -> AssetGenerationResult:
        image_png = request.output_dir / f"{request.world_id}_source.png"
        transparent_png = request.output_dir / f"{request.world_id}_transparent.png"
        glb_path = request.output_dir / f"{request.world_id}.glb"
        return AssetGenerationResult(
            provider=self.name,
            image_png=image_png,
            transparent_png=transparent_png,
            glb_path=glb_path,
            ok=False,
            error=f"{self.name} provider is not configured yet.",
        )
