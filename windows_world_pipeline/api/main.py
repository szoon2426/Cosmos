from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status

from windows_world_pipeline.api.models import EEGEmotionsPayload, GenerateWorldResponse
from windows_world_pipeline.api.world_service import DuplicateWorldError, WorldService

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "windows_world_pipeline" / "config.example.json"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    config_path = Path(os.environ.get("PIPELINE_CONFIG", str(DEFAULT_CONFIG_PATH))).expanduser()
    app.state.service = WorldService(config_path)
    try:
        yield
    finally:
        app.state.service.close()


app = FastAPI(title="Cosmos World Pipeline API", lifespan=lifespan)


@app.post(
    "/generate",
    response_model=GenerateWorldResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        409: {"description": "Generated world id already exists"},
        422: {"description": "Invalid EEG result payload"},
        500: {"description": "World generation failed"},
    },
)
def generate_world(
    request: Request,
    payload: EEGEmotionsPayload,
) -> GenerateWorldResponse:
    service: WorldService = request.app.state.service
    try:
        return service.generate(payload)
    except DuplicateWorldError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"World generation failed: {exc}") from exc


@app.post(
    "/generate/{_deprecated_world_id}",
    response_model=GenerateWorldResponse,
    status_code=status.HTTP_201_CREATED,
    deprecated=True,
)
def generate_world_deprecated_path(
    request: Request,
    payload: EEGEmotionsPayload,
    _deprecated_world_id: str,
) -> GenerateWorldResponse:
    return generate_world(request, payload)
