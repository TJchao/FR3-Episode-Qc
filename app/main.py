"""FastAPI application for FR3 episode quality checks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.db import get_qc_run, insert_qc_run, list_qc_runs
from app.qc import EpisodePathError, check_episode, resolve_episode_path
from app.schemas import QCRequest


app = FastAPI(
    title="fr3-episode-qc",
    description=(
        "Lightweight quality checks for FR3 robot episode data. "
        "Synthetic episodes are used because original robot data is not published."
    ),
    version="0.1.0",
)


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    _request,
    exc: RequestValidationError,
) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"detail": jsonable_encoder(exc.errors())},
    )


def _allowed_root() -> Path:
    configured_root = os.getenv("QC_ALLOWED_ROOT")
    if configured_root:
        return Path(configured_root).expanduser()

    docker_data_root = Path("/data")
    if docker_data_root.is_dir():
        return docker_data_root

    return Path.cwd()


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "allowed_root": str(_allowed_root()),
    }


@app.post("/qc")
def run_qc(request: QCRequest) -> dict[str, object]:
    allowed_root = _allowed_root()

    try:
        resolve_episode_path(request.episode_path, allowed_root)
    except EpisodePathError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    report = check_episode(request.episode_path, allowed_root)
    run_id = insert_qc_run(
        report,
        dataset_version=request.dataset_version,
        git_commit=request.git_commit,
    )

    return {
        "run_id": run_id,
        "dataset_version": request.dataset_version,
        "git_commit": request.git_commit,
        **report,
    }


@app.get("/runs")
def read_runs(
    limit: int = Query(default=100, ge=1, le=1000),
    status: Literal["passed", "failed"] | None = Query(default=None),
    dataset_version: str | None = Query(
        default=None,
        min_length=1,
        max_length=128,
    ),
) -> list[dict[str, object]]:
    if dataset_version is not None and not dataset_version.strip():
        raise HTTPException(
            status_code=400,
            detail="dataset_version must be a non-empty string",
        )

    return list_qc_runs(
        limit=limit,
        status=status,
        dataset_version=dataset_version,
    )


@app.get("/runs/{run_id}")
def read_run(run_id: int) -> dict[str, object]:
    if run_id < 1:
        raise HTTPException(status_code=400, detail="run_id must be positive")

    run = get_qc_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="QC run not found")

    return run
