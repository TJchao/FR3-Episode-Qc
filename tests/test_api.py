"""Tests for the FastAPI episode QC service."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import app


EXAMPLES_ROOT = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture()
def client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    monkeypatch.setenv("QC_ALLOWED_ROOT", str(EXAMPLES_ROOT))
    monkeypatch.setenv("QC_DB_PATH", str(tmp_path / "qc_history.db"))

    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_qc_valid_episode(client: TestClient) -> None:
    response = client.post(
        "/qc",
        json={"episode_path": str(EXAMPLES_ROOT / "ep_valid")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "passed"
    assert body["run_id"] == 1
    assert body["episode_id"] == "ep_valid"
    assert body["episode_hash"].startswith("sha256:")


def test_qc_invalid_episode_is_recorded(client: TestClient) -> None:
    response = client.post(
        "/qc",
        json={"episode_path": str(EXAMPLES_ROOT / "ep_camera_age_p95")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert "camera age P95 must be below 50.0ms" in body["errors"]
    assert body["run_id"] == 1


def test_qc_missing_request_field_returns_400(client: TestClient) -> None:
    response = client.post("/qc", json={})

    assert response.status_code == 400


def test_qc_missing_episode_returns_404(client: TestClient) -> None:
    response = client.post(
        "/qc",
        json={"episode_path": str(EXAMPLES_ROOT / "ep_not_exists")},
    )

    assert response.status_code == 404


def test_qc_outside_root_returns_403(
    client: TestClient,
    tmp_path: Path,
) -> None:
    outside_episode = tmp_path / "outside"
    outside_episode.mkdir()

    response = client.post(
        "/qc",
        json={"episode_path": str(outside_episode)},
    )

    assert response.status_code == 403


def test_runs_history(client: TestClient) -> None:
    client.post(
        "/qc",
        json={"episode_path": str(EXAMPLES_ROOT / "ep_valid")},
    )
    client.post(
        "/qc",
        json={"episode_path": str(EXAMPLES_ROOT / "ep_nan_value")},
    )

    response = client.get("/runs")

    assert response.status_code == 200
    runs = response.json()
    assert len(runs) == 2
    assert runs[0]["episode_id"] == "ep_nan_value"
    assert runs[0]["status"] == "failed"
    assert runs[1]["episode_id"] == "ep_valid"
    assert runs[1]["status"] == "passed"


def test_runs_filters_by_status_and_dataset_version(
    client: TestClient,
) -> None:
    client.post(
        "/qc",
        json={
            "episode_path": str(EXAMPLES_ROOT / "ep_valid"),
            "dataset_version": "v1",
        },
    )
    client.post(
        "/qc",
        json={
            "episode_path": str(EXAMPLES_ROOT / "ep_nan_value"),
            "dataset_version": "v2",
        },
    )

    passed = client.get("/runs", params={"status": "passed"})
    failed = client.get("/runs", params={"status": "failed"})
    v1 = client.get("/runs", params={"dataset_version": "v1"})
    failed_v2 = client.get(
        "/runs",
        params={"status": "failed", "dataset_version": "v2"},
    )

    assert [run["episode_id"] for run in passed.json()] == ["ep_valid"]
    assert [run["episode_id"] for run in failed.json()] == ["ep_nan_value"]
    assert [run["episode_id"] for run in v1.json()] == ["ep_valid"]
    assert [run["episode_id"] for run in failed_v2.json()] == ["ep_nan_value"]


def test_runs_rejects_invalid_filter(client: TestClient) -> None:
    invalid_status = client.get("/runs", params={"status": "invalid"})
    empty_version = client.get("/runs", params={"dataset_version": " "})

    assert invalid_status.status_code == 400
    assert empty_version.status_code == 400


def test_get_run_by_id(client: TestClient) -> None:
    created = client.post(
        "/qc",
        json={"episode_path": str(EXAMPLES_ROOT / "ep_valid")},
    )
    run_id = created.json()["run_id"]

    response = client.get(f"/runs/{run_id}")

    assert response.status_code == 200
    assert response.json()["id"] == run_id
    assert response.json()["report"]["status"] == "passed"


def test_get_missing_run_returns_404(client: TestClient) -> None:
    response = client.get("/runs/999999")

    assert response.status_code == 404
