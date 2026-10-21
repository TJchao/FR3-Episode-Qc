"""Pydantic models for the episode QC API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QCRequest(BaseModel):
    episode_path: str = Field(min_length=1)
    dataset_version: str = Field(default="unversioned", max_length=128)
    git_commit: str = Field(default="unknown", max_length=128)

