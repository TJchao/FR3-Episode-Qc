"""Shared reader data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class ImageSource(Protocol):
    """A minimal streamable image source."""

    def frame_at(self, timestamp_s: float) -> np.ndarray | None:
        """Return one BGR or grayscale frame at a timestamp."""


@dataclass
class EpisodeData:
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]
    image_source: ImageSource | None = None
