"""Video decoding with a conservative fallback for AV1 files."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


class VideoDecodeError(RuntimeError):
    """Raised when neither OpenCV nor ffmpeg can decode one frame."""


class VideoImageSource:
    """Seek one frame at a time without retaining the full video."""

    def __init__(
        self,
        path: str | Path,
        *,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.width = width
        self.height = height
        self._capture: cv2.VideoCapture | None = None
        self._last_timestamp: float | None = None

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            self._last_timestamp = None

    def __del__(self) -> None:
        self.close()

    def frame_at(self, timestamp_s: float) -> np.ndarray | None:
        requested = max(0.0, timestamp_s)
        if (
            self._capture is None
            or self._last_timestamp is None
            or requested < self._last_timestamp
        ):
            self.close()
            self._capture = cv2.VideoCapture(str(self.path))

        if self._capture is not None:
            self._capture.set(cv2.CAP_PROP_POS_MSEC, requested * 1000.0)
            ok, frame = self._capture.read()
            if ok and frame is not None:
                self._last_timestamp = requested
                return frame
            self.close()
        return self._ffmpeg_frame(timestamp_s)

    def _ffmpeg_frame(self, timestamp_s: float) -> np.ndarray | None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            return None
        command = [
            ffmpeg,
            "-nostdin",
            "-loglevel",
            "error",
            "-ss",
            f"{max(0.0, timestamp_s):.6f}",
            "-i",
            str(self.path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0 or not result.stdout:
            return None
        # Prefer dimensions declared by the dataset feature. Fallbacks cover
        # ad-hoc videos passed directly to the CLI.
        count = len(result.stdout)
        candidates: list[tuple[int, int]] = []
        if self.width and self.height:
            candidates.append((self.width, self.height))
        candidates.extend(((640, 480), (480, 640), (320, 240), (1280, 720)))
        for width, height in candidates:
            if width * height == count:
                return np.frombuffer(result.stdout, dtype=np.uint8).reshape(
                    height, width
                )
        raise VideoDecodeError(
            f"ffmpeg returned {count} pixels for an unsupported image size"
        )
