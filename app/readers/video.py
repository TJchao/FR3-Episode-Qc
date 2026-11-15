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

    def frame_at(self, timestamp_s: float) -> np.ndarray | None:
        capture = cv2.VideoCapture(str(self.path))
        try:
            capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_s) * 1000.0)
            ok, frame = capture.read()
            if ok and frame is not None:
                return frame
        finally:
            capture.release()
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
