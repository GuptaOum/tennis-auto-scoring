"""Streaming video I/O.

The baseline read every frame of a video into a Python list before doing any
work, which caps the tool at clips short enough to fit in RAM - a 1080p match
at 30 fps is roughly 6 MB per frame decoded, so an hour of footage would need
hundreds of gigabytes. It also hardcoded 24 fps in three places while its own
sample video is 30, making every speed and duration it reported wrong.

Both problems are the same fix: read metadata from the file, and stream.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0

    def __str__(self) -> str:
        return (
            f"{self.path.name}: {self.width}x{self.height} @ {self.fps:g} fps, "
            f"{self.frame_count} frames ({self.duration_seconds:.1f}s)"
        )


def probe(path: str | Path) -> VideoInfo:
    """Read video metadata without decoding the whole file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Some containers report 0 or a nonsense fps; 30 is the safer default
        # for handheld and phone footage than the baseline's hardcoded 24.
        if not fps or fps != fps or fps <= 1:
            fps = 30.0
        return VideoInfo(
            path=path,
            fps=float(fps),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def frames(
    path: str | Path,
    start: int = 0,
    limit: int | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, frame)`` one at a time. Never holds more than one."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {path}")
    try:
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        index = start
        emitted = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield index, frame
            index += 1
            emitted += 1
            if limit is not None and emitted >= limit:
                break
    finally:
        cap.release()


class VideoWriter:
    """Frame-at-a-time writer, so output never accumulates in memory either.

    Defaults to mp4v/.mp4 rather than the baseline's MJPG/.avi: MJPG stores
    every frame as an independent JPEG, which made its sample output 45 MB for
    7 seconds and unplayable in most browsers.
    """

    def __init__(
        self,
        path: str | Path,
        fps: float,
        width: int,
        height: int,
        fourcc: str = "mp4v",
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (width, height)
        )
        if not self._writer.isOpened():
            raise ValueError(f"could not open video writer for {path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
