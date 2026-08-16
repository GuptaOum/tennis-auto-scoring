"""Ball trajectory: gap filling and smoothing.

The detector finds the ball in roughly 45% of frames on real footage - it is
small, fast, and motion-blurred exactly when it matters. Everything downstream
(bounces, rallies, points) reads the trajectory rather than raw detections, so
this module is where missing frames are handled honestly:

- short gaps are interpolated, because the ball's flight is smooth and a few
  missing frames are recoverable
- long gaps are left as holes, because the ball genuinely was not there (or the
  detector genuinely lost it) and inventing a path across a second of video
  would manufacture bounces that never happened

The distinction is what keeps the interpolation from becoming fiction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BallSample:
    frame: int
    image: np.ndarray   # (2,) pixels
    court: np.ndarray   # (2,) metres on the court plane
    confidence: float
    interpolated: bool = False


class Trajectory:
    """A ball path over time, indexed by frame."""

    def __init__(self, samples: list[BallSample]) -> None:
        self.samples = sorted(samples, key=lambda s: s.frame)
        self._by_frame = {s.frame: s for s in self.samples}

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def get(self, frame: int) -> BallSample | None:
        return self._by_frame.get(frame)

    @property
    def frames(self) -> np.ndarray:
        return np.array([s.frame for s in self.samples])

    @property
    def image_xy(self) -> np.ndarray:
        return np.array([s.image for s in self.samples])

    @property
    def court_xy(self) -> np.ndarray:
        return np.array([s.court for s in self.samples])

    def detection_rate(self, total_frames: int) -> float:
        real = sum(1 for s in self.samples if not s.interpolated)
        return real / total_frames if total_frames else 0.0

    def segments(self, max_gap: int) -> list[list[BallSample]]:
        """Split into runs of samples with no gap longer than ``max_gap``.

        Each returned run is a stretch of continuous ball flight - a rally, or
        part of one. The gaps between them are where the ball was out of play
        or lost.
        """
        if not self.samples:
            return []
        runs: list[list[BallSample]] = [[self.samples[0]]]
        for previous, current in zip(self.samples, self.samples[1:]):
            if current.frame - previous.frame > max_gap:
                runs.append([current])
            else:
                runs[-1].append(current)
        return runs


def fill_gaps(samples: list[BallSample], max_gap: int = 8) -> Trajectory:
    """Linearly interpolate gaps of up to ``max_gap`` frames.

    8 frames is about a quarter-second at 30 fps - long enough to bridge the
    motion blur of a fast shot, short enough that a real absence survives as a
    gap. Interpolated samples are marked, so confidence can be discounted for
    any event that depends on them.
    """
    if len(samples) < 2:
        return Trajectory(list(samples))

    ordered = sorted(samples, key=lambda s: s.frame)
    filled: list[BallSample] = []
    for previous, current in zip(ordered, ordered[1:]):
        filled.append(previous)
        gap = current.frame - previous.frame
        if 1 < gap <= max_gap:
            for step in range(1, gap):
                t = step / gap
                filled.append(
                    BallSample(
                        frame=previous.frame + step,
                        image=previous.image + t * (current.image - previous.image),
                        court=previous.court + t * (current.court - previous.court),
                        confidence=min(previous.confidence, current.confidence) * 0.7,
                        interpolated=True,
                    )
                )
    filled.append(ordered[-1])
    return Trajectory(filled)


def smooth(trajectory: Trajectory, window: int = 5) -> Trajectory:
    """Centred moving average over each continuous run.

    Smoothing is deliberately applied per-run, never across a gap: averaging
    the last frame before a gap with the first frame after it would blend two
    unrelated flights and put a phantom turning point between them.

    A centred window matters too. The baseline used a trailing rolling mean,
    which shifts every detected event later by half the window - a systematic
    timing bias that would misplace bounces by ~2 frames at 30 fps.
    """
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd number >= 3")

    half = window // 2
    out: list[BallSample] = []
    for run in trajectory.segments(max_gap=1):
        image = np.array([s.image for s in run])
        court = np.array([s.court for s in run])
        for index, sample in enumerate(run):
            low = max(0, index - half)
            high = min(len(run), index + half + 1)
            out.append(
                BallSample(
                    frame=sample.frame,
                    image=image[low:high].mean(axis=0),
                    court=court[low:high].mean(axis=0),
                    confidence=sample.confidence,
                    interpolated=sample.interpolated,
                )
            )
    return Trajectory(out)


def from_detections(
    records: list[dict], max_gap: int = 8, smooth_window: int = 5
) -> Trajectory:
    """Build a smoothed, gap-filled trajectory from raw per-frame records.

    Each record needs ``frame``, ``image`` (2,), ``court`` (2,) and
    ``confidence``.
    """
    samples = [
        BallSample(
            frame=int(r["frame"]),
            image=np.asarray(r["image"], dtype=float),
            court=np.asarray(r["court"], dtype=float),
            confidence=float(r.get("confidence", 1.0)),
        )
        for r in records
    ]
    return smooth(fill_gaps(samples, max_gap=max_gap), window=smooth_window)
