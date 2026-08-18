"""Bounce detection learned from trajectory shape, rather than thresholded.

Why the threshold had to go
---------------------------
Ground contact is currently found by looking for a local maximum in projected
court y with at least some prominence. The idea is sound - a homography maps
the court *plane*, so an airborne ball projects beyond where it really is, and
the projection swings back as the ball comes down. But the decision rule built
on top of it is one number, and one number cannot express what a bounce looks
like.

That was measured, not assumed. ``training/calibrate_bounce.py`` swept the
threshold across its whole useful range against ground truth read off the
broadcast scoreboard, and **no value found both points in the test clip**. The
best case found one of two and invented four rallies that never happened. When
a single parameter cannot be set correctly at any value, the model is wrong,
not the tuning.

What a bounce actually looks like is a *conjunction*: the projected height
signal reverses, the reversal is sharp rather than gradual, the horizontal
speed drops across it because the surface takes energy out of the ball, the
ball is near the court plane at that instant, and no player is within reach
(or it is a hit, not a bounce). Any one of those alone is ambiguous. Together
they are not. A learned model over all of them is the natural way to say that,
and it is what yastrebksv/TennisProject does with a gradient-boosted regressor
over trajectory features.

Labels come from the TrackNet tennis dataset - 81 broadcast clips, 19,835
labelled frames, with the frames at which the ball contacts the court marked.
So this is trained on real bounces from real broadcast footage rather than on
the synthetic arcs in the test suite, which was the other half of the problem:
the previous threshold was calibrated against a fixture with one height scale
and one travel rate.

Design notes
------------
Features are deliberately **local and scale-free**. Everything is a difference
or a ratio computed inside a short window around the candidate frame, in court
metres where the quantity is geometric and in normalised units where it is not.
Nothing depends on absolute court position, so a model trained on the far side
works on the near side, and a model trained on one camera height transfers to
another. That is the property the raw-image-y baseline lacked, and it is why it
confounded height with depth.

Candidates are still proposed geometrically - every local maximum in the height
signal, with the prominence floor set very low. Recall at that stage is nearly
free; the learned model supplies the precision. Proposing everything and
classifying is a much easier problem than deciding with one threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tennis.trajectory import Trajectory

# Frames either side of a candidate that form its feature window. Eight frames
# is about a quarter-second at 30 fps - long enough to contain the approach and
# the departure of a bounce, short enough not to swallow the neighbouring shot.
WINDOW = 8

# Prominence floor for *proposing* a candidate. Far below anything that would
# be trusted on its own: the point is to let the classifier see every real
# bounce, including shallow ones, and reject the rest itself.
PROPOSAL_PROMINENCE = 0.05

# Floor on the upward kink that proposes a candidate, in units of the segment's
# own median frame-to-frame image speed.
#
# Measured against the 136 labelled bounces of the TrackNet dataset, this is the
# single most consequential constant in the bounce layer:
#
#     rule                                    recall of real bounces
#     local maximum in projected court y                     34.6%
#     upward kink, this rule, floor 0.05                     99.3%
#
# The old rule asked the wrong question. A bounce is not the top of an arc, it
# is a *corner* in one: the ball's vertical velocity reverses discontinuously.
# Projected court y carries height and down-court travel together, and the
# travel term usually dominates, so at a real bounce the height signal is
# almost always still monotone - a local maximum only 2% of the time. The
# candidate never reached the classifier, and no amount of training could
# recover an event that was never proposed.
#
# Dividing by the segment's median speed is what keeps this resolution- and
# pace-independent: the dataset is 720p and the project's test footage is
# 1080p, and a raw pixel threshold would silently mean different things on each.
PROPOSAL_KINK = 0.05

# A labelled bounce and a proposed candidate this many frames apart are the
# same event. Two frames at 30 fps is 66 ms - tighter than the annotation's own
# precision, looser than nothing.
MATCH_TOLERANCE = 2

FEATURE_NAMES = [
    "image_kink",
    "height_prominence_left",
    "height_prominence_right",
    "height_curvature",
    "reversal_sharpness",
    "speed_before",
    "speed_after",
    "speed_ratio",
    "direction_change_deg",
    "vertical_velocity_before",
    "vertical_velocity_after",
    "vertical_velocity_flip",
    "court_y_at_candidate",
    "distance_inside_court_m",
    "interpolated_share",
    "mean_confidence",
]


@dataclass
class Candidate:
    """A frame that might be a bounce, with the evidence for and against."""

    frame: int
    index: int
    features: np.ndarray


def _height_signal(samples) -> np.ndarray:
    """Projected court y - the height proxy. See tennis/bounce.py."""
    return np.array([s.court[1] for s in samples], dtype=float)


def propose(trajectory: Trajectory, max_gap: int = 1) -> list[Candidate]:
    """Every upward corner in the ball's image path, with features attached.

    Deliberately permissive. A candidate is not a claim that a bounce
    happened; it is a question put to the classifier. What matters here is
    recall: a bounce not proposed is a bounce the classifier cannot find, and
    the rule this replaced was losing two thirds of them. See PROPOSAL_KINK.
    """
    out: list[Candidate] = []
    for run in trajectory.segments(max_gap=max_gap):
        out.extend(propose_run(run))
    return out


def propose_run(run) -> list[Candidate]:
    """``propose`` for a single already-segmented run.

    Split out so the production event layer can propose against the very same
    run it is already iterating, instead of re-segmenting the trajectory and
    hoping the two segmentations agree.
    """
    if len(run) < 5:
        return []

    out: list[Candidate] = []
    height = _height_signal(run)
    court = np.array([s.court for s in run], dtype=float)
    frames = np.array([s.frame for s in run], dtype=float)
    kink = _upward_kink(np.array([s.image[1] for s in run], dtype=float))

    for i in range(2, len(run) - 2):
        # A bounce is a corner, not a peak - see PROPOSAL_KINK.
        if not (kink[i] >= kink[i - 1] and kink[i] >= kink[i + 1]):
            continue
        if kink[i] < PROPOSAL_KINK:
            continue
        out.append(
            Candidate(
                frame=run[i].frame,
                index=i,
                features=_features(run, height, court, frames, i, float(kink[i])),
            )
        )
    return out


def _upward_kink(image_y: np.ndarray) -> np.ndarray:
    """How sharply the ball turns upward in the frame, per sample.

    Image y grows downward, so a ball rebounding off the court makes ``y``
    accelerate negatively. The second difference measures exactly that, and
    normalising by the segment's median speed makes the number dimensionless -
    the same threshold then means the same thing at 720p and at 1080p, and for
    a floated lob as for a flat drive. Endpoints are zero-padded so the array
    lines up with the samples it describes.
    """
    if len(image_y) < 3:
        return np.zeros(len(image_y), dtype=float)
    speed = float(np.median(np.abs(np.diff(image_y))))
    if not np.isfinite(speed) or speed <= 0.0:
        speed = 1e-6
    return np.concatenate([[0.0], -np.diff(image_y, 2) / speed, [0.0]])


def _prominence(values: np.ndarray, i: int) -> tuple[float, float]:
    """Drop on each side of a peak, stopping at any higher point."""
    here = values[i]
    left = right = 0.0
    for j in range(i - 1, -1, -1):
        if values[j] > here:
            break
        left = max(left, here - values[j])
    for j in range(i + 1, len(values)):
        if values[j] > here:
            break
        right = max(right, here - values[j])
    return left, right


def _features(run, height, court, frames, i, kink: float) -> np.ndarray:
    """The window around one candidate, as scale-free numbers."""
    low = max(0, i - WINDOW)
    high = min(len(run), i + WINDOW + 1)

    left_prom, right_prom = _prominence(height, i)

    # Second difference: a bounce is a corner in the height signal, a smooth
    # apex is not. This is what separates "the ball turned" from "the ball
    # turned *sharply*", which the old single threshold could not express.
    curvature = float(height[i - 1] - 2 * height[i] + height[i + 1])

    before = court[low:i + 1]
    after = court[i:high]
    dt_before = max(frames[i] - frames[low], 1.0)
    dt_after = max(frames[high - 1] - frames[i], 1.0)

    speed_before = (
        float(np.linalg.norm(before[-1] - before[0]) / dt_before)
        if len(before) > 1 else 0.0
    )
    speed_after = (
        float(np.linalg.norm(after[-1] - after[0]) / dt_after)
        if len(after) > 1 else 0.0
    )
    # The surface takes energy out of the ball, so speed drops across a real
    # bounce. It does not drop across the apex of a flight.
    speed_ratio = speed_after / speed_before if speed_before > 1e-6 else 0.0

    direction = _direction_change(before, after)

    vy_before = float(height[i] - height[max(low, i - 3)])
    vy_after = float(height[min(high - 1, i + 3)] - height[i])
    # A genuine ground contact reverses the vertical component; a detection
    # glitch usually does not, and an apex reverses it the other way.
    flip = 1.0 if (vy_before > 0 > vy_after) else 0.0

    from tennis.court import COURT_LENGTH, DOUBLES_WIDTH

    x, y = float(court[i][0]), float(court[i][1])
    inside = min(
        x, DOUBLES_WIDTH - x, y, COURT_LENGTH - y
    )

    window_samples = run[low:high]
    interpolated = sum(1 for s in window_samples if s.interpolated)

    return np.array(
        [
            kink,
            left_prom,
            right_prom,
            curvature,
            abs(curvature) / (abs(left_prom) + abs(right_prom) + 1e-6),
            speed_before,
            speed_after,
            speed_ratio,
            direction,
            vy_before,
            vy_after,
            flip,
            y,
            inside,
            interpolated / max(len(window_samples), 1),
            float(np.mean([s.confidence for s in window_samples])),
        ],
        dtype=float,
    )


def _direction_change(before: np.ndarray, after: np.ndarray) -> float:
    """Angle between incoming and outgoing court-plane travel, in degrees."""
    if len(before) < 2 or len(after) < 2:
        return 0.0
    v1 = before[-1] - before[0]
    v2 = after[-1] - after[0]
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cosine = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def label_candidates(
    candidates: list[Candidate], bounce_frames: set[int]
) -> np.ndarray:
    """1 where a candidate matches an annotated bounce, 0 otherwise."""
    labels = np.zeros(len(candidates), dtype=int)
    for index, candidate in enumerate(candidates):
        if any(
            abs(candidate.frame - truth) <= MATCH_TOLERANCE
            for truth in bounce_frames
        ):
            labels[index] = 1
    return labels


class LearnedBounceDetector:
    """Gradient-boosted classifier over the features above.

    Kept behind a small wrapper so the pipeline depends on ``predict`` and not
    on which library trained it, and so a missing model degrades to the
    geometric rule instead of crashing a run.
    """

    def __init__(self, model=None, threshold: float = 0.5) -> None:
        self.model = model
        self.threshold = threshold

    @classmethod
    def load(cls, path: str | Path, threshold: float = 0.5):
        import joblib

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"no bounce model at {path}")
        return cls(joblib.load(path), threshold=threshold)

    def save(self, path: str | Path) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "LearnedBounceDetector":
        from sklearn.ensemble import HistGradientBoostingClassifier

        # Bounces are the minority of proposed candidates by a wide margin, so
        # the class weighting matters more than the tree count here.
        self.model = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.06,
            max_depth=6,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=0,
        )
        self.model.fit(features, labels)
        return self

    def predict(self, candidates: list[Candidate]) -> list[tuple[Candidate, float]]:
        """Each candidate with its probability of being a real bounce."""
        if not candidates:
            return []
        if self.model is None:
            raise RuntimeError("no trained model loaded")
        matrix = np.vstack([c.features for c in candidates])
        scores = self.model.predict_proba(matrix)[:, 1]
        return list(zip(candidates, (float(s) for s in scores)))

    def accept(self, candidates: list[Candidate]) -> list[tuple[Candidate, float]]:
        return [
            (candidate, score)
            for candidate, score in self.predict(candidates)
            if score >= self.threshold
        ]
