"""TrackNet: ball detection that can see motion.

Why a second ball detector
--------------------------
YOLO - and RT-DETR, and every other box detector - looks at one frame at a
time. A tennis ball on a broadcast wide shot is about ten pixels across and is
smeared into a streak whenever it is travelling fast, which is precisely when
it matters. From a single frame it is often genuinely ambiguous: a bright
segment of court line looks more like a ball than the ball does.

From three frames it is not ambiguous at all. The ball is the thing that moved.

TrackNet takes three consecutive frames stacked into one nine-channel input and
predicts a heatmap rather than a box, so motion is available to the network
itself instead of being reconstructed afterwards. ``tennis/balltrack.py`` is a
hand-built version of the same idea - it recovers the ball's identity from
continuity *after* detection, using a Viterbi pass over candidates. That works,
and it lifted detection from 69.3% to 89.4% on the Wimbledon clip, but it can
only choose among boxes YOLO already proposed. It cannot recover a ball that
was never proposed at all.

Architecture
------------
Deliberately a faithful reimplementation of the ``BallTrackerNet`` used by
`yastrebksv/TrackNet <https://github.com/yastrebksv/TrackNet>`_, whose
pretrained tennis weights this project intends to load directly. The layer
names and ordering matter for that reason and should not be tidied: the
checkpoint is a flat ``state_dict`` keyed by these attribute names, so renaming
``conv7`` breaks loading with no error message worth reading.

It is a VGG-style encoder (64, 64, pool, 128, 128, pool, 256x3, pool, 512x3)
mirrored by an upsampling decoder, ending in a 256-way channel dimension. That
last part is unusual and worth stating plainly: the network does not regress a
coordinate. It classifies each output pixel into one of 256 intensity levels,
so the "heatmap" is the argmax over that dimension. Training it as
classification rather than regression is what keeps a small bright object from
being averaged away by a loss that prefers the mean.

Input is 360x640, which is *not* the source resolution - detections come back
in that space and must be scaled to the original frame. ``BallDetector`` in
tennis/detect.py has no such step, so the two are not interchangeable without
the scaling done here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Input geometry the pretrained weights expect. Changing either means the
# checkpoint no longer applies.
INPUT_HEIGHT = 360
INPUT_WIDTH = 640

# Frames stacked per forward pass. Three is the architecture, not a tunable:
# the first convolution takes nine channels.
FRAME_STACK = 3

# Intensity levels the output layer classifies each pixel into.
OUTPUT_LEVELS = 256

# Below this peak intensity the frame is reported as having no ball rather than
# a faint guess. The heatmap peaks near the maximum when the ball is genuinely
# found, so a low peak means the network saw nothing it liked.
MIN_PEAK = 128


@dataclass
class BallPoint:
    """One frame's ball position, in the *source* frame's pixels."""

    frame: int
    xy: np.ndarray
    confidence: float


def _conv_block(in_channels: int, out_channels: int):
    """Conv -> ReLU -> BatchNorm, in that order.

    The ordering is unusual - normalisation almost always precedes the
    activation - but it is what the published weights were trained with, and
    swapping it silently degrades them.
    """
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
    )


def build_model(out_channels: int = OUTPUT_LEVELS):
    """The BallTrackerNet graph, with layer names matching the checkpoint."""
    import torch
    import torch.nn as nn

    class BallTrackerNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.out_channels = out_channels

            self.conv1 = _conv_block(FRAME_STACK * 3, 64)
            self.conv2 = _conv_block(64, 64)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv3 = _conv_block(64, 128)
            self.conv4 = _conv_block(128, 128)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.conv5 = _conv_block(128, 256)
            self.conv6 = _conv_block(256, 256)
            self.conv7 = _conv_block(256, 256)
            self.pool3 = nn.MaxPool2d(2, 2)
            self.conv8 = _conv_block(256, 512)
            self.conv9 = _conv_block(512, 512)
            self.conv10 = _conv_block(512, 512)

            self.ups1 = nn.Upsample(scale_factor=2)
            self.conv11 = _conv_block(512, 256)
            self.conv12 = _conv_block(256, 256)
            self.conv13 = _conv_block(256, 256)
            self.ups2 = nn.Upsample(scale_factor=2)
            self.conv14 = _conv_block(256, 128)
            self.conv15 = _conv_block(128, 128)
            self.ups3 = nn.Upsample(scale_factor=2)
            self.conv16 = _conv_block(128, 64)
            self.conv17 = _conv_block(64, 64)
            self.conv18 = _conv_block(64, out_channels)

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.pool1(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.pool2(x)
            x = self.conv5(x)
            x = self.conv6(x)
            x = self.conv7(x)
            x = self.pool3(x)
            x = self.conv8(x)
            x = self.conv9(x)
            x = self.conv10(x)
            x = self.ups1(x)
            x = self.conv11(x)
            x = self.conv12(x)
            x = self.conv13(x)
            x = self.ups2(x)
            x = self.conv14(x)
            x = self.conv15(x)
            x = self.ups3(x)
            x = self.conv16(x)
            x = self.conv17(x)
            x = self.conv18(x)
            # Flatten the spatial dimensions so the 256 channels become a
            # per-pixel class distribution, which is how the loss was defined.
            return x.reshape(x.size(0), self.out_channels, -1)

    return BallTrackerNet()


def _peak(heatmap: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Brightest point of one heatmap, in model-input pixels."""
    peak_value = float(heatmap.max())
    if peak_value < MIN_PEAK:
        return None, peak_value
    flat = int(np.argmax(heatmap))
    y, x = divmod(flat, INPUT_WIDTH)
    return np.array([float(x), float(y)]), peak_value


def _peak_hough(heatmap, previous, max_step_px=80.0):
    """Ball position from the heatmap by blob shape plus continuity.

    ``_peak`` takes the single brightest pixel, which is fragile: any one hot
    pixel on a line marking or a shoe wins outright. The heatmap TrackNet emits
    is a *blob*, so finding circles in it and taking the one nearest the
    previous frame's ball uses both the shape and the fact that a ball cannot
    teleport. This is the postprocessing yastrebksv's implementation uses, and
    it is the main thing our version was missing.

    ``previous`` is last frame's point in model-input pixels, or None.
    """
    import cv2

    field = np.clip(heatmap, 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(field, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        binary, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2,
        minRadius=2, maxRadius=7,
    )
    if circles is None:
        return None
    found = circles[0]
    if previous is not None:
        # Nearest to where the ball just was, provided it is reachable.
        best, best_d = None, max_step_px
        for cx, cy, _r in found:
            d = float(np.hypot(cx - previous[0], cy - previous[1]))
            if d < best_d:
                best, best_d = (float(cx), float(cy)), d
        if best is not None:
            return np.array(best)
        return None
    return np.array([float(found[0][0]), float(found[0][1])])


def adapt_state_dict(checkpoint: dict, target: dict) -> dict:
    """Reconcile a published checkpoint's key names with this graph's.

    The upstream implementation wraps each conv/ReLU/norm triple in a small
    ``ConvBlock`` module holding a ``self.block`` Sequential, so its keys read
    ``conv1.block.0.weight``. Expressing the same triple as a bare Sequential
    here gives ``conv1.0.weight``. The tensors are identical and in the same
    order; only the path differs.

    Rather than hard-code one convention and fail obscurely against the other -
    ``load_state_dict`` reports a wall of missing and unexpected keys that says
    nothing about the cause - match by position within each layer. If the
    shapes line up in order, the mapping is unambiguous.
    """
    if set(checkpoint) == set(target):
        return checkpoint

    def normalise(key: str) -> str:
        return key.replace(".block.", ".")

    remapped = {normalise(key): value for key, value in checkpoint.items()}
    if set(remapped) == set(target):
        return remapped

    # Last resort: same count and same shapes, in order. Anything else is a
    # different architecture and should fail loudly rather than half-load.
    if len(checkpoint) != len(target):
        raise ValueError(
            f"checkpoint has {len(checkpoint)} tensors, this graph expects "
            f"{len(target)} - it is not a BallTrackerNet checkpoint"
        )
    out = {}
    for (target_key, target_value), source_value in zip(
        target.items(), checkpoint.values()
    ):
        if tuple(target_value.shape) != tuple(source_value.shape):
            raise ValueError(
                f"shape mismatch at {target_key}: checkpoint has "
                f"{tuple(source_value.shape)}, graph expects "
                f"{tuple(target_value.shape)}"
            )
        out[target_key] = source_value
    return out


class TrackNetDetector:
    """Runs TrackNet over a video's frames and returns ball positions.

    Consumes frames in triples, so the first two frames of a clip cannot be
    predicted and are reported as misses rather than guesses. That is honest
    but it is also why this must not be benchmarked on very short clips: two
    missing frames out of thirty is a 7% detection penalty that says nothing
    about the model.
    """

    def __init__(self, weights: str, device: str = "cpu") -> None:
        import torch

        path = Path(weights)
        if not path.exists():
            raise FileNotFoundError(
                f"TrackNet weights not found at {path}. Download the pretrained "
                "tennis weights from https://github.com/yastrebksv/TrackNet"
            )
        self.device = device
        self.model = build_model()
        state = torch.load(str(path), map_location=device)
        # Checkpoints in the wild are sometimes wrapped in a training dict.
        if isinstance(state, dict):
            for wrapper in ("model_state", "state_dict", "model"):
                if wrapper in state and isinstance(state[wrapper], dict):
                    state = state[wrapper]
                    break
        state = adapt_state_dict(state, self.model.state_dict())
        self.model.load_state_dict(state)
        self.model = self.model.to(device).eval()

    def _prepare(self, frames: list[np.ndarray]):
        """Three BGR frames -> one normalised 9-channel tensor."""
        import cv2
        import torch

        resized = [
            cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT)) for frame in frames
        ]
        # Newest frame first: the checkpoint was trained with the current frame
        # leading and the two previous ones as context.
        stacked = np.concatenate(resized[::-1], axis=2).astype(np.float32) / 255.0
        tensor = torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def detect_sequence(
        self, frames: list[tuple[int, np.ndarray]]
    ) -> list[BallPoint]:
        """Ball positions for a list of ``(frame_index, image)`` pairs."""
        import torch

        if len(frames) < FRAME_STACK:
            return []

        source_height, source_width = frames[0][1].shape[:2]
        scale_x = source_width / INPUT_WIDTH
        scale_y = source_height / INPUT_HEIGHT

        out: list[BallPoint] = []
        previous_point = None
        with torch.no_grad():
            for i in range(FRAME_STACK - 1, len(frames)):
                window = [frames[i - 2][1], frames[i - 1][1], frames[i][1]]
                logits = self.model(self._prepare(window))
                # argmax over the intensity dimension gives the heatmap itself.
                heatmap = (
                    logits.argmax(dim=1)
                    .reshape(INPUT_HEIGHT, INPUT_WIDTH)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                peak = float(heatmap.max())
                point = _peak_hough(heatmap, previous_point)
                if point is None:
                    # Fall back to the brightest pixel so a frame Hough cannot
                    # resolve still yields a position rather than a gap.
                    point, peak = _peak(heatmap)
                if point is None:
                    previous_point = None
                    continue
                previous_point = point
                out.append(
                    BallPoint(
                        frame=frames[i][0],
                        # Back to source resolution - the rest of the pipeline
                        # measures in the original frame's pixels.
                        xy=np.array([point[0] * scale_x, point[1] * scale_y]),
                        confidence=round(min(peak / (OUTPUT_LEVELS - 1), 1.0), 3),
                    )
                )
        return out


def as_candidates(points: list[BallPoint]) -> list[dict]:
    """Shape TrackNet output like the candidate lists balltrack.resolve wants.

    TrackNet emits one position per frame, so each frame contributes a single
    candidate. Passing it through the same Viterbi pass as the YOLO path is not
    redundant: the missing-ball state still lets an isolated bad frame be
    dropped rather than dragged into the trajectory, and it keeps both
    detectors feeding one downstream contract.
    """
    return [
        {
            "frame": point.frame,
            "boxes": [
                {"conf": point.confidence, "xy": [float(point.xy[0]), float(point.xy[1])]}
            ],
        }
        for point in points
    ]
