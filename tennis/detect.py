"""Model wrappers: players, ball, court keypoints.

Thin layers over the baseline's three models, with its sharpest edges removed:

- Detections are never silently loaded from someone else's cached run. The
  baseline hardcoded ``read_from_stub=True``, so it replayed pickles recorded
  from the author's own video no matter what you fed it - the single biggest
  reason people get nonsense output from that repo. Here the cache is opt-in
  and keyed to the input file.
- The ball detector keeps the highest-confidence box per frame. The baseline
  looped over candidates and kept whichever happened to come last.
- Court keypoints are re-estimated periodically instead of being read once from
  frame 0 and trusted for the rest of the video.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import models, transforms
from ultralytics import YOLO


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]
    confidence: float
    track_id: int | None = None

    @property
    def centre(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    @property
    def feet(self) -> np.ndarray:
        """Bottom-centre of the box - where the player meets the court plane.

        The homography maps the court *plane*, so a player's position is only
        meaningful at ground level. Using the box centre would place everyone
        roughly a metre behind where they stand.
        """
        x1, _, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2, y2])


class PlayerDetector:
    def __init__(self, model_path: str = "yolov8x.pt", conf: float = 0.5,
                 device: str = "cpu") -> None:
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.track(
            frame, persist=True, classes=[0], conf=self.conf,
            device=self.device, verbose=False
        )[0]
        out: list[Detection] = []
        for box in result.boxes:
            out.append(
                Detection(
                    bbox=tuple(box.xyxy.tolist()[0]),
                    confidence=float(box.conf.item()),
                    track_id=int(box.id.item()) if box.id is not None else None,
                )
            )
        return out


class BallDetector:
    # A tennis ball is ~15 px across in a 1080p frame. YOLO's default imgsz of
    # 640 letterboxes that frame down by a factor of three, leaving a ball of
    # ~5 px - close to the smallest thing the network can represent at all.
    # Measured on a 451-frame clip, raising imgsz to 960 took detection from
    # 47.0% to 95.6% of frames, with no retraining. It is the single largest
    # accuracy change in the project, and it is a one-line default.
    #
    #   imgsz  detection  mean conf  speed
    #     640      47.0%      0.329   32 fps
    #     960      95.6%      0.366   20 fps   <- chosen
    #    1280      78.3%      0.394   12 fps
    #    1920      96.2%      0.337    6 fps
    #
    # 1920 is no better than 960 and three times slower. 1280 scoring worse
    # than both is a letterboxing artefact of this model's stride, not noise -
    # it reproduces across runs, which is why the default is set empirically
    # rather than by assuming bigger is better.
    def __init__(self, model_path: str, conf: float = 0.15,
                 device: str = "cpu", imgsz: int = 960) -> None:
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device
        self.imgsz = imgsz

    def candidates(self, frame: np.ndarray, top_k: int = 5) -> list[Detection]:
        """Every plausible ball in this frame, best-scoring first.

        Which of these is the ball is not decidable from one frame: a bright
        line marking often outscores a motion-blurred ball. The choice is made
        once the whole flight is known, in tennis.balltrack.
        """
        result = self.model.predict(
            frame, conf=self.conf, imgsz=self.imgsz, device=self.device,
            verbose=False
        )[0]
        ranked = sorted(
            result.boxes, key=lambda b: float(b.conf.item()), reverse=True
        )[:top_k]
        return [
            Detection(
                bbox=tuple(b.xyxy.tolist()[0]), confidence=float(b.conf.item())
            )
            for b in ranked
        ]

    def detect(self, frame: np.ndarray) -> Detection | None:
        # Highest confidence, not last-in-loop. A tennis ball is one object;
        # extra boxes are line markings and shoes. Kept for callers that want a
        # single-frame answer; the pipeline uses candidates() instead.
        found = self.candidates(frame, top_k=1)
        return found[0] if found else None


class CourtDetector:
    """ResNet50 regressing 14 court keypoints (28 values)."""

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self.device = torch.device(device)
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 14 * 2)
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        self.model = model.to(self.device).eval()
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @torch.no_grad()
    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Return a (14, 2) array of keypoints in this frame's pixel space."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        raw = self.model(tensor).squeeze().cpu().numpy()
        height, width = frame.shape[:2]
        raw[0::2] *= width / 224.0
        raw[1::2] *= height / 224.0
        return raw.reshape(-1, 2)


def cache_key(video_path: str | Path, tag: str) -> Path:
    """A cache path tied to the video's identity, not just its name.

    Hashing size and mtime alongside the path means renaming a file, or editing
    it in place, produces a different key - so a cached run can never be
    replayed against different footage.
    """
    path = Path(video_path)
    stat = path.stat()
    digest = hashlib.sha256(
        f"{path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}".encode()
    ).hexdigest()[:16]
    return Path("tracker_stubs") / f"{path.stem}.{tag}.{digest}.pkl"


def load_cache(key: Path):
    if key.exists():
        with key.open("rb") as handle:
            return pickle.load(handle)
    return None


def save_cache(key: Path, payload) -> None:
    key.parent.mkdir(parents=True, exist_ok=True)
    with key.open("wb") as handle:
        pickle.dump(payload, handle)
