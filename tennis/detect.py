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
    def __init__(self, model_path: str = "yolov8x.pt", conf: float = 0.5) -> None:
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.track(
            frame, persist=True, classes=[0], conf=self.conf, verbose=False
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
    def __init__(self, model_path: str, conf: float = 0.15) -> None:
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame: np.ndarray) -> Detection | None:
        result = self.model.predict(frame, conf=self.conf, verbose=False)[0]
        if not len(result.boxes):
            return None
        # Highest confidence, not last-in-loop. A tennis ball is one object;
        # extra boxes are line markings and shoes.
        best = max(result.boxes, key=lambda b: float(b.conf.item()))
        return Detection(
            bbox=tuple(best.xyxy.tolist()[0]), confidence=float(best.conf.item())
        )


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
