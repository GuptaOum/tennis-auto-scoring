"""Train the court keypoint regressor: ResNet50 -> 14 (x, y) points.

A corrected, evaluated rewrite of the baseline's
`tennis_court_keypoints_training.ipynb`, which does not execute as published.
Its bugs, and what each one cost:

    items['kps']            NameError on the first batch. The notebook cannot
                            have been run in the state it was committed.
    model.stat_dict()       AttributeError at the final line, after all the
                            training time had been spent. Nothing was saved.
    devic = torch.device()  the typo'd name is never used, so `model.to(device)`
                            would raise NameError.
    val_loader              constructed, then never iterated. No validation
                            loss, no metric, no early stopping, no way to know
                            whether the model had overfitted 20 epochs ago.

The last one is the substantive failure. A regression model trained for a
fixed epoch count with no validation signal is not a trained model, it is a
model that stopped. This version measures PCK every epoch and keeps the best
weights by that metric.

PCK (Percentage of Correct Keypoints) is the standard measure here: the share
of predicted points landing within a threshold distance of the truth, the
threshold expressed as a fraction of image size so it is resolution
independent. PCK@0.05 on a 224 px input means "within ~11 px". Raw MSE is
useless for comparing across resolutions and gives no intuition about whether
a court would actually be usable.

Usage:

    python training/train_keypoints.py --data-dir path/to/data
    python training/train_keypoints.py --data-dir ... --epochs 60 --device cuda

Expected layout, matching the upstream dataset:

    data/
      images/          <id>.png
      data_train.json  [{"id": ..., "kps": [[x, y], ... 14 pairs]}, ...]
      data_val.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

NUM_KEYPOINTS = 14
INPUT_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class KeypointsDataset(Dataset):
    """Court images with 14 labelled keypoints, resized to the network input.

    Keypoints are scaled into the resized frame, which the baseline did
    correctly - but note the direction. Labels are scaled *into* 224x224 space
    here, and predictions are scaled back *out* at inference. Getting those two
    out of step is the classic way to produce a model that looks trained and
    predicts nonsense.
    """

    def __init__(self, image_dir: str | Path, annotation_file: str | Path) -> None:
        self.image_dir = Path(image_dir)
        with open(annotation_file, encoding="utf-8") as handle:
            self.items = json.load(handle)

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.items[index]          # baseline wrote `items[...]` here
        path = self.image_dir / f"{item['id']}.png"
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"could not read {path}")

        height, width = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image)

        keypoints = np.array(item["kps"], dtype=np.float32).flatten()
        keypoints[0::2] *= INPUT_SIZE / width
        keypoints[1::2] *= INPUT_SIZE / height
        return tensor, torch.from_numpy(keypoints)


def build_model(device: torch.device) -> torch.nn.Module:
    """ResNet50 with its classifier swapped for a 28-value regression head.

    `weights=` rather than the deprecated `pretrained=True`. ImageNet features
    transfer well here: court lines are edges and corners, which is exactly
    what the early layers already detect.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_KEYPOINTS * 2)
    return model.to(device)


def pck(
    predicted: torch.Tensor, target: torch.Tensor, threshold: float = 0.05
) -> float:
    """Percentage of Correct Keypoints within ``threshold`` * image size.

    Both tensors are (batch, 28) in 224x224 space.
    """
    pred = predicted.reshape(-1, NUM_KEYPOINTS, 2)
    true = target.reshape(-1, NUM_KEYPOINTS, 2)
    distances = torch.linalg.norm(pred - true, dim=2)
    return float((distances < threshold * INPUT_SIZE).float().mean())


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """The validation pass the baseline built a loader for and never ran."""
    model.eval()
    total_loss = 0.0
    all_pred, all_true = [], []

    for images, keypoints in loader:
        images, keypoints = images.to(device), keypoints.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, keypoints).item() * images.size(0)
        all_pred.append(outputs.cpu())
        all_true.append(keypoints.cpu())

    predicted = torch.cat(all_pred)
    target = torch.cat(all_true)
    pixel_error = torch.linalg.norm(
        predicted.reshape(-1, NUM_KEYPOINTS, 2)
        - target.reshape(-1, NUM_KEYPOINTS, 2),
        dim=2,
    ).mean()

    return {
        "loss": total_loss / len(loader.dataset),
        "pck_005": pck(predicted, target, 0.05),
        "pck_010": pck(predicted, target, 0.10),
        "mean_pixel_error": float(pixel_error),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the court keypoint model")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="models/keypoints_finetuned.pth")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)     # baseline typo'd this as `devic`
    data_dir = Path(args.data_dir)

    train_set = KeypointsDataset(data_dir / "images", data_dir / "data_train.json")
    val_set = KeypointsDataset(data_dir / "images", data_dir / "data_val.json")
    print(f"train {len(train_set)} images | val {len(val_set)} images")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    model = build_model(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Cosine decay rather than a flat rate: keypoint regression benefits from
    # small steps at the end, where the remaining error is a few pixels.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_pck = -1.0
    history = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, keypoints in train_loader:
            images, keypoints = images.to(device), keypoints.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), keypoints)
            loss.backward()
            optimizer.step()
            running += loss.detach().item() * images.size(0)

        scheduler.step()
        train_loss = running / len(train_set)
        metrics = evaluate(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})

        marker = ""
        if metrics["pck_005"] > best_pck:
            best_pck = metrics["pck_005"]
            # state_dict(), not the baseline's stat_dict(), and saved on every
            # improvement rather than once at the very end - so a crash at
            # epoch 55 does not throw away the whole run.
            torch.save(model.state_dict(), out_path)
            marker = "  <- best, saved"

        print(
            f"epoch {epoch:3d}/{args.epochs}  "
            f"train {train_loss:8.2f}  val {metrics['loss']:8.2f}  "
            f"PCK@0.05 {metrics['pck_005']:.3f}  PCK@0.10 {metrics['pck_010']:.3f}  "
            f"px err {metrics['mean_pixel_error']:5.2f}{marker}",
            flush=True,
        )

    elapsed = time.time() - started
    report = {
        "epochs": args.epochs,
        "minutes": round(elapsed / 60, 1),
        "best_pck_005": round(best_pck, 4),
        "final": history[-1],
        "history": history,
        "weights": str(out_path),
    }
    Path(out_path).with_suffix(".training.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"\nbest PCK@0.05 {best_pck:.4f} in {elapsed / 60:.1f} min")
    print(f"weights: {out_path}")


if __name__ == "__main__":
    main()
