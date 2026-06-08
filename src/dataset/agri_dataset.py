"""
agri_dataset.py
---------------
PyTorch Dataset class for IndAgri-VLM.
Reads train/val/test CSV manifests, loads images,
applies split-aware augmentations.

Author : Dr. P.V.V. Kishore, KL University (KLEF)
Paper  : IndAgri-VLM (Computers and Electronics in Agriculture, 2026)
"""

import json
import pandas as pd
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ── Augmentation pipelines ─────────────────────────────────────────────────────

def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """Strong augmentation for training — reduces overfitting."""
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3,
            saturation=0.3, hue=0.1
        ),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet stats
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    """Minimal transforms for val/test — no randomness."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# ── Dataset class ──────────────────────────────────────────────────────────────

class AgriDataset(Dataset):
    """
    India-crop disease dataset.

    Args:
        csv_path   : Path to train.csv / val.csv / test.csv
        label_map  : Path to label_map.json
        split      : 'train' | 'val' | 'test'
        img_size   : Resize target (default 224)
        transform  : Optional custom transform (overrides default)
    """

    def __init__(
        self,
        csv_path   : str,
        label_map  : str,
        split      : str = "train",
        img_size   : int = 224,
        transform  : Optional[Callable] = None,
    ):
        self.split    = split
        self.img_size = img_size
        self.df       = pd.read_csv(csv_path)

        # Load label mapping
        with open(label_map) as f:
            lm = json.load(f)
        self.label2id = lm["label2id"]
        self.id2label = {int(k): v for k, v in lm["id2label"].items()}
        self.num_classes = len(self.label2id)

        # Set transform
        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = get_train_transforms(img_size)
        else:
            self.transform = get_val_transforms(img_size)

        # Filter to valid images only
        self._validate()

    def _validate(self) -> None:
        """Remove rows where image file does not exist."""
        before = len(self.df)
        self.df = self.df[
            self.df["image_path"].apply(lambda p: Path(p).exists())
        ].reset_index(drop=True)
        after = len(self.df)
        if before != after:
            print(f"[{self.split}] Dropped {before - after} missing images. "
                  f"Remaining: {after}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        row       = self.df.iloc[idx]
        img_path  = row["image_path"]
        label_str = row["label"]
        label_id  = self.label2id[label_str]

        try:
            image = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            # Return black image if file is corrupt
            image = Image.new("RGB", (self.img_size, self.img_size), 0)

        image = self.transform(image)
        return image, torch.tensor(label_id, dtype=torch.long)

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for weighted loss.
        Useful for imbalanced classes (e.g. corn has far fewer samples).
        """
        counts = self.df["label"].value_counts()
        weights = []
        for i in range(self.num_classes):
            lbl = self.id2label[i]
            cnt = counts.get(lbl, 1)
            weights.append(1.0 / cnt)
        weights = torch.tensor(weights, dtype=torch.float)
        weights = weights / weights.sum() * self.num_classes
        return weights


# ── DataLoader factory ─────────────────────────────────────────────────────────

def get_dataloaders(
    processed_dir : str,
    batch_size    : int = 32,
    img_size      : int = 224,
    num_workers   : int = 4,
) -> dict:
    """
    Returns dict with 'train', 'val', 'test' DataLoaders.

    Args:
        processed_dir : Path to data/processed/
        batch_size    : Batch size (default 32)
        img_size      : Image resize target (default 224)
        num_workers   : DataLoader workers (default 4)
    """
    processed = Path(processed_dir)
    label_map = str(processed / "label_map.json")

    datasets = {
        split: AgriDataset(
            csv_path  = str(processed / f"{split}.csv"),
            label_map = label_map,
            split     = split,
            img_size  = img_size,
        )
        for split in ["train", "val", "test"]
    }

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size  = batch_size,
            shuffle     = True,
            num_workers = num_workers,
            pin_memory  = True,
            drop_last   = True,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size  = batch_size * 2,
            shuffle     = False,
            num_workers = num_workers,
            pin_memory  = True,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size  = batch_size * 2,
            shuffle     = False,
            num_workers = num_workers,
            pin_memory  = True,
        ),
    }

    print("\n── DataLoader summary ──")
    for split, ds in datasets.items():
        print(f"  {split:>5}: {len(ds):>6} images | "
              f"{len(loaders[split]):>4} batches | "
              f"classes: {ds.num_classes}")

    return loaders, datasets


# ── Quick smoke test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    processed_dir = str(Path(root) / "data/processed")

    print("=" * 50)
    print("AgriDataset — smoke test")
    print("=" * 50)

    loaders, datasets = get_dataloaders(
        processed_dir = processed_dir,
        batch_size    = 8,
        img_size      = 224,
        num_workers   = 2,
    )

    # Test one batch from each split
    for split, loader in loaders.items():
        images, labels = next(iter(loader))
        print(f"\n[{split}] batch shape : {images.shape}")
        print(f"[{split}] labels      : {labels[:8].tolist()}")
        print(f"[{split}] image range : "
              f"[{images.min():.2f}, {images.max():.2f}]")

    # Class weights
    weights = datasets["train"].get_class_weights()
    print(f"\nClass weights shape : {weights.shape}")
    print(f"Class weights       : {weights.round(decimals=3)}")
    print("\n[DONE] Smoke test passed.")
