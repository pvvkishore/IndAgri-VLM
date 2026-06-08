"""
prepare_dataset.py
------------------
Merges PlantVillage and PlantDoc datasets.
Filters to India-relevant classes.
Generates train/val/test splits (70/15/15).
Saves manifests as CSV to data/processed/.

Usage:
    python src/dataset/prepare_dataset.py

Author : Dr. P.V.V. Kishore, KL University (KLEF)
Paper  : IndAgri-VLM (Computers and Electronics in Agriculture, 2026)
"""

import os
import json
import random
import shutil
import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)

# ── India-relevant class mapping ───────────────────────────────────────────────
# Maps raw folder names → unified canonical label
CLASS_MAP = {
    # PlantVillage classes
    "Pepper__bell___Bacterial_spot"     : "pepper_bacterial_spot",
    "Pepper__bell___healthy"            : "pepper_healthy",
    "Potato___Early_blight"             : "potato_early_blight",
    "Potato___Late_blight"              : "potato_late_blight",
    "Potato___healthy"                  : "potato_healthy",
    "Tomato_Bacterial_spot"             : "tomato_bacterial_spot",
    "Tomato_Early_blight"               : "tomato_early_blight",
    "Tomato_Late_blight"                : "tomato_late_blight",
    "Tomato_Leaf_Mold"                  : "tomato_leaf_mold",
    "Tomato_Septoria_leaf_spot"         : "tomato_septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite" : "tomato_spider_mites",
    "Tomato__Target_Spot"               : "tomato_target_spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus" : "tomato_yellow_leaf_curl",
    "Tomato__Tomato_mosaic_virus"       : "tomato_mosaic_virus",
    "Tomato_healthy"                    : "tomato_healthy",
    # PlantDoc classes
    "Bell_pepper_leaf_spot"             : "pepper_bacterial_spot",
    "Bell_pepper_leaf"                  : "pepper_healthy",
    "Potato_leaf_early_blight"          : "potato_early_blight",
    "Potato_leaf_late_blight"           : "potato_late_blight",
    "Tomato_leaf_bacterial_spot"        : "tomato_bacterial_spot",
    "Tomato_Early_blight_leaf"          : "tomato_early_blight",
    "Tomato_leaf_late_blight"           : "tomato_late_blight",
    "Tomato_mold_leaf"                  : "tomato_leaf_mold",
    "Tomato_Septoria_leaf_spot"         : "tomato_septoria_leaf_spot",
    "Tomato_leaf_yellow_virus"          : "tomato_yellow_leaf_curl",
    "Tomato_leaf_mosaic_virus"          : "tomato_mosaic_virus",
    "Tomato_two_spotted_spider_mites_leaf" : "tomato_spider_mites",
    "Tomato_leaf"                       : "tomato_healthy",
    "Corn_leaf_blight"                  : "corn_leaf_blight",
    "Corn_rust_leaf"                    : "corn_rust",
    "Corn_Gray_leaf_spot"               : "corn_gray_leaf_spot",
}

# ── Split ratios ───────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def collect_images(root: Path, source_name: str) -> list[dict]:
    """Walk a dataset root and collect image paths with labels."""
    records = []
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        canonical = CLASS_MAP.get(class_dir.name)
        if canonical is None:
            continue  # skip non-India classes
        for img_path in class_dir.iterdir():
            if img_path.suffix in IMG_EXTENSIONS:
                records.append({
                    "image_path"    : str(img_path),
                    "raw_label"     : class_dir.name,
                    "label"         : canonical,
                    "source"        : source_name,
                })
    return records


def split_records(records: list[dict]) -> tuple:
    """Stratified split by label."""
    by_label = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)

    train, val, test = [], [], []
    for label, items in by_label.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)
        train  += items[:n_train]
        val    += items[n_train:n_train + n_val]
        test   += items[n_train + n_val:]
    return train, val, test


def save_split(records: list[dict], path: Path, split_name: str) -> None:
    df = pd.DataFrame(records)
    df["split"] = split_name
    df.to_csv(path, index=False)
    print(f"  [{split_name:>5}] {len(df):>5} images → {path}")


def print_summary(train, val, test) -> None:
    all_records = train + val + test
    df = pd.DataFrame(all_records)
    print("\n── Class distribution after merge + filter ──")
    print(f"\n{'Label':<40} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>6}")
    print("-" * 62)
    labels = sorted(df["label"].unique())
    for lbl in labels:
        sub = df[df["label"] == lbl]
        tr = len(sub[sub["split"] == "train"]) if "split" in sub else 0
        counts = {
            "train" : sum(1 for r in train if r["label"] == lbl),
            "val"   : sum(1 for r in val   if r["label"] == lbl),
            "test"  : sum(1 for r in test  if r["label"] == lbl),
        }
        tot = counts["train"] + counts["val"] + counts["test"]
        print(f"{lbl:<40} {counts['train']:>6} {counts['val']:>6} "
              f"{counts['test']:>6} {tot:>6}")
    print("-" * 62)
    print(f"{'TOTAL':<40} {len(train):>6} {len(val):>6} "
          f"{len(test):>6} {len(all_records):>6}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare merged India-crop dataset"
    )
    parser.add_argument("--root", default=".",
                        help="Project root (default: .)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    pv_dir  = root / "data/raw/plantvillage/PlantVillage"
    pd_train = root / "data/raw/plantdoc/train"
    pd_test  = root / "data/raw/plantdoc/test"
    out_dir  = root / "data/processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("IndAgri-VLM — Dataset Preparation")
    print("=" * 60)

    # Collect
    print("\nCollecting images...")
    records = []
    records += collect_images(pv_dir,   "plantvillage")
    records += collect_images(pd_train, "plantdoc_train")
    records += collect_images(pd_test,  "plantdoc_test")
    print(f"Total collected : {len(records)} images")

    # Split
    print("\nGenerating stratified splits (70/15/15)...")
    train, val, test = split_records(records)

    # Save manifests
    print("\nSaving CSV manifests...")
    save_split(train, out_dir / "train.csv", "train")
    save_split(val,   out_dir / "val.csv",   "val")
    save_split(test,  out_dir / "test.csv",  "test")

    # Summary
    print_summary(train, val, test)

    # Save label map
    labels = sorted({r["label"] for r in records})
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    label_map = {"label2id": label2id, "id2label": id2label}
    lm_path = out_dir / "label_map.json"
    with open(lm_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"\n[Saved] Label map ({len(labels)} classes) → {lm_path}")
    print("\n[DONE] Dataset preparation complete.")


if __name__ == "__main__":
    main()
