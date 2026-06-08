"""
download_data.py
----------------
Downloads PlantVillage, PlantDoc datasets from Kaggle.
Prints class distribution and flags India-relevant crop classes.

Usage:
    python scripts/download_data.py --dataset plantvillage
    python scripts/download_data.py --dataset plantdoc
    python scripts/download_data.py --dataset all

Author : Dr. P.V.V. Kishore, KL University (KLEF)
Paper  : IndAgri-VLM (Computers and Electronics in Agriculture, 2026)
"""

import os
import argparse
import zipfile
from pathlib import Path
from collections import Counter
import json

# ── India-relevant crop classes (AP/Telangana focus) ──────────────────────────
INDIA_CROPS = {
    "Chili__healthy",
    "Chili__leaf_curl",
    "Chili__leaf_spot",
    "Chili__whitefly",
    "Chili__yellowish",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Cotton___Bacterial_blight",
    "Cotton___Curl_Virus",
    "Cotton___Fussarium_Wilt",
    "Cotton___healthy",
    "Groundnut___Early_leaf_spot",
    "Groundnut___Late_leaf_spot",
    "Groundnut___healthy",
    "Coconut___Bud_rot",
    "Coconut___Gray_leaf_spot",
    "Coconut___healthy",
    "Rice___Brown_spot",
    "Rice___Leaf_blast",
    "Rice___Neck_blast",
    "Rice___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
}

# ── Dataset configs ────────────────────────────────────────────────────────────
DATASETS = {
    "plantvillage": {
        "kaggle_id" : "emmarex/plantdisease",
        "zip_name"  : "plantdisease.zip",
        "out_dir"   : "data/raw/plantvillage",
        "description": "PlantVillage — 54K leaf images, 38 classes",
    },
    "plantdoc": {
        "kaggle_id" : "nirmalsankalana/plantdoc-dataset",
        "zip_name"  : "plantdoc-dataset.zip",
        "out_dir"   : "data/raw/plantdoc",
        "description": "PlantDoc — 2.5K real-field images, 27 diseases",
    },
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def download_dataset(name: str, project_root: Path) -> None:
    cfg = DATASETS[name]
    out_dir = project_root / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Downloading : {cfg['description']}")
    print(f"Kaggle ID   : {cfg['kaggle_id']}")
    print(f"Output dir  : {out_dir}")
    print(f"{'='*60}")

    cmd = (
        f"kaggle datasets download -d {cfg['kaggle_id']} "
        f"--path {out_dir} --unzip"
    )
    ret = os.system(cmd)
    if ret != 0:
        print(f"[ERROR] Download failed for {name}. Check kaggle auth.")
        return

    print(f"\n[OK] {name} downloaded.")
    inspect_dataset(out_dir, name)


def inspect_dataset(data_dir: Path, name: str) -> None:
    """Count images per class and flag India-relevant classes."""
    print(f"\n── Class distribution: {name} ──")

    counts = Counter()
    for cls_dir in sorted(data_dir.rglob("*")):
        if cls_dir.is_dir():
            imgs = list(cls_dir.glob("*.jpg")) + \
                   list(cls_dir.glob("*.JPG")) + \
                   list(cls_dir.glob("*.png")) + \
                   list(cls_dir.glob("*.PNG")) + \
                   list(cls_dir.glob("*.jpeg"))
            if imgs:
                counts[cls_dir.name] = len(imgs)

    total = sum(counts.values())
    india_total = 0
    india_classes = []

    print(f"\n{'Class':<50} {'Count':>6}  {'India?':>6}")
    print("-" * 66)
    for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        is_india = any(ic.lower() in cls.lower() for ic in
                       ["chili", "pepper", "tomato", "cotton",
                        "groundnut", "coconut", "rice", "corn",
                        "maize", "potato", "mango", "banana"])
        marker = "  ★" if is_india else ""
        print(f"{cls:<50} {cnt:>6}{marker}")
        if is_india:
            india_total += cnt
            india_classes.append(cls)

    print("-" * 66)
    print(f"{'TOTAL':<50} {total:>6}")
    print(f"{'INDIA-RELEVANT TOTAL':<50} {india_total:>6}")
    print(f"\nIndia-relevant classes found: {len(india_classes)}")

    # Save summary to results
    summary = {
        "dataset"       : name,
        "total_images"  : total,
        "total_classes" : len(counts),
        "india_images"  : india_total,
        "india_classes" : india_classes,
        "all_classes"   : dict(counts),
    }
    summary_path = Path("results/tables") / f"{name}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Saved] Summary → {summary_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Download and inspect agricultural datasets"
    )
    parser.add_argument(
        "--dataset",
        choices=["plantvillage", "plantdoc", "all"],
        default="plantvillage",
        help="Which dataset to download (default: plantvillage)",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Project root directory (default: current dir)",
    )
    args = parser.parse_args()

    project_root = Path(args.root).resolve()
    print(f"Project root : {project_root}")

    if args.dataset == "all":
        for name in DATASETS:
            download_dataset(name, project_root)
    else:
        download_dataset(args.dataset, project_root)


if __name__ == "__main__":
    main()
