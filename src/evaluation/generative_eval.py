"""
generative_eval.py
------------------
Real generative evaluation of IndAgri-VLM.
Runs .generate() inference (not teacher-forcing).
Parses Disease (English): line from model output.
Computes accuracy, macro-F1, per-class metrics.

Usage:
    python src/evaluation/generative_eval.py \
        --checkpoint models/checkpoints/qlora_qwen25vl_3b_V2/epoch_05 \
        --split test \
        --n_samples 500

Author : Dr. P.V.V. Kishore, KL University (KLEF)
Paper  : IndAgri-VLM (Computers and Electronics in Agriculture, 2026)
"""

import os
import re
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel

# ── Label utilities ────────────────────────────────────────────────────────────
def load_label_map(path: str) -> dict:
    with open(path) as f:
        lm = json.load(f)
    return lm["label2id"], {int(k): v for k, v in lm["id2label"].items()}

def normalise(text: str) -> str:
    """Normalise label string for fuzzy matching."""
    return re.sub(r"[^a-z0-9]", "_", text.lower().strip()).strip("_")

def parse_prediction(output: str, id2label: dict) -> str | None:
    """
    Parse 'Disease (English): <label>' from generated text.
    Falls back to fuzzy match against all known labels.
    """
    # Primary: exact line match
    match = re.search(
        r"Disease\s*\(English\)\s*[:\-]\s*(.+)", output, re.IGNORECASE
    )
    if match:
        raw = match.group(1).strip().split("\n")[0]
        norm = normalise(raw)
        # Direct match
        for label in id2label.values():
            if normalise(label) == norm:
                return label
        # Partial match
        for label in id2label.values():
            if norm in normalise(label) or normalise(label) in norm:
                return label

    # Fallback: scan output for any known label
    norm_out = normalise(output)
    for label in id2label.values():
        if normalise(label) in norm_out:
            return label
    return None

# ── Prompt builder ─────────────────────────────────────────────────────────────
WEATHER_DEFAULT = "Temp: 32°C, RH: 78%, Rainfall: 45mm last 30 days, Season: Kharif"
STAGE_DEFAULT   = "flowering"

def build_messages(image: Image.Image) -> list:
    return [
        {
            "role": "system",
            "content": (
                "You are an expert agricultural scientist specialising in "
                "South Indian crop diseases. Diagnose the disease in the image "
                "and provide treatment in English and Telugu."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        f"Weather context: {WEATHER_DEFAULT}. "
                        f"Growth stage: {STAGE_DEFAULT}. "
                        "What disease is present? Give diagnosis and treatment."
                    ),
                },
            ],
        },
    ]

# ── Main evaluation ────────────────────────────────────────────────────────────
def evaluate(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # Load label map
    label2id, id2label = load_label_map(args.label_map)
    all_labels = list(id2label.values())

    # Load CSV split
    df = pd.read_csv(args.csv)
    df = df[df["image_path"].apply(lambda p: Path(p).exists())]
    if args.n_samples and args.n_samples < len(df):
        df = df.sample(n=args.n_samples, random_state=42)
    print(f"Evaluating on {len(df)} samples from '{args.split}' split")

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(
        args.base_model, trust_remote_code=True
    )
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, args.checkpoint)
    model.eval()
    print("Model loaded.")

    # Inference loop
    y_true, y_pred, raw_outputs, failed = [], [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        true_label = row["label"]
        try:
            image = Image.open(row["image_path"]).convert("RGB")
            image = image.resize((224, 224))
        except Exception:
            failed.append(row["image_path"])
            continue

        messages = build_messages(image)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens  = 150,
                do_sample       = False,
                temperature     = 1.0,
                repetition_penalty = 1.1,
            )

        # Decode only new tokens
        new_ids = out_ids[:, inputs["input_ids"].shape[1]:]
        output_text = processor.batch_decode(
            new_ids, skip_special_tokens=True
        )[0]

        pred_label = parse_prediction(output_text, id2label)
        if pred_label is None:
            pred_label = "__unknown__"

        y_true.append(true_label)
        y_pred.append(pred_label)
        raw_outputs.append({
            "image_path" : row["image_path"],
            "true_label" : true_label,
            "pred_label" : pred_label,
            "output_text": output_text[:300],
        })

        torch.cuda.empty_cache()

    # Metrics
    valid = [(t, p) for t, p in zip(y_true, y_pred) if p != "__unknown__"]
    y_true_v = [t for t, _ in valid]
    y_pred_v = [p for _, p in valid]

    acc    = accuracy_score(y_true_v, y_pred_v)
    macro_f1 = f1_score(y_true_v, y_pred_v, average="macro", zero_division=0)
    parsed_rate = len(valid) / len(y_true) if y_true else 0

    print("\n" + "=" * 60)
    print("IndAgri-VLM — Generative Evaluation Results")
    print("=" * 60)
    print(f"Samples evaluated : {len(y_true)}")
    print(f"Parse rate        : {parsed_rate:.3f} ({len(valid)}/{len(y_true)})")
    print(f"Top-1 Accuracy    : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Macro-F1          : {macro_f1:.4f}")
    print("\nPer-class report:")
    print(classification_report(
        y_true_v, y_pred_v, zero_division=0, digits=3
    ))

    # Save results
    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "checkpoint"   : args.checkpoint,
        "split"        : args.split,
        "n_samples"    : len(y_true),
        "parse_rate"   : round(parsed_rate, 4),
        "accuracy"     : round(acc, 4),
        "macro_f1"     : round(macro_f1, 4),
        "failed_images": len(failed),
    }
    results_path = out_dir / f"eval_{args.split}_{Path(args.checkpoint).name}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] Results → {results_path}")

    # Save raw outputs
    raw_path = out_dir / f"raw_outputs_{args.split}_{Path(args.checkpoint).name}.json"
    with open(raw_path, "w") as f:
        json.dump(raw_outputs, f, indent=2, ensure_ascii=False)
    print(f"[Saved] Raw outputs → {raw_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generative evaluation of IndAgri-VLM"
    )
    parser.add_argument(
        "--checkpoint",
        default="models/checkpoints/qlora_qwen25vl_3b_V2/epoch_05",
    )
    parser.add_argument(
        "--base_model",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Override CSV path (default: data/processed/<split>.csv)",
    )
    parser.add_argument(
        "--label_map",
        default="data/processed/label_map.json",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=500,
        help="Number of samples to evaluate (default: 500)",
    )
    args = parser.parse_args()

    if args.csv is None:
        args.csv = f"data/processed/{args.split}.csv"

    evaluate(args)


if __name__ == "__main__":
    main()
