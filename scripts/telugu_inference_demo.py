"""
telugu_inference_demo.py
------------------------
Demonstrates IndAgri-VLM's Telugu-language crop disease diagnosis.
Takes a crop image → outputs diagnosis + treatment in English and Telugu.
Used to generate qualitative examples for the paper (Table 3 / Figure 5).

Usage:
    python scripts/telugu_inference_demo.py --image path/to/image.jpg
    python scripts/telugu_inference_demo.py --class_name tomato_late_blight
    python scripts/telugu_inference_demo.py --run_all

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
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel

# ── Config ─────────────────────────────────────────────────────────────────────
CHECKPOINT   = "models/checkpoints/qlora_qwen25vl_3b_V2/epoch_05"
BASE_MODEL   = "Qwen/Qwen2.5-VL-3B-Instruct"
LABEL_MAP    = "data/processed/label_map.json"
TEST_CSV     = "data/processed/test.csv"

# Representative AP/Telangana weather contexts
WEATHER_CONTEXTS = {
    "kharif_monsoon"   : "Temp: 30°C, RH: 85%, Rainfall: 92mm/30d, Season: Kharif Monsoon",
    "rabi_winter"      : "Temp: 22°C, RH: 60%, Rainfall: 8mm/30d, Season: Rabi Winter",
    "summer_dry"       : "Temp: 38°C, RH: 42%, Rainfall: 1mm/30d, Season: Summer Dry",
    "post_monsoon"     : "Temp: 28°C, RH: 72%, Rainfall: 35mm/30d, Season: Post-Monsoon",
}

DEMO_CLASSES = [
    "tomato_late_blight",
    "tomato_yellow_leaf_curl",
    "pepper_bacterial_spot",
    "potato_early_blight",
    "tomato_early_blight",
    "tomato_septoria_leaf_spot",
]


def load_model(checkpoint: str):
    print(f"Loading model from: {checkpoint}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL, trust_remote_code=True
    )
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, checkpoint)
    model.eval()
    print("Model ready.\n")
    return model, processor


def run_inference(
    model, processor, image: Image.Image,
    weather: str = None, stage: str = "flowering"
) -> str:
    if weather is None:
        weather = WEATHER_CONTEXTS["kharif_monsoon"]

    device = next(model.parameters()).device

    messages = [
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
                        f"Weather context: {weather}. "
                        f"Growth stage: {stage}. "
                        "What disease is present? Give diagnosis and treatment."
                    ),
                },
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[image],
        return_tensors="pt", padding=True
    ).to(device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens     = 200,
            do_sample          = False,
            repetition_penalty = 1.1,
        )

    new_ids = out_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new_ids, skip_special_tokens=True)[0]


def get_sample_image(class_name: str) -> Image.Image | None:
    """Get a sample image for a given class from the test set."""
    df = pd.read_csv(TEST_CSV)
    subset = df[df["label"] == class_name]
    if subset.empty:
        df2 = pd.read_csv("data/processed/val.csv")
        subset = df2[df2["label"] == class_name]
    if subset.empty:
        return None, None
    row = subset.sample(1, random_state=42).iloc[0]
    try:
        img = Image.open(row["image_path"]).convert("RGB").resize((224, 224))
        return img, row["image_path"]
    except Exception:
        return None, None


def print_demo(class_name: str, output: str, weather: str, img_path: str):
    print("=" * 70)
    print(f"Class        : {class_name}")
    print(f"Image        : {Path(img_path).name if img_path else 'N/A'}")
    print(f"Weather      : {weather}")
    print("-" * 70)
    print("Model Output :")
    print(output)
    print("=" * 70)
    print()


def run_demo_all(model, processor):
    """Run inference on representative samples from 6 key classes."""
    results = []
    weather = WEATHER_CONTEXTS["kharif_monsoon"]

    for class_name in DEMO_CLASSES:
        image, img_path = get_sample_image(class_name)
        if image is None:
            print(f"[SKIP] No image found for: {class_name}")
            continue

        output = run_inference(model, processor, image, weather)
        print_demo(class_name, output, weather, img_path)

        results.append({
            "class"      : class_name,
            "image_path" : img_path,
            "weather"    : weather,
            "output"     : output,
        })

        torch.cuda.empty_cache()

    # Save qualitative results for paper Table 3
    out_path = Path("results/tables/telugu_demo_outputs.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[Saved] Telugu demo outputs → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="IndAgri-VLM Telugu inference demo"
    )
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--image",      default=None,
                        help="Path to a specific image file")
    parser.add_argument("--class_name", default=None,
                        help="Run on a sample from this class")
    parser.add_argument("--weather",    default="kharif_monsoon",
                        choices=list(WEATHER_CONTEXTS.keys()))
    parser.add_argument("--stage",      default="flowering")
    parser.add_argument("--run_all",    action="store_true",
                        help="Run demo on all 6 representative classes")
    args = parser.parse_args()

    model, processor = load_model(args.checkpoint)
    weather_str = WEATHER_CONTEXTS[args.weather]

    if args.run_all:
        run_demo_all(model, processor)

    elif args.image:
        image = Image.open(args.image).convert("RGB").resize((224, 224))
        output = run_inference(model, processor, image, weather_str, args.stage)
        print_demo("user_image", output, weather_str, args.image)

    elif args.class_name:
        image, img_path = get_sample_image(args.class_name)
        if image is None:
            print(f"No image found for class: {args.class_name}")
            return
        output = run_inference(model, processor, image, weather_str, args.stage)
        print_demo(args.class_name, output, weather_str, img_path)

    else:
        # Default: run on tomato_late_blight
        image, img_path = get_sample_image("tomato_late_blight")
        if image:
            output = run_inference(model, processor, image, weather_str, args.stage)
            print_demo("tomato_late_blight", output, weather_str, img_path)


if __name__ == "__main__":
    main()
