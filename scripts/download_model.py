"""
download_model.py
-----------------
Downloads Qwen2.5-VL-3B-Instruct from HuggingFace Hub
and saves to models/checkpoints/base_model/.

Usage:
    python scripts/download_model.py

Author : Dr. P.V.V. Kishore, KL University (KLEF)
Paper  : IndAgri-VLM (Computers and Electronics in Agriculture, 2026)
"""

import os
import sys
from pathlib import Path

def main():
    save_dir = Path("models/checkpoints/base_model")
    save_dir.mkdir(parents=True, exist_ok=True)

    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

    print("=" * 60)
    print(f"Downloading : {model_name}")
    print(f"Save dir    : {save_dir.resolve()}")
    print("=" * 60)

    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    except ImportError:
        print("[ERROR] transformers not installed.")
        sys.exit(1)

    print("\nDownloading processor...")
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    processor.save_pretrained(str(save_dir))
    print("[OK] Processor saved.")

    print("\nDownloading model weights (4-bit NF4)...")
    import torch
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit               = True,
        bnb_4bit_quant_type        = "nf4",
        bnb_4bit_compute_dtype     = torch.bfloat16,
        bnb_4bit_use_double_quant  = True,
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config = bnb_config,
        device_map          = "auto",
        trust_remote_code   = True,
    )

    print("\n[OK] Model loaded in 4-bit.")
    print(f"     Parameters : {sum(p.numel() for p in model.parameters()):,}")

    # VRAM check
    import subprocess
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free",
         "--format=csv,noheader"],
        capture_output=True, text=True
    )
    print(f"     VRAM usage : {result.stdout.strip()}")
    print("\n[DONE] Model ready for QLoRA fine-tuning.")


if __name__ == "__main__":
    main()
