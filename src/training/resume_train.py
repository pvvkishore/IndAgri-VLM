"""
resume_train.py
---------------
Resumes QLoRA training from epoch 1 checkpoint.
Starts from epoch 2, loads best checkpoint weights.

Usage:
    python src/training/resume_train.py --config configs/qlora_qwen25vl_3b.yaml

Author : Dr. P.V.V. Kishore, KL University (KLEF)
Paper  : IndAgri-VLM (Computers and Electronics in Agriculture, 2026)
"""

import os
import sys
import json
import yaml
import argparse
import logging
import warnings
warnings.filterwarnings("ignore")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training,
)
from tqdm import tqdm

# Import shared components from train_qlora
sys.path.insert(0, str(Path(__file__).parent))
from train_qlora import (
    IndAgriVLMDataset,
    collate_fn,
    compute_accuracy,
    WEATHER_TEMPLATES,
    GROWTH_STAGES,
    TELUGU_LABELS,
)

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    handlers= [logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def resume_train(cfg: dict, start_epoch: int = 2) -> None:
    """Resume training from a saved checkpoint."""

    root    = Path(".")
    out_dir = root / cfg["training"]["output_dir"]
    log_dir = root / cfg["paths"]["logs"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device : {device}")
    log.info(f"GPU    : {torch.cuda.get_device_name(0)}")

    # ── BnB config ──
    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = torch.bfloat16,
        bnb_4bit_use_double_quant = True,
    )

    # ── Load processor ──
    model_name = cfg["model"]["name"]
    log.info(f"Loading processor: {model_name}")
    processor = AutoProcessor.from_pretrained(
        model_name, trust_remote_code=True
    )

    # ── Check for saved checkpoint ──
    ckpt_path = out_dir / "best_model"
    epoch1_path = out_dir / "epoch_01"

    if ckpt_path.exists():
        resume_from = str(ckpt_path)
        log.info(f"Resuming from best_model checkpoint: {resume_from}")
    elif epoch1_path.exists():
        resume_from = str(epoch1_path)
        log.info(f"Resuming from epoch_01 checkpoint: {resume_from}")
    else:
        log.warning("No checkpoint found — starting from base model.")
        resume_from = model_name

    # ── Load base model ──
    log.info("Loading base model in 4-bit...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config = bnb_config,
        device_map          = "auto",
        torch_dtype         = torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )

    # ── Load LoRA weights if checkpoint exists ──
    if resume_from != model_name:
        log.info("Loading LoRA adapter weights...")
        model = PeftModel.from_pretrained(
            model, resume_from, is_trainable=True
        )
        log.info("LoRA weights loaded.")
    else:
        lora_cfg = LoraConfig(
            r              = cfg["lora"]["r"],
            lora_alpha     = cfg["lora"]["lora_alpha"],
            lora_dropout   = cfg["lora"]["lora_dropout"],
            bias           = cfg["lora"]["bias"],
            task_type      = TaskType.CAUSAL_LM,
            target_modules = cfg["lora"]["target_modules"],
        )
        model = get_peft_model(model, lora_cfg)

    model.print_trainable_parameters()

    # ── Datasets ──
    label_map    = cfg["data"]["label_map"]
    train_ds = IndAgriVLMDataset(
        cfg["data"]["train_csv"], label_map, processor, "train"
    )
    val_ds = IndAgriVLMDataset(
        cfg["data"]["val_csv"], label_map, processor, "val"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg["training"]["per_device_batch"],
        shuffle     = True,
        num_workers = cfg["training"]["dataloader_workers"],
        collate_fn  = lambda b: collate_fn(b, processor),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = 1,
        shuffle     = False,
        num_workers = 1,
        collate_fn  = lambda b: collate_fn(b, processor),
    )

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = cfg["training"]["learning_rate"],
        weight_decay = cfg["training"]["weight_decay"],
    )

    num_epochs   = cfg["training"]["num_epochs"]
    total_steps  = len(train_loader) * (num_epochs - start_epoch + 1)
    warmup_steps = int(total_steps * cfg["training"]["warmup_ratio"])
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )

    grad_accum   = cfg["training"]["gradient_accumulation"]
    best_val_loss = float("inf")

    # Load existing history if any
    hist_path = Path("results/tables/training_history.json")
    if hist_path.exists():
        with open(hist_path) as f:
            history = json.load(f)
        log.info(f"Loaded existing history ({len(history)} epochs)")
    else:
        history = []

    log.info("=" * 60)
    log.info(f"Resuming from epoch {start_epoch}/{num_epochs}")
    log.info(f"Train batches : {len(train_loader)}")
    log.info(f"Val batches   : {len(val_loader)}")
    log.info("=" * 60)

    for epoch in range(start_epoch, num_epochs + 1):

        # ── Train ──
        model.train()
        train_loss = 0.0
        train_acc  = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch}/{num_epochs} [train]")
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()
                     if isinstance(v, torch.Tensor)}
            outputs = model(**batch)
            loss    = outputs.loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg["training"]["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            acc = compute_accuracy(outputs.logits, batch["labels"])
            train_loss += outputs.loss.item()
            train_acc  += acc
            pbar.set_postfix({
                "loss": f"{outputs.loss.item():.4f}",
                "acc" : f"{acc:.3f}",
                "lr"  : f"{scheduler.get_last_lr()[0]:.2e}",
            })

            # Free cache every 500 steps
            if step % 500 == 0:
                torch.cuda.empty_cache()

            # Save mid-epoch checkpoint every 2000 steps
            if step > 0 and step % 2000 == 0:
                step_ckpt = out_dir / f"epoch_{epoch:02d}_step_{step:05d}"
                model.save_pretrained(str(step_ckpt))
                log.info(f"  → Mid-epoch checkpoint: {step_ckpt}")

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc  = train_acc  / len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_acc  = 0.0
        torch.cuda.empty_cache()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                batch = {k: v.to(device) for k, v in batch.items()
                         if isinstance(v, torch.Tensor)}
                outputs   = model(**batch)
                val_loss += outputs.loss.item()
                val_acc  += compute_accuracy(outputs.logits, batch["labels"])
                torch.cuda.empty_cache()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc  = val_acc  / len(val_loader)

        log.info(
            f"Epoch {epoch} | "
            f"train_loss={avg_train_loss:.4f} train_acc={avg_train_acc:.4f} | "
            f"val_loss={avg_val_loss:.4f} val_acc={avg_val_acc:.4f}"
        )

        # ── Save best ──
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt = out_dir / "best_model"
            model.save_pretrained(str(ckpt))
            processor.save_pretrained(str(ckpt))
            log.info(f"  → Best model saved: {ckpt}")

        # ── Save epoch checkpoint ──
        epoch_ckpt = out_dir / f"epoch_{epoch:02d}"
        model.save_pretrained(str(epoch_ckpt))
        log.info(f"  → Checkpoint: {epoch_ckpt}")

        history.append({
            "epoch"     : epoch,
            "train_loss": round(avg_train_loss, 4),
            "train_acc" : round(avg_train_acc,  4),
            "val_loss"  : round(avg_val_loss,   4),
            "val_acc"   : round(avg_val_acc,    4),
        })

        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

    log.info("=" * 60)
    log.info("Training complete.")
    log.info(f"Best val loss : {best_val_loss:.4f}")
    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qlora_qwen25vl_3b.yaml")
    parser.add_argument("--start_epoch", type=int, default=2)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log.info(f"Config : {args.config}")
    log.info(f"Model  : {cfg['model']['name']}")
    torch.manual_seed(cfg["training"]["seed"])
    resume_train(cfg, start_epoch=args.start_epoch)


if __name__ == "__main__":
    main()
