"""
train_qlora.py
--------------
QLoRA fine-tuning of Qwen2.5-VL-3B-Instruct for
South Indian crop disease diagnosis with Telugu output
and seasonal weather context conditioning.

Usage:
    python src/training/train_qlora.py --config configs/qlora_qwen25vl_3b.yaml

Author : Dr. P.V.V. Kishore, KL University (KLEF)
Paper  : IndAgri-VLM (Computers and Electronics in Agriculture, 2026)
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd
from PIL import Image
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
    prepare_model_for_kbit_training,
)
from tqdm import tqdm

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    handlers= [logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Weather context templates (AP/Telangana seasons) ──────────────────────────
WEATHER_TEMPLATES = [
    "Temp: 32°C, RH: 78%, Rainfall: 45mm last 30 days, Season: Kharif",
    "Temp: 28°C, RH: 65%, Rainfall: 12mm last 30 days, Season: Rabi",
    "Temp: 36°C, RH: 55%, Rainfall: 2mm last 30 days, Season: Summer",
    "Temp: 30°C, RH: 82%, Rainfall: 85mm last 30 days, Season: Monsoon",
    "Temp: 25°C, RH: 70%, Rainfall: 30mm last 30 days, Season: Post-monsoon",
]

GROWTH_STAGES = [
    "vegetative", "flowering", "fruiting",
    "maturity", "seedling", "tillering",
]

# ── Telugu disease labels ──────────────────────────────────────────────────────
TELUGU_LABELS = {
    "pepper_bacterial_spot"   : "మిరప బాక్టీరియల్ మచ్చ వ్యాధి",
    "pepper_healthy"          : "మిరప ఆరోగ్యంగా ఉంది",
    "potato_early_blight"     : "బంగాళాదుంప తొలి దద్దుర్లు",
    "potato_late_blight"      : "బంగాళాదుంప ఆలస్య దద్దుర్లు",
    "potato_healthy"          : "బంగాళాదుంప ఆరోగ్యంగా ఉంది",
    "tomato_bacterial_spot"   : "టమాటా బాక్టీరియల్ మచ్చ",
    "tomato_early_blight"     : "టమాటా తొలి దద్దుర్లు",
    "tomato_late_blight"      : "టమాటా ఆలస్య దద్దుర్లు",
    "tomato_leaf_mold"        : "టమాటా ఆకు బూజు",
    "tomato_septoria_leaf_spot": "టమాటా సెప్టోరియా మచ్చ",
    "tomato_spider_mites"     : "టమాటా సాలీడు పురుగులు",
    "tomato_target_spot"      : "టమాటా లక్ష్య మచ్చ",
    "tomato_yellow_leaf_curl" : "టమాటా పసుపు ఆకు మురి వైరస్",
    "tomato_mosaic_virus"     : "టమాటా మొజాయిక్ వైరస్",
    "tomato_healthy"          : "టమాటా ఆరోగ్యంగా ఉంది",
    "corn_leaf_blight"        : "మొక్కజొన్న ఆకు వడలు",
    "corn_rust"               : "మొక్కజొన్న తుప్పు",
    "corn_gray_leaf_spot"     : "మొక్కజొన్న బూడిద మచ్చ",
}

# ── VLM Dataset ────────────────────────────────────────────────────────────────
class IndAgriVLMDataset(Dataset):
    """
    Converts image+label pairs into VLM instruction-following format.
    Each sample: system prompt + weather context + image → Telugu diagnosis.
    """

    def __init__(
        self,
        csv_path   : str,
        label_map  : str,
        processor  ,
        split      : str = "train",
        img_size   : int = 224,
    ):
        self.df        = pd.read_csv(csv_path)
        self.processor = processor
        self.split     = split
        self.img_size  = img_size

        with open(label_map) as f:
            lm = json.load(f)
        self.label2id = lm["label2id"]
        self.id2label = {int(k): v for k, v in lm["id2label"].items()}

        # Filter missing files
        self.df = self.df[
            self.df["image_path"].apply(lambda p: Path(p).exists())
        ].reset_index(drop=True)
        log.info(f"[{split}] {len(self.df)} valid samples loaded.")

    def _build_prompt(self, label: str, weather: str, stage: str) -> tuple:
        """Build instruction-following prompt with Telugu output."""
        telugu = TELUGU_LABELS.get(label, label)
        english = label.replace("_", " ").title()

        system = (
            "You are an expert agricultural scientist specialising in "
            "South Indian crop diseases. Diagnose the disease in the image "
            "and provide treatment in English and Telugu."
        )
        user = (
            f"Weather context: {weather}. "
            f"Growth stage: {stage}. "
            f"What disease is present? Give diagnosis and treatment."
        )
        assistant = (
            f"Disease (English): {english}\n"
            f"వ్యాధి (Telugu): {telugu}\n"
            f"Diagnosis: The crop shows symptoms consistent with {english}. "
            f"Given the weather conditions ({weather}), this is a high-risk period.\n"
            f"Treatment: Apply recommended fungicide/bactericide as per "
            f"ICAR guidelines. Remove infected plant parts. "
            f"Ensure proper drainage and spacing."
        )
        return system, user, assistant

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        import random
        row     = self.df.iloc[idx]
        label   = row["label"]
        weather = random.choice(WEATHER_TEMPLATES)
        stage   = random.choice(GROWTH_STAGES)

        # Load image
        try:
            image = Image.open(row["image_path"]).convert("RGB")
            image = image.resize((self.img_size, self.img_size))
        except Exception:
            image = Image.new("RGB", (self.img_size, self.img_size), 0)

        system, user, assistant = self._build_prompt(label, weather, stage)

        # Build chat messages
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": user},
                ],
            },
            {"role": "assistant", "content": assistant},
        ]

        return {
            "messages" : messages,
            "label"    : self.label2id[label],
            "label_str": label,
        }


# ── Collate function ───────────────────────────────────────────────────────────
def collate_fn(batch, processor, device="cuda"):
    """Process batch of messages into model inputs."""
    texts = []
    images_list = []

    for item in batch:
        text = processor.apply_chat_template(
            item["messages"],
            tokenize         = False,
            add_generation_prompt = False,
        )
        texts.append(text)
        # Extract image from messages
        for msg in item["messages"]:
            if isinstance(msg["content"], list):
                for c in msg["content"]:
                    if c["type"] == "image":
                        images_list.append(c["image"])

    inputs = processor(
        text   = texts,
        images = images_list if images_list else None,
        return_tensors = "pt",
        padding        = True,
        truncation     = True,
        max_length     = 512,
    )
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Token-level accuracy on non-padding tokens."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    preds = shift_logits.argmax(dim=-1)
    mask  = shift_labels != -100
    if mask.sum() == 0:
        return 0.0
    correct = (preds == shift_labels) & mask
    return (correct.sum().float() / mask.sum().float()).item()


# ── Training loop ──────────────────────────────────────────────────────────────
def train(cfg: dict) -> None:
    """Main QLoRA training loop."""

    # ── Paths ──
    root       = Path(".")
    out_dir    = root / cfg["training"]["output_dir"]
    log_dir    = root / cfg["paths"]["logs"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device : {device}")
    log.info(f"GPU    : {torch.cuda.get_device_name(0)}")

    # ── BnB config ──
    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = cfg["qlora"]["load_in_4bit"],
        bnb_4bit_quant_type       = cfg["qlora"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype    = torch.bfloat16,
        bnb_4bit_use_double_quant = cfg["qlora"]["bnb_4bit_use_double_quant"],
    )

    # ── Load processor ──
    model_name = cfg["model"]["name"]
    log.info(f"Loading processor: {model_name}")
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code = True,
    )

    # ── Load model ──
    log.info("Loading model in 4-bit...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config = bnb_config,
        device_map          = "auto",
        torch_dtype         = torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing = cfg["training"]["gradient_checkpointing"],
    )

    # ── LoRA config ──
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
    label_map = cfg["data"]["label_map"]
    train_ds  = IndAgriVLMDataset(
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
        batch_size  = cfg["training"]["per_device_batch"] * 2,
        shuffle     = False,
        num_workers = cfg["training"]["dataloader_workers"],
        collate_fn  = lambda b: collate_fn(b, processor),
    )

    # ── Optimizer + scheduler ──
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = cfg["training"]["learning_rate"],
        weight_decay = cfg["training"]["weight_decay"],
    )
    total_steps  = len(train_loader) * cfg["training"]["num_epochs"]
    warmup_steps = int(total_steps * cfg["training"]["warmup_ratio"])
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )

    # ── Training ──
    log.info("=" * 60)
    log.info("Starting QLoRA fine-tuning")
    log.info(f"  Epochs        : {cfg['training']['num_epochs']}")
    log.info(f"  Train batches : {len(train_loader)}")
    log.info(f"  Total steps   : {total_steps}")
    log.info(f"  Warmup steps  : {warmup_steps}")
    log.info("=" * 60)

    best_val_loss = float("inf")
    history = []
    grad_accum = cfg["training"]["gradient_accumulation"]

    for epoch in range(1, cfg["training"]["num_epochs"] + 1):
        # ── Train epoch ──
        model.train()
        train_loss = 0.0
        train_acc  = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch}/{cfg['training']['num_epochs']} [train]")

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

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc  = train_acc  / len(train_loader)

        # ── Val epoch ──
        model.eval()
        val_loss = 0.0
        val_acc  = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader,
                              desc=f"Epoch {epoch} [val]"):
                batch = {k: v.to(device) for k, v in batch.items()
                         if isinstance(v, torch.Tensor)}
                outputs  = model(**batch)
                val_loss += outputs.loss.item()
                val_acc  += compute_accuracy(outputs.logits, batch["labels"])

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
            ckpt_path = out_dir / "best_model"
            model.save_pretrained(str(ckpt_path))
            processor.save_pretrained(str(ckpt_path))
            log.info(f"  → Best model saved: {ckpt_path}")

        # ── Checkpoint every epoch ──
        epoch_ckpt = out_dir / f"epoch_{epoch:02d}"
        model.save_pretrained(str(epoch_ckpt))
        log.info(f"  → Checkpoint saved: {epoch_ckpt}")

        history.append({
            "epoch"       : epoch,
            "train_loss"  : round(avg_train_loss, 4),
            "train_acc"   : round(avg_train_acc, 4),
            "val_loss"    : round(avg_val_loss, 4),
            "val_acc"     : round(avg_val_acc, 4),
        })

    # ── Save training history ──
    hist_path = Path("results/tables/training_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    log.info(f"Training history saved: {hist_path}")
    log.info("=" * 60)
    log.info("QLoRA fine-tuning complete.")
    log.info(f"Best val loss : {best_val_loss:.4f}")
    log.info("=" * 60)


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for IndAgri-VLM"
    )
    parser.add_argument(
        "--config",
        type    = str,
        default = "configs/qlora_qwen25vl_3b.yaml",
        help    = "Path to YAML config file",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log.info(f"Config loaded: {args.config}")
    log.info(f"Model        : {cfg['model']['name']}")

    torch.manual_seed(cfg["training"]["seed"])
    train(cfg)


if __name__ == "__main__":
    main()
