#!/usr/bin/env python3
"""Standard LoRA fine-tuning for JSON extraction.

Uses PEFT (HuggingFace) for LoRA — no custom adapter code.

Usage:
    # Local (0.5B, CPU/MPS)
    python src/train.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --data data/Restaurants_1_train.jsonl --epochs 5 --device cpu

    # RunPod (7B/32B, CUDA)
    python src/train.py --model Qwen/Qwen2.5-7B-Instruct \
        --data data/Restaurants_1_train.jsonl --epochs 10 --device cuda
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig, get_peft_model, TaskType


def seed_everything(seed: int) -> None:
    """Seed every RNG that affects training output.

    Pinning this is essential for the paper: without it, retraining at the
    same scale on the same data produces a numerically different LoRA
    each run (PyTorch CUDA kernel order, dropout, dataloader shuffling).
    The qualitative per-role story holds across runs but specific decimals
    do not — pinning the seed makes the published numbers reproducible.

    Sources covered:
      - Python `random` (used by some HF utilities).
      - `torch` CPU + CUDA RNGs (LoRA init, dropout, shuffling).
      - `transformers.set_seed` (dropout, label smoothing init, etc.).
      - cuDNN: deterministic algorithms, no autotune.
      - cuBLAS: deterministic workspace via env var.
      - `torch.use_deterministic_algorithms(warn_only=True)`: opt into
        deterministic kernels where available; warn rather than crash on
        the (small) set of ops that lack a deterministic impl.
      - `PYTHONHASHSEED`: dict iteration / set ordering.
      - `numpy.random` (defensive — not currently used directly).
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def _seeded_worker_init_fn(worker_id: int) -> None:
    """DataLoader worker initializer: re-seed each worker deterministically.
    Only matters if num_workers>0; harmless otherwise."""
    base = torch.initial_seed() & 0xFFFFFFFF
    random.seed(base + worker_id)
    try:
        import numpy as np
        np.random.seed((base + worker_id) & 0xFFFFFFFF)
    except ImportError:
        pass


class JsonExtractionDataset(Dataset):
    """Dataset of (prompt, target_json) pairs for causal LM training."""

    def __init__(self, data_path: str, tokenizer, max_seq_len: int = 2048):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        with open(data_path) as f:
            for line in f:
                rec = json.loads(line)
                self.examples.append(rec)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt = ex["prompt"]
        target = ex["target_json"]

        # Tokenize prompt and target.
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        target_ids = self.tokenizer.encode(target, add_special_tokens=False)
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            target_ids = target_ids + [eos_id]

        # Reserve the full target budget; truncate the prompt from the
        # front if needed so the JSON target is always present in the
        # labels. The previous implementation right-truncated the
        # concatenation, which silently dropped every target token on
        # CUAD-length prompts and produced NaN loss (labels = all -100).
        prompt_budget = self.max_seq_len - len(target_ids)
        if prompt_budget < 1:
            raise ValueError(
                f"example {idx}: target alone is {len(target_ids)} tokens, "
                f"exceeds max_seq_len={self.max_seq_len}; raise --max-seq-len"
            )
        if len(prompt_ids) > prompt_budget:
            # Keep the tail of the prompt — it contains the most recent /
            # most query-adjacent context, which is what the target was
            # written in response to.
            prompt_ids = prompt_ids[-prompt_budget:]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        # By construction len(input_ids) == len(labels) <= max_seq_len.

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch):
    """Pad batch to same length."""
    max_len = max(len(b["input_ids"]) for b in batch)

    input_ids = []
    labels = []
    attention_mask = []

    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        input_ids.append(F.pad(b["input_ids"], (0, pad_len), value=0))
        labels.append(F.pad(b["labels"], (0, pad_len), value=-100))
        mask = torch.ones(len(b["input_ids"]), dtype=torch.long)
        attention_mask.append(F.pad(mask, (0, pad_len), value=0))

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }


def train_epoch(model, dataloader, optimizer, device, epoch, grad_clip=1.0):
    """Train one epoch, return mean loss."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    start = time.time()

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        # Count non-masked tokens for reporting
        n_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    elapsed = time.time() - start
    mean_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    print(f"  Epoch {epoch}: loss={mean_loss:.4f} "
          f"tokens={total_tokens} time={elapsed:.1f}s")
    return mean_loss


def main():
    parser = argparse.ArgumentParser(description="Standard LoRA fine-tuning")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", required=True, help="Training JSONL file")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-targets", default="q_proj,v_proj",
                        help="Comma-separated LoRA target modules")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing (saves VRAM, slower)")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--checkpoint-prefix", default="lora")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed every RNG (torch, cuda, python, transformers). "
                             "Pinned by default so retraining reproduces exactly.")
    args = parser.parse_args()

    seed_everything(args.seed)

    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"LoRA targets: {args.lora_targets}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=True
    ).to(args.device)

    # Apply LoRA
    target_modules = [m.strip() for m in args.lora_targets.split(",")]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Gradient checkpointing saves VRAM at the cost of ~20% slower training
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        print("Gradient checkpointing: enabled")

    # Load data. The shuffle generator is explicitly seeded so example
    # order at every epoch is reproducible from the same --seed.
    dataset = JsonExtractionDataset(args.data, tokenizer, args.max_seq_len)
    shuffle_gen = torch.Generator()
    shuffle_gen.manual_seed(args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=shuffle_gen,
        worker_init_fn=_seeded_worker_init_fn,
    )
    print(f"Training examples: {len(dataset)}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    ckpt_dir = Path(args.checkpoint_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, dataloader, optimizer, args.device, epoch)

        # Save checkpoint every epoch
        ckpt_path = ckpt_dir / f"{args.checkpoint_prefix}_epoch{epoch}"
        model.save_pretrained(str(ckpt_path))
        print(f"  Saved {ckpt_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
