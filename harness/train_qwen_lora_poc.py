#!/usr/bin/env python3
"""Tiny QLoRA overfit loop for STS2 SFT chat examples."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


Json = dict[str, Any]


def parse_max_memory(value: str | None) -> dict[Any, str] | None:
    if not value:
        return None
    result: dict[Any, str] = {}
    for item in value.split(","):
        key, mem = item.split(":", 1)
        result[int(key) if key.isdigit() else key] = mem
    return result


def load_examples(path: Path, limit: int | None = None) -> list[Json]:
    examples: list[Json] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            messages = record.get("messages")
            if not isinstance(messages, list) or len(messages) < 3:
                raise ValueError(f"{path}:{line_number} is missing chat messages")
            examples.append(record)
            if limit is not None and len(examples) >= limit:
                break
    if not examples:
        raise ValueError(f"{path} did not contain any examples")
    return examples


def quantile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def format_texts(processor: Any, messages: list[Json]) -> tuple[str, str]:
    full_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        preserve_thinking=True,
    )
    prompt_text = processor.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return str(full_text), str(prompt_text)


def tokenize_examples(
    examples: list[Json],
    processor: Any,
    tokenizer: Any,
    *,
    max_seq_length: int,
    too_long: str,
) -> list[Json]:
    tokenized: list[Json] = []
    skipped = 0
    lengths: list[int] = []
    prompt_lengths: list[int] = []

    for index, example in enumerate(examples, start=1):
        messages = example["messages"]
        full_text, prompt_text = format_texts(processor, messages)
        input_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        if not isinstance(input_ids, list) or not isinstance(prompt_ids, list):
            raise TypeError("tokenizer returned unexpected token structure")

        labels = list(input_ids)
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        if len(input_ids) > max_seq_length:
            if too_long == "error":
                raise ValueError(f"example {index} has {len(input_ids)} tokens, above max_seq_length={max_seq_length}")
            if too_long == "skip":
                skipped += 1
                continue
            overflow = len(input_ids) - max_seq_length
            input_ids = input_ids[overflow:]
            labels = labels[overflow:]

        if all(label == -100 for label in labels):
            skipped += 1
            continue

        lengths.append(len(input_ids))
        prompt_lengths.append(prompt_len)
        tokenized.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
                "metadata": example.get("metadata", {}),
            }
        )

    if not tokenized:
        raise ValueError(f"all examples were skipped; skipped={skipped}")

    print(
        json.dumps(
            {
                "tokenized_examples": len(tokenized),
                "skipped_examples": skipped,
                "tokens": {
                    "min": min(lengths),
                    "p50": quantile(lengths, 0.50),
                    "p90": quantile(lengths, 0.90),
                    "max": max(lengths),
                },
                "prompt_tokens": {
                    "min": min(prompt_lengths),
                    "p50": quantile(prompt_lengths, 0.50),
                    "p90": quantile(prompt_lengths, 0.90),
                    "max": max(prompt_lengths),
                },
            },
            indent=2,
        ),
        flush=True,
    )
    return tokenized


def collate_batch(batch: list[Json], pad_token_id: int) -> Json:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    attention_mask = []
    labels = []
    for item in batch:
        pad = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad)
        attention_mask.append(item["attention_mask"] + [0] * pad)
        labels.append(item["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def trainable_parameter_summary(model: torch.nn.Module) -> Json:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_percent": round(100 * trainable / total, 4) if total else 0,
    }


def move_batch(batch: Json, device: torch.device | str) -> Json:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit Qwen3.6-27B on STS2 SFT examples with QLoRA.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-dir", default="/workspace/models/Qwen3.6-27B")
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/qwen36-sts2-lora-poc"))
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--too-long", choices=["skip", "error", "keep_end"], default="skip")
    parser.add_argument("--limit-examples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target module names.",
    )
    parser.add_argument("--max-memory", default=None, help='Optional accelerate max_memory, e.g. "0:42GiB,cpu:120GiB"')
    parser.add_argument("--dry-run-tokenize", action="store_true", help="Only render/tokenize examples; do not load the model.")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "sts2mcp-qwen-lora"))
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Loading processor from {args.model_dir}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")

    examples = load_examples(args.dataset, args.limit_examples)
    tokenized = tokenize_examples(
        examples,
        processor,
        tokenizer,
        max_seq_length=args.max_seq_length,
        too_long=args.too_long,
    )
    if args.dry_run_tokenize:
        return 0

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model_kwargs: Json = {
        "local_files_only": True,
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "quantization_config": quant_config,
    }
    max_memory = parse_max_memory(args.max_memory)
    if max_memory:
        model_kwargs["max_memory"] = max_memory

    print(f"Loading model from {args.model_dir}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, **model_kwargs)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.train()
    summary = trainable_parameter_summary(model)
    print(json.dumps(summary, indent=2), flush=True)

    run = None
    if not args.no_wandb:
        import wandb

        run_name = args.wandb_run_name or f"qwen36-lora-overfit-{time.strftime('%Y%m%d-%H%M%S')}"
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                **config,
                "model_dir": str(args.model_dir),
                "dataset": str(args.dataset),
                **summary,
            },
        )

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    dataloader = DataLoader(
        tokenized,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, pad_token_id),
    )
    batches = itertools.cycle(dataloader)
    input_device = getattr(model, "device", None) or next(model.parameters()).device

    optimizer.zero_grad(set_to_none=True)
    recent_losses: list[float] = []
    started = time.time()
    for step in range(1, args.max_steps + 1):
        step_loss = 0.0
        for _ in range(args.gradient_accumulation_steps):
            batch = move_batch(next(batches), input_device)
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            step_loss += float(loss.detach().cpu())
        torch.nn.utils.clip_grad_norm_((parameter for parameter in model.parameters() if parameter.requires_grad), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        recent_losses.append(step_loss)
        log_data: Json = {
            "train/loss": step_loss,
            "train/step": step,
            "train/examples": len(tokenized),
            "train/elapsed_sec": round(time.time() - started, 2),
        }
        if torch.cuda.is_available():
            log_data["gpu/max_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        if run is not None:
            run.log(log_data, step=step)
        window = recent_losses[-5:]
        print(f"step {step:04d} loss={step_loss:.4f} avg5={sum(window)/len(window):.4f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving LoRA adapter to {args.output_dir}", flush=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    if run is not None:
        run.summary["final_loss"] = recent_losses[-1]
        run.summary["initial_loss"] = recent_losses[0]
        run.summary["loss_delta"] = recent_losses[-1] - recent_losses[0]
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
