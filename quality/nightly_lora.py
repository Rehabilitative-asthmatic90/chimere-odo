#!/usr/bin/env python3
"""nightly_lora.py — Overnight LoRA training on think_router response pairs

Reads training pairs from think_router logs, deduplicates, filters for quality,
converts to ShareGPT format, and trains a LoRA adapter using unsloth on
Qwen3.5-35B-A3B.

Usage:
    python3 nightly_lora.py
    python3 nightly_lora.py --dry-run
    python3 nightly_lora.py --min-pairs 50 --epochs 3 --lr 1e-4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_INPUT = Path.home() / ".chimere" / "logs" / "training_pairs.jsonl"
QUALITY_SCORES = Path.home() / ".chimere" / "logs" / "quality_scores.jsonl"
DEFAULT_OUTPUT = Path.home() / ".chimere" / "lora" / "latest"
MIN_RESPONSE_LEN = 50
MIN_REASONING_LEN = 100
MIN_QUALITY_SCORE = 4  # Only train on responses scored >= 4 by quality gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overnight LoRA training on think_router response pairs",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to training_pairs.jsonl (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"LoRA adapter output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=20,
        help="Minimum number of valid pairs to proceed with training (default: 20)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs (default: 2)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size (default: 4)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="Maximum sequence length (default: 4096)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only convert and validate data, do not train",
    )
    return parser.parse_args()


def load_pairs(input_path: Path) -> list[dict[str, Any]]:
    """Load training pairs from JSONL file."""
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    pairs: list[dict[str, Any]] = []
    errors = 0
    with input_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                pairs.append(entry)
            except json.JSONDecodeError as e:
                errors += 1
                if errors <= 5:
                    print(f"  WARNING: Skipping malformed JSON at line {lineno}: {e}")

    if errors > 5:
        print(f"  WARNING: {errors} total malformed lines skipped")

    print(f"  Loaded {len(pairs)} raw entries from {input_path}")
    return pairs


def deduplicate(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate entries by SHA-256 hash of prompt text.

    When duplicates are found, the latest entry (by position) is kept.
    """
    seen: dict[str, int] = {}
    for idx, entry in enumerate(pairs):
        prompt = entry.get("prompt", "")
        h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        seen[h] = idx  # last occurrence wins

    deduped = [pairs[i] for i in sorted(seen.values())]
    removed = len(pairs) - len(deduped)
    if removed:
        print(f"  Deduplicated: removed {removed} duplicate prompts")
    print(f"  Unique entries: {len(deduped)}")
    return deduped


def _load_quality_scores() -> dict[str, int]:
    """Load quality scores indexed by prompt_hash for cross-referencing.

    Returns dict mapping prompt_hash → score.
    Both training_pairs.jsonl and quality_scores.jsonl now include prompt_hash.
    Falls back to prompt_len matching for old entries without hash.
    """
    scores: dict[str, int] = {}
    if not QUALITY_SCORES.exists():
        return scores
    try:
        with QUALITY_SCORES.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    # Primary key: prompt_hash (16 chars SHA256 prefix)
                    ph = entry.get("prompt_hash")
                    if ph:
                        scores[ph] = entry.get("score", 3)
                    else:
                        # Fallback: route:prompt_len (old format)
                        key = f"{entry.get('route', '')}:{entry.get('prompt_len', 0)}"
                        scores[key] = entry.get("score", 3)
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return scores


def filter_quality(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter entries by length AND quality gate scores.

    Cross-references with quality_scores.jsonl when available.
    Entries without a quality score pass through (backward compat).
    Entries with score < MIN_QUALITY_SCORE are excluded.
    """
    quality_scores = _load_quality_scores()
    has_scores = len(quality_scores) > 0

    filtered: list[dict[str, Any]] = []
    skipped_response = 0
    skipped_reasoning = 0
    skipped_quality = 0
    boosted = 0

    for entry in pairs:
        response = entry.get("response", "")
        reasoning = entry.get("reasoning", "")
        prompt = entry.get("prompt", "")

        if len(response) < MIN_RESPONSE_LEN:
            skipped_response += 1
            continue
        if len(reasoning) < MIN_REASONING_LEN:
            skipped_reasoning += 1
            continue

        # Cross-reference quality score if available
        if has_scores:
            # Primary: match by prompt_hash
            prompt_hash = entry.get("prompt_hash")
            matched_score = quality_scores.get(prompt_hash) if prompt_hash else None

            # Fallback: match by prompt_len (old entries without hash)
            if matched_score is None:
                prompt_len = len(prompt)
                for key, score in quality_scores.items():
                    if ":" in key:
                        _, plen_str = key.rsplit(":", 1)
                        try:
                            if abs(int(plen_str) - prompt_len) <= 5:
                                matched_score = score
                                break
                        except ValueError:
                            continue

            if matched_score is not None:
                if matched_score < MIN_QUALITY_SCORE:
                    skipped_quality += 1
                    continue
                if matched_score >= 4:
                    boosted += 1

        filtered.append(entry)

    if skipped_response:
        print(f"  Filtered out {skipped_response} entries with response < {MIN_RESPONSE_LEN} chars")
    if skipped_reasoning:
        print(f"  Filtered out {skipped_reasoning} entries with reasoning < {MIN_REASONING_LEN} chars")
    if skipped_quality:
        print(f"  Filtered out {skipped_quality} entries with quality score < {MIN_QUALITY_SCORE}")
    if boosted:
        print(f"  Quality-verified entries (score >= 4): {boosted}")
    if has_scores:
        print(f"  Quality scores loaded: {len(quality_scores)} entries")
    print(f"  Valid entries after filtering: {len(filtered)}")
    return filtered


def to_sharegpt(pairs: list[dict[str, Any]]) -> list[dict[str, list[dict[str, str]]]]:
    """Convert training pairs to ShareGPT conversation format.

    Output format per entry:
    {
        "conversations": [
            {"from": "human", "value": "prompt_text"},
            {"from": "gpt", "value": "<think>\\nreasoning\\n</think>\\nresponse"}
        ]
    }
    """
    dataset: list[dict[str, list[dict[str, str]]]] = []
    for entry in pairs:
        prompt = entry["prompt"]
        reasoning = entry["reasoning"]
        response = entry["response"]

        gpt_value = f"<think>\n{reasoning}\n</think>\n{response}"

        dataset.append({
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": gpt_value},
            ]
        })
    return dataset


def split_dataset(
    dataset: list[dict],
    eval_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split dataset into train (90%) and eval (10%) sets."""
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    eval_size = max(1, int(len(dataset) * eval_ratio))
    eval_indices = set(indices[:eval_size])

    train = [dataset[i] for i in range(len(dataset)) if i not in eval_indices]
    eval_set = [dataset[i] for i in range(len(dataset)) if i in eval_indices]

    print(f"  Train split: {len(train)} samples")
    print(f"  Eval split:  {len(eval_set)} samples")
    return train, eval_set


def check_unsloth() -> bool:
    """Check if unsloth is importable."""
    try:
        import unsloth  # noqa: F401
        return True
    except ImportError:
        return False


def train_lora(
    train_data: list[dict],
    eval_data: list[dict],
    args: argparse.Namespace,
) -> None:
    """Train a LoRA adapter using unsloth on Qwen3.5-35B-A3B."""
    if not check_unsloth():
        print(
            "\nERROR: unsloth is not installed.\n"
            "Install it with:\n"
            "  pip install unsloth\n"
            "Or follow the official guide:\n"
            "  https://github.com/unslothai/unsloth#installation\n"
        )
        sys.exit(1)

    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    # Use local BF16 if available (avoids re-downloading 67 GB)
    local_bf16 = Path.home() / ".chimere" / "models" / "Qwen3.5-35B-A3B-BF16"
    if local_bf16.exists() and (local_bf16 / "config.json").exists():
        model_name = str(local_bf16)
        print(f"  Using local BF16: {model_name}")
    else:
        model_name = "Qwen/Qwen3.5-35B-A3B"

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/4] Loading model: {model_name}")
    print(f"  Max sequence length: {args.max_seq_len}")
    t0 = time.monotonic()

    # Qwen3.5-35B-A3B MoE: 35B total params, 3.5B active.
    # In 4-bit: ~17.5 GB total, ~14 GB with gradient checkpointing.
    # RTX 5060 Ti: 16 GB VRAM → tight but possible with Unsloth optimizations.
    # Let Unsloth handle device placement (no manual device_map).
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_seq_len,
        dtype=None,  # auto-detect
        load_in_4bit=True,
    )

    print(f"  Model loaded in {time.monotonic() - t0:.1f}s")

    print("\n[2/4] Configuring LoRA adapter")
    # Only target attention layers (not MoE experts) to reduce VRAM
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,  # Reduced rank for MoE (less VRAM, still effective)
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    print("\n[3/4] Starting training")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output dir: {output_dir}")

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    def formatting_func(examples: dict) -> list[str]:
        """Format ShareGPT conversations into chat template strings."""
        texts = []
        for convos in examples["conversations"]:
            messages = []
            for turn in convos:
                role = "user" if turn["from"] == "human" else "assistant"
                messages.append({"role": role, "content": turn["value"]})
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return texts

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=False,
        bf16=True,
        gradient_accumulation_steps=max(1, 16 // args.batch_size),
        optim="adamw_8bit",
        seed=42,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_func,
        max_seq_length=args.max_seq_len,
        args=training_args,
    )

    t0 = time.monotonic()
    train_result = trainer.train()
    train_time = time.monotonic() - t0

    print(f"\n  Training completed in {train_time:.1f}s")
    print(f"  Train loss: {train_result.training_loss:.4f}")

    print("\n[4/4] Evaluating on held-out split")
    eval_result = trainer.evaluate()
    print(f"  Eval loss: {eval_result['eval_loss']:.4f}")

    print(f"\n  Saving LoRA adapter to {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Write training metadata
    metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "base_model": model_name,
        "train_samples": len(train_data),
        "eval_samples": len(eval_data),
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "max_seq_len": args.max_seq_len,
        "train_loss": train_result.training_loss,
        "eval_loss": eval_result["eval_loss"],
        "train_time_seconds": round(train_time, 1),
        "input_file": str(args.input),
    }
    metadata_path = output_dir / "training_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Metadata saved to {metadata_path}")

    print("\nDone. LoRA adapter ready at:", output_dir)


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("nightly_lora.py — LoRA training pipeline")
    print("=" * 60)

    # Step 1: Load
    print("\n[Step 1] Loading training pairs...")
    pairs = load_pairs(args.input)
    if not pairs:
        print("ERROR: No training pairs found. Exiting.")
        sys.exit(1)

    # Step 2: Deduplicate
    print("\n[Step 2] Deduplicating by prompt hash...")
    pairs = deduplicate(pairs)

    # Step 3: Filter
    print("\n[Step 3] Filtering for quality...")
    pairs = filter_quality(pairs)

    # Step 4: Check minimum threshold
    if len(pairs) < args.min_pairs:
        print(
            f"\nERROR: Only {len(pairs)} valid pairs found, "
            f"but --min-pairs requires {args.min_pairs}. "
            f"Skipping training."
        )
        sys.exit(1)

    # Step 5: Convert to ShareGPT
    print("\n[Step 4] Converting to ShareGPT format...")
    dataset = to_sharegpt(pairs)
    print(f"  Converted {len(dataset)} entries to ShareGPT format")

    # Step 6: Split
    print("\n[Step 5] Splitting train/eval (90/10)...")
    train_data, eval_data = split_dataset(dataset)

    # Step 7: Dry-run or train
    if args.dry_run:
        print("\n[DRY RUN] Validation complete. Skipping training.")
        print(f"  Would train on {len(train_data)} samples, evaluate on {len(eval_data)} samples")

        # Save converted dataset for inspection
        dry_run_path = args.output_dir / "dry_run_preview.jsonl"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with dry_run_path.open("w", encoding="utf-8") as f:
            for entry in dataset[:10]:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"  Preview (first 10 entries) saved to {dry_run_path}")

        # Show a sample
        if dataset:
            sample = dataset[0]
            human_msg = sample["conversations"][0]["value"]
            gpt_msg = sample["conversations"][1]["value"]
            print(f"\n  Sample entry:")
            print(f"    Human: {human_msg[:120]}{'...' if len(human_msg) > 120 else ''}")
            print(f"    GPT:   {gpt_msg[:120]}{'...' if len(gpt_msg) > 120 else ''}")
        return

    print("\n[Step 6] Training LoRA adapter...")
    train_lora(train_data, eval_data, args)


if __name__ == "__main__":
    main()
