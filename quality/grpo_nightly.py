#!/usr/bin/env python3
"""grpo_nightly.py — GRPO (Group Relative Policy Optimization) training pipeline

Replaces SFT nightly with RL-based training using verifiable rewards.
For each prompt, generates K candidates, scores them with domain-specific
reward functions, and updates the LoRA adapter via GRPO.

Pipeline:
  1. Load prompts from training_pairs.jsonl (reuse existing data)
  2. For each prompt, generate K=4 candidate responses via chimere-server
  3. Score each candidate with the appropriate reward function
  4. Train LoRA with GRPO (TRL's GRPOTrainer)

Usage:
    python3 grpo_nightly.py --dry-run
    python3 grpo_nightly.py --candidates 4 --epochs 1

Requires: unsloth, trl >= 0.15, grpo_rewards.py in same directory
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

# --- Paths ---
TRAINING_PAIRS = Path.home() / ".chimere" / "logs" / "training_pairs.jsonl"
QUALITY_SCORES = Path.home() / ".chimere" / "logs" / "quality_scores.jsonl"
DEFAULT_OUTPUT = Path.home() / ".chimere" / "lora" / "grpo-latest"
CHIMERE_URL = "http://127.0.0.1:8081"
MIN_PROMPTS = 10
MIN_QUALITY_SCORE = 3  # Lower bar than SFT: GRPO learns from bad examples too

# --- Domain classification (mirrors ODO routes) ---
DOMAIN_KEYWORDS = {
    "code": ["python", "rust", "javascript", "function", "class", "import",
             "def ", "fn ", "code", "script", "algorithm", "API", "debug",
             "compile", "test", "git"],
    "tools": ["tool", "function_call", "json", "schema", "API call",
              "web_search", "get_weather", "calculate"],
    "kine": ["kiné", "rééducation", "tendon", "muscle", "arthrose",
             "ligament", "exercice", "protocole", "HAS", "coiffe",
             "épaule", "genou", "LCA", "lombalgie"],
    "cyber": ["malware", "CVE", "IoC", "Suricata", "YARA", "phishing",
              "ransomware", "CTI", "MITRE", "SIEM"],
}


def classify_domain(prompt: str) -> str:
    """Simple keyword-based domain classification."""
    prompt_lower = prompt.lower()
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw.lower() in prompt_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO nightly training")
    parser.add_argument("--input", type=Path, default=TRAINING_PAIRS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidates", type=int, default=4,
                        help="Number of candidate responses per prompt (K)")
    parser.add_argument("--min-prompts", type=int, default=MIN_PROMPTS)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="GRPO learning rate (lower than SFT)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Use existing responses instead of generating new ones")
    return parser.parse_args()


def load_prompts(input_path: Path) -> list[dict[str, Any]]:
    """Load prompts from training_pairs.jsonl, deduplicate."""
    if not input_path.exists():
        print(f"ERROR: {input_path} not found")
        sys.exit(1)

    seen_hashes: set[str] = set()
    prompts: list[dict[str, Any]] = []

    with input_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                prompt = entry.get("prompt", "")
                prompt_hash = entry.get("prompt_hash", "")
                if not prompt or len(prompt) < 20:
                    continue
                if prompt_hash in seen_hashes:
                    continue
                seen_hashes.add(prompt_hash)

                domain = classify_domain(prompt)
                prompts.append({
                    "prompt": prompt,
                    "domain": domain,
                    "prompt_hash": prompt_hash,
                    # Keep original response for reference
                    "reference_response": entry.get("response", ""),
                })
            except json.JSONDecodeError:
                continue

    print(f"  Loaded {len(prompts)} unique prompts")
    # Domain distribution
    from collections import Counter
    dist = Counter(p["domain"] for p in prompts)
    for d, c in dist.most_common():
        print(f"    {d}: {c}")
    return prompts


def generate_candidates(
    prompt: str,
    k: int,
    max_tokens: int = 2048,
    temperature: float = 0.8,
) -> list[str]:
    """Generate K candidate responses from chimere-server."""
    import http.client
    import urllib.parse

    parsed = urllib.parse.urlparse(CHIMERE_URL)
    candidates = []

    for i in range(k):
        try:
            conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=120)
            body = json.dumps({
                "model": "qwen3.5",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature + (i * 0.05),  # Slight diversity
                "top_p": 0.95,
            })
            conn.request("POST", "/v1/chat/completions",
                         body=body,
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            data = json.loads(resp.read())
            content = data["choices"][0]["message"]["content"]
            candidates.append(content)
            conn.close()
        except Exception as e:
            print(f"    WARNING: candidate {i+1}/{k} failed: {e}", file=sys.stderr)
            candidates.append("")

    return candidates


def score_candidates(
    prompt: str,
    candidates: list[str],
    domain: str,
) -> list[float]:
    """Score candidates using domain-appropriate reward function."""
    # Import reward functions (same directory)
    sys.path.insert(0, str(Path(__file__).parent))
    from grpo_rewards import combined_reward

    scores = []
    for resp in candidates:
        if not resp:
            scores.append(0.0)
            continue
        score = combined_reward(prompt, resp, domain)
        scores.append(score)
    return scores


def build_grpo_dataset(
    prompts: list[dict],
    candidates_per_prompt: dict[str, list[str]],
    scores_per_prompt: dict[str, list[float]],
) -> list[dict]:
    """Build dataset in GRPO format: prompt + completions + rewards."""
    dataset = []
    for entry in prompts:
        ph = entry["prompt_hash"]
        if ph not in candidates_per_prompt:
            continue
        cands = candidates_per_prompt[ph]
        scores = scores_per_prompt[ph]
        if not cands or not scores:
            continue

        dataset.append({
            "prompt": entry["prompt"],
            "completions": cands,
            "rewards": scores,
            "domain": entry["domain"],
        })
    return dataset


def train_grpo(
    dataset: list[dict],
    args: argparse.Namespace,
) -> None:
    """Train LoRA with GRPO using TRL's GRPOTrainer."""
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    model_name = "Qwen/Qwen3.5-35B-A3B"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/4] Loading model: {model_name}")
    t0 = time.monotonic()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=True,
    )
    print(f"  Loaded in {time.monotonic() - t0:.1f}s")

    print("\n[2/4] Configuring LoRA")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    print("\n[3/4] Preparing GRPO dataset")
    # Convert to HF Dataset format
    # GRPO needs: prompt (str), completions (list[str]), rewards (list[float])
    hf_data = []
    for entry in dataset:
        hf_data.append({
            "prompt": entry["prompt"],
            "completions": entry["completions"],
            "rewards": entry["rewards"],
        })
    train_dataset = Dataset.from_list(hf_data)

    print(f"  {len(hf_data)} prompt groups, {args.candidates} candidates each")

    grpo_config = GRPOConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=True,
        gradient_accumulation_steps=max(1, 8 // args.batch_size),
        optim="adamw_8bit",
        seed=42,
        report_to="none",
        # GRPO-specific
        num_generations=args.candidates,
        max_completion_length=args.max_seq_len,
    )

    print("\n[4/4] Training with GRPO")
    t0 = time.monotonic()

    trainer = GRPOTrainer(
        model=model,
        config=grpo_config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    result = trainer.train()
    train_time = time.monotonic() - t0

    print(f"\n  GRPO training completed in {train_time:.1f}s")
    print(f"  Final loss: {result.training_loss:.4f}")

    print(f"\n  Saving LoRA adapter to {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Metadata
    metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": "GRPO",
        "base_model": model_name,
        "prompts": len(dataset),
        "candidates_per_prompt": args.candidates,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "train_loss": result.training_loss,
        "train_time_seconds": round(train_time, 1),
    }
    meta_path = output_dir / "training_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"  Metadata: {meta_path}")
    print("\nDone. GRPO LoRA adapter ready at:", output_dir)


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("grpo_nightly.py — GRPO training pipeline")
    print("=" * 60)

    # Step 1: Load prompts
    print("\n[Step 1] Loading prompts...")
    prompts = load_prompts(args.input)
    if len(prompts) < args.min_prompts:
        print(f"ERROR: {len(prompts)} prompts < {args.min_prompts} minimum")
        sys.exit(1)

    # Shuffle and limit (avoid very long generation phases)
    random.seed(42)
    random.shuffle(prompts)
    max_prompts = 50  # Cap at 50 prompts per night
    if len(prompts) > max_prompts:
        prompts = prompts[:max_prompts]
        print(f"  Capped to {max_prompts} prompts")

    # Step 2: Generate candidates
    print(f"\n[Step 2] Generating {args.candidates} candidates per prompt...")
    candidates_per_prompt: dict[str, list[str]] = {}
    scores_per_prompt: dict[str, list[float]] = {}

    if args.skip_generation:
        print("  --skip-generation: using reference responses only")
        for entry in prompts:
            ph = entry["prompt_hash"]
            ref = entry.get("reference_response", "")
            candidates_per_prompt[ph] = [ref] if ref else []
            if ref:
                scores_per_prompt[ph] = [0.5]  # Neutral score
    elif args.dry_run:
        print("  [DRY RUN] Skipping candidate generation")
        for entry in prompts:
            ph = entry["prompt_hash"]
            candidates_per_prompt[ph] = ["[dry-run candidate]"] * args.candidates
            scores_per_prompt[ph] = [0.5] * args.candidates
    else:
        for i, entry in enumerate(prompts):
            ph = entry["prompt_hash"]
            prompt = entry["prompt"]
            domain = entry["domain"]
            print(f"  [{i+1}/{len(prompts)}] {domain}: {prompt[:60]}...")

            cands = generate_candidates(prompt, args.candidates)
            candidates_per_prompt[ph] = cands

            scores = score_candidates(prompt, cands, domain)
            scores_per_prompt[ph] = scores

            best_idx = max(range(len(scores)), key=lambda j: scores[j])
            print(f"    scores: {[f'{s:.2f}' for s in scores]} → best={best_idx} ({scores[best_idx]:.2f})")

    # Step 3: Build dataset
    print(f"\n[Step 3] Building GRPO dataset...")
    dataset = build_grpo_dataset(prompts, candidates_per_prompt, scores_per_prompt)
    print(f"  {len(dataset)} prompt groups with candidates + rewards")

    if args.dry_run:
        print("\n[DRY RUN] Validation complete.")
        print(f"  Would train on {len(dataset)} groups × {args.candidates} candidates")

        # Preview
        if dataset:
            entry = dataset[0]
            print(f"\n  Sample prompt: {entry['prompt'][:100]}...")
            print(f"  Domain: {entry['domain']}")
            print(f"  Rewards: {entry['rewards']}")
        return

    # Step 4: Train GRPO
    print(f"\n[Step 4] Training GRPO LoRA...")
    train_grpo(dataset, args)


if __name__ == "__main__":
    main()
