#!/usr/bin/env python3
"""spin_training.py — SPIN data generation for DPO training

Implements the data-generation phase of SPIN (Self-Play Fine-Tuning,
arXiv 2401.01335). The SPIN algorithm:

  1. For each prompt in training_pairs.jsonl, generate a response using
     the CURRENT model via chimere-server (port 8081, inference only)
  2. Pair: teacher_response (quality-gated, score >= 4) = chosen,
           model_response = rejected
  3. Save DPO-ready pairs to ~/.chimere/data/spin_pairs.jsonl

The actual DPO training is deferred to cloud GPU (same OOM constraint as
LoRA). This script handles the expensive LOCAL generation part.

Reads from:
  - ~/.chimere/logs/training_pairs.jsonl (ODO training pairs)
  - ~/.chimere/logs/quality_scores.jsonl (quality gate scores)
  - Optionally: ~/.chimere/data/dspy_datasets/*_opus_gold.jsonl (gold data)

Outputs:
  - ~/.chimere/data/spin_pairs.jsonl (DPO-ready dataset)

Usage:
    python3 spin_training.py                    # generate SPIN pairs
    python3 spin_training.py --dry-run          # preview without calling model
    python3 spin_training.py --include-gold      # include DSPy gold datasets
    python3 spin_training.py --max-prompts 50   # limit number of prompts
    python3 spin_training.py --iteration 2      # tag iteration number
    python3 spin_training.py --resume            # skip prompts already in output
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

# ── Paths ────────────────────────────────────────────────────────────────────

TRAINING_PAIRS = Path.home() / ".chimere" / "logs" / "training_pairs.jsonl"
QUALITY_SCORES = Path.home() / ".chimere" / "logs" / "quality_scores.jsonl"
GOLD_DIR = Path.home() / ".chimere" / "data" / "dspy_datasets"
OUTPUT_PATH = Path.home() / ".chimere" / "data" / "spin_pairs.jsonl"

# ── Inference config ─────────────────────────────────────────────────────────

CHIMERE_URL = os.environ.get("SPIN_SERVER_URL", "http://127.0.0.1:8081")
MAX_TOKENS = 2048
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
REQUEST_TIMEOUT = 180  # seconds, generous for long responses

# ── Quality thresholds ───────────────────────────────────────────────────────

MIN_QUALITY_SCORE = 4      # only use teacher responses scored >= 4
MIN_TEACHER_LEN = 100      # minimum teacher response length (chars)
MIN_MODEL_LEN = 50         # minimum model response length to keep pair
MIN_PROMPT_LEN = 20        # skip trivial prompts
MAX_PROMPT_LEN = 4000      # skip excessively long prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SPIN data generation: create DPO-ready pairs from "
                    "teacher responses + model self-play",
    )
    parser.add_argument(
        "--input", type=Path, default=TRAINING_PAIRS,
        help=f"Training pairs JSONL (default: {TRAINING_PAIRS})",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH,
        help=f"Output SPIN pairs JSONL (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--include-gold", action="store_true",
        help="Also include DSPy gold datasets (Opus-distilled) as teacher data",
    )
    parser.add_argument(
        "--max-prompts", type=int, default=0,
        help="Limit number of prompts to process (0 = all)",
    )
    parser.add_argument(
        "--iteration", type=int, default=1,
        help="SPIN iteration number (1-4, for metadata tagging)",
    )
    parser.add_argument(
        "--server-url", type=str, default=CHIMERE_URL,
        help=f"Inference server URL (default: {CHIMERE_URL})",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=MAX_TOKENS,
        help=f"Max generation tokens (default: {MAX_TOKENS})",
    )
    parser.add_argument(
        "--temperature", type=float, default=TEMPERATURE,
        help=f"Sampling temperature (default: {TEMPERATURE})",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip prompts already present in the output file",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview prompts and teacher data without calling the model",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print full teacher/model responses",
    )
    return parser.parse_args()


# ── Quality scores ───────────────────────────────────────────────────────────

def load_quality_scores() -> dict[str, int]:
    """Load quality gate scores indexed by prompt_hash."""
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
                    ph = entry.get("prompt_hash")
                    if ph:
                        scores[ph] = entry.get("score", 3)
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return scores


# ── Load teacher data ────────────────────────────────────────────────────────

def load_training_pairs(
    input_path: Path,
    quality_scores: dict[str, int],
) -> list[dict]:
    """Load and filter training pairs for teacher responses.

    Only keeps entries where:
    - Response is non-trivial (>= MIN_TEACHER_LEN chars)
    - Prompt is not trivial (<= MIN_PROMPT_LEN) or too long (>= MAX_PROMPT_LEN)
    - Response doesn't contain tool_call (we want completed answers)
    - Quality score >= MIN_QUALITY_SCORE (when score is available)
    """
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        return []

    pairs: list[dict] = []
    skipped_short_prompt = 0
    skipped_long_prompt = 0
    skipped_short_response = 0
    skipped_tool_call = 0
    skipped_quality = 0
    skipped_json = 0

    with input_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                skipped_json += 1
                continue

            prompt = entry.get("prompt", "")
            response = entry.get("response", "")

            # Skip trivial prompts
            if len(prompt) < MIN_PROMPT_LEN:
                skipped_short_prompt += 1
                continue

            # Skip excessively long prompts (won't fit in generation context)
            if len(prompt) > MAX_PROMPT_LEN:
                skipped_long_prompt += 1
                continue

            # Skip short responses (not useful as teacher signal)
            if len(response) < MIN_TEACHER_LEN:
                skipped_short_response += 1
                continue

            # Skip tool_call responses (incomplete answers)
            if "<tool_call>" in response or "function_call" in response:
                skipped_tool_call += 1
                continue

            # Quality gate check
            prompt_hash = entry.get("prompt_hash")
            if not prompt_hash:
                prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

            if quality_scores:
                score = quality_scores.get(prompt_hash)
                if score is not None and score < MIN_QUALITY_SCORE:
                    skipped_quality += 1
                    continue

            pairs.append({
                "prompt": prompt,
                "teacher_response": response,
                "teacher_reasoning": entry.get("reasoning", ""),
                "prompt_hash": prompt_hash,
                "source": "training_pairs",
            })

    print(f"  Loaded {len(pairs)} valid teacher pairs from {input_path}")
    if skipped_json:
        print(f"    Skipped {skipped_json} malformed JSON lines")
    if skipped_short_prompt:
        print(f"    Skipped {skipped_short_prompt} trivial prompts (< {MIN_PROMPT_LEN} chars)")
    if skipped_long_prompt:
        print(f"    Skipped {skipped_long_prompt} overlong prompts (> {MAX_PROMPT_LEN} chars)")
    if skipped_short_response:
        print(f"    Skipped {skipped_short_response} short responses (< {MIN_TEACHER_LEN} chars)")
    if skipped_tool_call:
        print(f"    Skipped {skipped_tool_call} tool_call responses")
    if skipped_quality:
        print(f"    Skipped {skipped_quality} low quality (score < {MIN_QUALITY_SCORE})")

    return pairs


def load_gold_datasets() -> list[dict]:
    """Load DSPy gold datasets (Opus-distilled) as additional teacher data."""
    pairs: list[dict] = []
    if not GOLD_DIR.exists():
        return pairs

    for gold_file in sorted(GOLD_DIR.glob("*_opus_gold.jsonl")):
        count = 0
        with gold_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                question = entry.get("question", "")
                answer = entry.get("answer", "")
                if len(question) < MIN_PROMPT_LEN or len(answer) < MIN_TEACHER_LEN:
                    continue

                prompt_hash = hashlib.sha256(question.encode()).hexdigest()[:16]
                pairs.append({
                    "prompt": question,
                    "teacher_response": answer,
                    "teacher_reasoning": "",  # Gold datasets don't have reasoning
                    "prompt_hash": prompt_hash,
                    "source": f"gold:{gold_file.stem}",
                })
                count += 1

        if count:
            print(f"  Loaded {count} gold entries from {gold_file.name}")

    return pairs


def deduplicate_by_prompt(pairs: list[dict]) -> list[dict]:
    """Deduplicate by prompt hash, keeping the latest (last) entry."""
    seen: dict[str, int] = {}
    for idx, entry in enumerate(pairs):
        seen[entry["prompt_hash"]] = idx
    deduped = [pairs[i] for i in sorted(seen.values())]
    removed = len(pairs) - len(deduped)
    if removed:
        print(f"  Deduplicated: removed {removed} duplicates")
    return deduped


def load_existing_hashes(output_path: Path) -> set[str]:
    """Load prompt hashes from existing output for --resume."""
    hashes: set[str] = set()
    if not output_path.exists():
        return hashes
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                ph = entry.get("prompt_hash", "")
                if ph:
                    hashes.add(ph)
            except json.JSONDecodeError:
                continue
    return hashes


# ── Model generation ─────────────────────────────────────────────────────────

def generate_model_response(
    prompt: str,
    server_url: str,
    max_tokens: int,
    temperature: float,
) -> tuple[str, float]:
    """Generate a response from the current model via inference server.

    Returns (response_text, elapsed_seconds).
    """
    parsed = urlparse(server_url)
    body = json.dumps({
        "model": "qwen3.5",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": TOP_P,
        "top_k": TOP_K,
        # No-think for SPIN generation: we want the raw model output,
        # not the thinking-enhanced version. This tests the model's
        # base capability which is what DPO will improve.
        "chat_template_kwargs": {"enable_thinking": False},
    })

    t0 = time.time()
    conn = http.client.HTTPConnection(
        parsed.hostname, parsed.port, timeout=REQUEST_TIMEOUT,
    )
    conn.request("POST", "/v1/chat/completions", body=body, headers={
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
    })
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    elapsed = time.time() - t0

    # Extract content
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError(f"No choices in response: {data.get('error', data)}")

    content = choices[0].get("message", {}).get("content", "")
    return content, elapsed


def check_server(server_url: str) -> bool:
    """Quick health check on the inference server."""
    parsed = urlparse(server_url)
    try:
        conn = http.client.HTTPConnection(
            parsed.hostname, parsed.port, timeout=5,
        )
        conn.request("GET", "/health")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        return resp.status == 200
    except Exception:
        return False


# ── SPIN pair generation ─────────────────────────────────────────────────────

def generate_spin_pairs(
    pairs: list[dict],
    args: argparse.Namespace,
) -> list[dict]:
    """Generate SPIN pairs by running the model on each teacher prompt.

    For each prompt:
    - Teacher response = chosen (from training data / gold)
    - Model response = rejected (generated by current model)
    """
    server_url = args.server_url
    max_tokens = args.max_tokens
    temperature = args.temperature
    iteration = args.iteration
    output_path = args.output
    verbose = args.verbose

    # Resume support
    existing_hashes: set[str] = set()
    if args.resume:
        existing_hashes = load_existing_hashes(output_path)
        if existing_hashes:
            print(f"  Resume mode: {len(existing_hashes)} existing pairs, "
                  f"will skip matching prompts")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    spin_pairs: list[dict] = []
    skipped_resume = 0
    skipped_short_model = 0
    skipped_error = 0
    total = len(pairs)

    for idx, pair in enumerate(pairs):
        prompt = pair["prompt"]
        prompt_hash = pair["prompt_hash"]

        # Skip if already generated (resume mode)
        if prompt_hash in existing_hashes:
            skipped_resume += 1
            continue

        progress = f"[{idx + 1}/{total}]"
        prompt_preview = prompt[:80].replace("\n", " ")
        print(f"  {progress} {prompt_preview}...", end=" ", flush=True)

        if args.dry_run:
            print("SKIP (dry-run)")
            continue

        # Generate model response
        try:
            model_response, elapsed = generate_model_response(
                prompt, server_url, max_tokens, temperature,
            )
        except Exception as e:
            print(f"ERROR: {e}")
            skipped_error += 1
            # Brief pause to avoid hammering a struggling server
            time.sleep(2)
            continue

        # Validate model response
        if len(model_response) < MIN_MODEL_LEN:
            print(f"SHORT ({len(model_response)} chars, skipped)")
            skipped_short_model += 1
            continue

        # Build DPO pair
        teacher = pair["teacher_response"]
        # Include reasoning in teacher response if available (richer signal)
        if pair.get("teacher_reasoning"):
            teacher_full = f"<think>\n{pair['teacher_reasoning']}\n</think>\n{teacher}"
        else:
            teacher_full = teacher

        spin_entry = {
            "prompt": prompt,
            "chosen": teacher_full,
            "rejected": model_response,
            "prompt_hash": prompt_hash,
            "iteration": iteration,
            "source": pair.get("source", "training_pairs"),
            "teacher_len": len(teacher),
            "model_len": len(model_response),
            "gen_time_s": round(elapsed, 2),
            "ts": datetime.now().isoformat(),
        }

        spin_pairs.append(spin_entry)

        # Append to output file immediately (crash-safe)
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(spin_entry, ensure_ascii=False) + "\n")

        tok_per_s = len(model_response.split()) / elapsed if elapsed > 0 else 0
        print(f"OK ({len(model_response)} chars, {elapsed:.1f}s, ~{tok_per_s:.0f} w/s)")

        if verbose:
            print(f"    TEACHER: {teacher[:200]}...")
            print(f"    MODEL:   {model_response[:200]}...")

    # Summary
    print(f"\n{'='*60}")
    print(f"SPIN Generation Summary (iteration {iteration})")
    print(f"{'='*60}")
    print(f"  Total prompts:       {total}")
    print(f"  Generated pairs:     {len(spin_pairs)}")
    if skipped_resume:
        print(f"  Skipped (resume):    {skipped_resume}")
    if skipped_short_model:
        print(f"  Skipped (short):     {skipped_short_model}")
    if skipped_error:
        print(f"  Skipped (errors):    {skipped_error}")
    if not args.dry_run:
        print(f"  Output: {output_path}")

    return spin_pairs


# ── Stats ────────────────────────────────────────────────────────────────────

def print_dataset_stats(output_path: Path):
    """Print stats about the accumulated SPIN dataset."""
    if not output_path.exists():
        return

    entries = []
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not entries:
        return

    iterations = set(e.get("iteration", 1) for e in entries)
    sources = {}
    for e in entries:
        s = e.get("source", "unknown")
        sources[s] = sources.get(s, 0) + 1

    avg_teacher = sum(e.get("teacher_len", 0) for e in entries) / len(entries)
    avg_model = sum(e.get("model_len", 0) for e in entries) / len(entries)
    avg_gen = sum(e.get("gen_time_s", 0) for e in entries) / len(entries)

    print(f"\nDataset stats ({output_path}):")
    print(f"  Total pairs:         {len(entries)}")
    print(f"  Iterations:          {sorted(iterations)}")
    print(f"  Avg teacher length:  {avg_teacher:.0f} chars")
    print(f"  Avg model length:    {avg_model:.0f} chars")
    print(f"  Avg gen time:        {avg_gen:.1f}s")
    print(f"  Sources:")
    for s, c in sorted(sources.items()):
        print(f"    {s}: {c}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"SPIN Data Generator v1.0")
    print(f"  Server:    {args.server_url}")
    print(f"  Iteration: {args.iteration}")
    print(f"  Output:    {args.output}")
    print(f"{'='*60}")

    # Check server health (unless dry-run)
    if not args.dry_run:
        print("  Checking inference server...", end=" ", flush=True)
        if not check_server(args.server_url):
            print("FAILED")
            print(f"ERROR: Server at {args.server_url} is not responding.",
                  file=sys.stderr)
            print("  Start qwen35-custom.service or set --server-url",
                  file=sys.stderr)
            sys.exit(1)
        print("OK")

    # Load quality scores for filtering
    quality_scores = load_quality_scores()
    if quality_scores:
        print(f"  Quality scores loaded: {len(quality_scores)} entries")

    # Load teacher data
    print(f"\nLoading teacher data...")
    pairs = load_training_pairs(args.input, quality_scores)

    if args.include_gold:
        print(f"\nLoading gold datasets...")
        gold_pairs = load_gold_datasets()
        pairs.extend(gold_pairs)

    if not pairs:
        print("ERROR: No valid teacher data found.", file=sys.stderr)
        sys.exit(1)

    # Deduplicate
    pairs = deduplicate_by_prompt(pairs)
    print(f"  Total unique teacher prompts: {len(pairs)}")

    # Limit if requested
    if args.max_prompts > 0 and len(pairs) > args.max_prompts:
        pairs = pairs[:args.max_prompts]
        print(f"  Limited to {args.max_prompts} prompts")

    # Generate SPIN pairs
    print(f"\nGenerating model responses...")
    spin_pairs = generate_spin_pairs(pairs, args)

    # Print accumulated stats
    print_dataset_stats(args.output)

    # Hint about next steps
    if spin_pairs and not args.dry_run:
        print(f"\nNext steps:")
        print(f"  1. Transfer {args.output} to cloud GPU")
        print(f"  2. Run DPO training with trl DPOTrainer")
        print(f"  3. Merge LoRA adapter and convert back to GGUF")
        print(f"  4. Increment --iteration and re-run for SPIN iteration {args.iteration + 1}")


if __name__ == "__main__":
    main()
