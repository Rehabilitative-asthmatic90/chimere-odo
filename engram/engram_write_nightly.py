#!/usr/bin/env python3
"""
engram_write_nightly.py — Nightly Engram WRITE from quality-validated responses.

Pipeline:
  1. Score unscored responses via 9B scorer (port 8085, batch)
  2. Read quality_scores.jsonl → filter score >= 4
  3. Cross-reference with training_pairs.jsonl via prompt_hash
  4. Extract good responses → text corpus
  5. Run engram_ingest.py → update .engr tables per route
  6. Decay: halve weight of n-grams unused >30d, delete >90d
  7. Conflict resolution: keep highest-scoring n-gram on collision
  8. Archive processed entries

Usage:
    engram_write_nightly.py                    # process all routes
    engram_write_nightly.py --route kine       # process only kine
    engram_write_nightly.py --dry-run          # show what would be ingested
    engram_write_nightly.py --min-score 5      # only perfect responses
    engram_write_nightly.py --skip-scoring     # skip 9B batch scoring
    engram_write_nightly.py --skip-decay       # skip decay pass
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

QUALITY_LOG = Path.home() / ".chimere/logs/quality_scores.jsonl"
TRAINING_LOG = Path.home() / ".chimere/logs/training_pairs.jsonl"
ENGRAM_DIR = Path.home() / ".chimere/data/engram"
ARCHIVE_DIR = Path.home() / ".chimere/logs/archive"
INGEST_SCRIPT = Path.home() / ".chimere/bin/engram_ingest.py"
SCORER_URL = "http://127.0.0.1:8085"

MIN_RESPONSE_LEN = 100
DECAY_HALF_DAYS = 30
DECAY_DELETE_DAYS = 90


def load_quality_scores(min_score: int = 4) -> dict:
    """Load quality scores, filter by min_score. Returns {prompt_hash: entry}."""
    good = {}
    if not QUALITY_LOG.exists():
        return good
    with open(QUALITY_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("score", 0) >= min_score and entry.get("prompt_hash"):
                    good[entry["prompt_hash"]] = entry
            except json.JSONDecodeError:
                continue
    return good


def load_training_pairs() -> dict:
    """Load training pairs. Returns {prompt_hash: entry}."""
    pairs = {}
    if not TRAINING_LOG.exists():
        return pairs
    with open(TRAINING_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                ph = entry.get("prompt_hash")
                if ph:
                    pairs[ph] = entry
            except json.JSONDecodeError:
                continue
    return pairs


def extract_good_responses(min_score: int = 4, route_filter: str = None) -> dict:
    """Cross-reference quality scores with training pairs.

    Returns {route: [response_text, ...]} for quality-validated responses.
    """
    scores = load_quality_scores(min_score)
    pairs = load_training_pairs()

    by_route = {}
    matched = 0

    for ph, score_entry in scores.items():
        pair = pairs.get(ph)
        if not pair:
            continue

        route = score_entry.get("route", "general")
        if route_filter and route != route_filter:
            continue

        response = pair.get("response", "")
        if len(response) < MIN_RESPONSE_LEN:
            continue

        by_route.setdefault(route, []).append(response)
        matched += 1

    return by_route


def ingest_route(route: str, texts: list, dry_run: bool = False):
    """Ingest texts into the Engram table for a route."""
    ENGRAM_DIR.mkdir(parents=True, exist_ok=True)
    engram_path = ENGRAM_DIR / f"{route}.engr"

    # Write texts to a temp file
    tmp_file = Path(f"/tmp/engram_write_{route}_{int(time.time())}.txt")
    with open(tmp_file, "w") as f:
        for text in texts:
            f.write(text + "\n\n")

    total_chars = sum(len(t) for t in texts)

    if dry_run:
        print(f"  [DRY RUN] {route}: {len(texts)} responses, {total_chars} chars → {engram_path}")
        tmp_file.unlink(missing_ok=True)
        return

    # Run engram_ingest.py
    if not INGEST_SCRIPT.exists():
        print(f"  ERROR: {INGEST_SCRIPT} not found", file=sys.stderr)
        return

    cmd = [
        sys.executable, str(INGEST_SCRIPT),
        "--input", str(tmp_file),
        "--output", str(engram_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print(f"  {route}: ingested {len(texts)} responses ({total_chars} chars) → {engram_path}")
            if result.stdout.strip():
                # Show n-gram count from ingestion
                for line in result.stdout.strip().split("\n")[-3:]:
                    print(f"    {line}")
        else:
            print(f"  ERROR {route}: {result.stderr[:200]}", file=sys.stderr)
    except subprocess.TimeoutExpired:
        print(f"  ERROR {route}: timeout", file=sys.stderr)
    finally:
        tmp_file.unlink(missing_ok=True)


def batch_score_unscored(dry_run: bool = False):
    """Score unscored training pairs via the 9B model (port 8085, CPU)."""
    import hashlib
    import urllib.request

    if not TRAINING_LOG.exists():
        return 0

    # Load existing scores
    scored_hashes = set()
    if QUALITY_LOG.exists():
        with open(QUALITY_LOG) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get("prompt_hash"):
                        scored_hashes.add(entry["prompt_hash"])
                except (json.JSONDecodeError, KeyError):
                    continue

    # Find unscored pairs
    unscored = []
    with open(TRAINING_LOG) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                ph = entry.get("prompt_hash") or hashlib.md5(entry.get("prompt", "").encode()).hexdigest()[:16]
                if ph not in scored_hashes and len(entry.get("response", "")) > MIN_RESPONSE_LEN:
                    entry["prompt_hash"] = ph
                    unscored.append(entry)
            except (json.JSONDecodeError, KeyError):
                continue

    if not unscored:
        print("  No unscored responses to process.")
        return 0

    print(f"  Scoring {len(unscored)} unscored responses via 9B (port 8085)...")
    if dry_run:
        print(f"  [DRY RUN] Would score {len(unscored)} responses")
        return 0

    # Check if scorer is available
    try:
        req = urllib.request.urlopen(f"{SCORER_URL}/health", timeout=5)
        if req.status != 200:
            print("  WARN: 9B scorer not available, skipping batch scoring")
            return 0
    except Exception:
        print("  WARN: 9B scorer not available (port 8085), skipping batch scoring")
        return 0

    scored = 0
    for entry in unscored:
        prompt = entry.get("prompt", "")[:200]
        response = entry.get("response", "")[:500]
        route = entry.get("route", "general")

        scoring_prompt = json.dumps({
            "model": "scorer",
            "messages": [{"role": "user", "content":
                f"Rate this AI response for accuracy and helpfulness on a scale of 1-5. "
                f"Reply with ONLY a single digit (1-5).\n\n"
                f"User question: {prompt}\n\n"
                f"AI response: {response}"}],
            "max_tokens": 8,
            "temperature": 0.1,
            "chat_template_kwargs": {"enable_thinking": False}
        }).encode()

        try:
            req = urllib.request.Request(
                f"{SCORER_URL}/v1/chat/completions",
                data=scoring_prompt,
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=60)
            result = json.loads(resp.read())
            content = result["choices"][0]["message"]["content"].strip()
            # Extract score digit
            score = None
            for ch in content:
                if ch.isdigit() and 1 <= int(ch) <= 5:
                    score = int(ch)
                    break

            if score is not None:
                score_entry = {
                    "prompt_hash": entry["prompt_hash"],
                    "score": score,
                    "route": route,
                    "scorer": "qwen9b-batch",
                    "ts": datetime.now().isoformat(),
                }
                with open(QUALITY_LOG, "a") as f:
                    f.write(json.dumps(score_entry) + "\n")
                scored += 1
        except Exception as e:
            print(f"  WARN: scoring failed for {entry['prompt_hash'][:8]}: {e}")
            continue

    print(f"  Scored {scored}/{len(unscored)} responses")
    return scored


def decay_engram_meta(dry_run: bool = False):
    """Decay old n-gram metadata. Halve weight >30d, delete >90d.

    Uses .engr.meta JSON sidecar files to track last_used timestamps.
    The .engr binary is rebuilt from scratch during ingest, so decay
    affects which responses are RE-ingested on next nightly.
    """
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    half_cutoff = now - timedelta(days=DECAY_HALF_DAYS)
    delete_cutoff = now - timedelta(days=DECAY_DELETE_DAYS)

    if not QUALITY_LOG.exists():
        return

    # Read all quality scores
    entries = []
    decayed = 0
    deleted = 0

    with open(QUALITY_LOG) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                ts_str = entry.get("ts", "")
                if ts_str:
                    try:
                        ts = datetime.fromisoformat(ts_str)
                    except ValueError:
                        ts = now  # keep if unparseable
                else:
                    ts = now

                if ts < delete_cutoff:
                    deleted += 1
                    continue  # drop old entries
                elif ts < half_cutoff:
                    entry["score"] = max(1, entry.get("score", 3) // 2)
                    decayed += 1

                entries.append(entry)
            except (json.JSONDecodeError, KeyError):
                entries.append(json.loads(line.strip()) if line.strip() else None)

    entries = [e for e in entries if e is not None]

    if dry_run:
        print(f"  [DRY RUN] Decay: {decayed} halved, {deleted} deleted out of {decayed+deleted+len(entries)}")
        return

    if decayed > 0 or deleted > 0:
        # Backup and rewrite
        backup = ARCHIVE_DIR / f"quality_scores_{now.strftime('%Y%m%d')}.jsonl"
        if not backup.exists():
            import shutil
            shutil.copy(QUALITY_LOG, backup)

        with open(QUALITY_LOG, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        print(f"  Decay: {decayed} scores halved (>{DECAY_HALF_DAYS}d), {deleted} deleted (>{DECAY_DELETE_DAYS}d)")
    else:
        print(f"  Decay: nothing to decay ({len(entries)} entries all fresh)")


def resolve_conflicts(dry_run: bool = False):
    """Resolve conflicting quality scores for the same prompt_hash.
    Keep the entry with the highest score. Log conflicts.
    """
    if not QUALITY_LOG.exists():
        return

    by_hash = {}
    conflicts = 0

    with open(QUALITY_LOG) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                ph = entry.get("prompt_hash")
                if not ph:
                    continue
                if ph in by_hash:
                    conflicts += 1
                    if entry.get("score", 0) > by_hash[ph].get("score", 0):
                        by_hash[ph] = entry
                else:
                    by_hash[ph] = entry
            except (json.JSONDecodeError, KeyError):
                continue

    if conflicts > 0 and not dry_run:
        with open(QUALITY_LOG, "w") as f:
            for entry in by_hash.values():
                f.write(json.dumps(entry) + "\n")
        print(f"  Conflicts: {conflicts} resolved (kept highest score)")
    elif conflicts > 0:
        print(f"  [DRY RUN] Would resolve {conflicts} conflicts")
    else:
        print(f"  Conflicts: none found")


def main():
    parser = argparse.ArgumentParser(description="Nightly Engram WRITE from quality-validated responses")
    parser.add_argument("--route", default=None, help="Process only this route")
    parser.add_argument("--min-score", type=int, default=4, help="Minimum quality score (default: 4)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be ingested")
    parser.add_argument("--skip-scoring", action="store_true", help="Skip 9B batch scoring")
    parser.add_argument("--skip-decay", action="store_true", help="Skip decay pass")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"Engram WRITE v2 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")

    # Step 0: Batch score unscored responses via 9B
    if not args.skip_scoring:
        print("\n[1/4] Batch scoring unscored responses...")
        batch_score_unscored(args.dry_run)
    else:
        print("\n[1/4] Batch scoring: SKIPPED")

    # Step 1: Decay old entries
    if not args.skip_decay:
        print("\n[2/4] Decay pass...")
        decay_engram_meta(args.dry_run)
    else:
        print("\n[2/4] Decay: SKIPPED")

    # Step 2: Resolve conflicts
    print("\n[3/4] Conflict resolution...")
    resolve_conflicts(args.dry_run)

    # Step 3: Extract and ingest
    print("\n[4/4] Ingesting quality-validated responses...")
    by_route = extract_good_responses(args.min_score, args.route)

    if not by_route:
        print("  No quality-validated responses found (score >= {}).".format(args.min_score))
        print(f"  Quality scores: {QUALITY_LOG} ({sum(1 for _ in open(QUALITY_LOG)) if QUALITY_LOG.exists() else 0} entries)")
        print(f"  Training pairs: {TRAINING_LOG} ({sum(1 for _ in open(TRAINING_LOG)) if TRAINING_LOG.exists() else 0} entries)")
        return

    total = sum(len(v) for v in by_route.values())
    print(f"\n  Found {total} quality-validated responses across {len(by_route)} routes:")
    for route, texts in sorted(by_route.items()):
        print(f"    {route}: {len(texts)} responses")

    print()
    for route, texts in sorted(by_route.items()):
        ingest_route(route, texts, args.dry_run)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
