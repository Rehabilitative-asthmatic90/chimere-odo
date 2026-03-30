#!/usr/bin/env python3
"""Engram Semantic Tier 2 — FAISS-based fuzzy matching for Engram.

Stores (prompt_hash, response_embedding) pairs in a FAISS flat index.
Query returns top-K similar responses by cosine similarity.

The embedding is computed as the mean hidden state from the 35B model
via the /v1/embeddings endpoint (if available) or a lightweight
sentence-transformer fallback.

Usage:
    engram_semantic.py add --prompt-hash abc123 --text "capsulite rétractile..."
    engram_semantic.py query --text "épaule gelée" --top-k 5
    engram_semantic.py build --from-quality-log  # build from quality_scores.jsonl
    engram_semantic.py stats
"""

import argparse
import json
import os
import sys
import struct
import numpy as np
from pathlib import Path

_chimere_home = Path(os.environ.get("CHIMERE_HOME", str(Path.home() / ".chimere")))
SEMANTIC_DIR = _chimere_home / "data" / "engram" / "semantic"
INDEX_FILE = SEMANTIC_DIR / "faiss.index"
META_FILE = SEMANTIC_DIR / "meta.jsonl"
EMBEDDING_DIM = 384  # gte-small default, upgrade to 2048 with 35B embeddings later
QUALITY_LOG = _chimere_home / "logs" / "quality_scores.jsonl"
TRAINING_LOG = _chimere_home / "logs" / "training_pairs.jsonl"


def get_embedding(text: str, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Get text embedding. Uses gte-modernbert cross-encoder if available,
    falls back to simple TF-IDF-style bag-of-chars embedding."""
    import urllib.request

    # Try the 35B /v1/embeddings endpoint first
    try:
        req_data = json.dumps({
            "input": text[:512],
            "model": "qwen35"
        }).encode()
        req = urllib.request.Request(
            os.environ.get("EMBEDDING_URL", os.environ.get("ODO_BACKEND", "http://127.0.0.1:8081")) + "/v1/embeddings",
            data=req_data,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=10)
        result = json.loads(resp.read())
        emb = np.array(result["data"][0]["embedding"], dtype=np.float32)
        return emb / (np.linalg.norm(emb) + 1e-10)
    except Exception:
        pass

    # Fallback: simple character n-gram hash embedding (no model needed)
    emb = np.zeros(dim, dtype=np.float32)
    text = text.lower()[:1000]
    for i in range(len(text) - 2):
        trigram = text[i:i+3]
        h = hash(trigram) % dim
        emb[h] += 1.0
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb /= norm
    return emb


def load_index():
    """Load FAISS index and metadata."""
    meta = []
    if META_FILE.exists():
        with open(META_FILE) as f:
            for line in f:
                try:
                    meta.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

    embeddings = None
    if INDEX_FILE.exists():
        embeddings = np.fromfile(str(INDEX_FILE), dtype=np.float32)
        if len(meta) > 0:
            dim = len(embeddings) // len(meta)
            embeddings = embeddings.reshape(len(meta), dim)

    return meta, embeddings


def save_index(meta, embeddings):
    """Save FAISS index and metadata."""
    SEMANTIC_DIR.mkdir(parents=True, exist_ok=True)
    with open(META_FILE, "w") as f:
        for entry in meta:
            f.write(json.dumps(entry) + "\n")
    if embeddings is not None and len(embeddings) > 0:
        embeddings.astype(np.float32).tofile(str(INDEX_FILE))


def add_entry(prompt_hash: str, text: str, route: str = "general", score: int = 0):
    """Add a response to the semantic index."""
    meta, embeddings = load_index()

    # Check if already exists
    if any(m.get("prompt_hash") == prompt_hash for m in meta):
        return False

    emb = get_embedding(text)
    meta.append({
        "prompt_hash": prompt_hash,
        "route": route,
        "score": score,
        "text_preview": text[:200],
    })

    if embeddings is None or len(embeddings) == 0:
        embeddings = emb.reshape(1, -1)
    else:
        if embeddings.shape[1] != len(emb):
            print(f"WARN: dimension mismatch ({embeddings.shape[1]} vs {len(emb)}), rebuilding index")
            embeddings = emb.reshape(1, -1)
            meta = [meta[-1]]
        else:
            embeddings = np.vstack([embeddings, emb.reshape(1, -1)])

    save_index(meta, embeddings)
    return True


def query(text: str, top_k: int = 5):
    """Query the semantic index for similar responses."""
    meta, embeddings = load_index()
    if not meta or embeddings is None:
        return []

    q_emb = get_embedding(text, dim=embeddings.shape[1])
    # Cosine similarity (embeddings are normalized)
    scores = embeddings @ q_emb
    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_idx:
        results.append({
            **meta[idx],
            "similarity": float(scores[idx]),
        })
    return results


def build_from_quality_log():
    """Build semantic index from quality-scored training pairs."""
    if not QUALITY_LOG.exists() or not TRAINING_LOG.exists():
        print("Missing quality_scores.jsonl or training_pairs.jsonl")
        return

    # Load scored pairs
    scores = {}
    with open(QUALITY_LOG) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                ph = entry.get("prompt_hash")
                if ph and entry.get("score", 0) >= 4:
                    scores[ph] = entry
            except (json.JSONDecodeError, KeyError):
                continue

    pairs = {}
    with open(TRAINING_LOG) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                ph = entry.get("prompt_hash")
                if ph:
                    pairs[ph] = entry
            except (json.JSONDecodeError, KeyError):
                continue

    added = 0
    for ph, score_entry in scores.items():
        pair = pairs.get(ph)
        if not pair or len(pair.get("response", "")) < 100:
            continue
        text = f"{pair.get('prompt', '')}\n{pair.get('response', '')}"
        if add_entry(ph, text, score_entry.get("route", "general"), score_entry.get("score", 0)):
            added += 1

    print(f"Added {added} entries to semantic index")


def main():
    parser = argparse.ArgumentParser(description="Engram Semantic Tier 2")
    sub = parser.add_subparsers(dest="command")

    p_add = sub.add_parser("add", help="Add an entry")
    p_add.add_argument("--prompt-hash", required=True)
    p_add.add_argument("--text", required=True)
    p_add.add_argument("--route", default="general")

    p_query = sub.add_parser("query", help="Query similar responses")
    p_query.add_argument("--text", required=True)
    p_query.add_argument("--top-k", type=int, default=5)

    p_build = sub.add_parser("build", help="Build from quality log")

    p_stats = sub.add_parser("stats", help="Show index stats")

    args = parser.parse_args()

    if args.command == "add":
        ok = add_entry(args.prompt_hash, args.text, args.route)
        print("Added" if ok else "Already exists")

    elif args.command == "query":
        results = query(args.text, args.top_k)
        for r in results:
            print(f"  [{r['similarity']:.3f}] {r.get('route','?')} (score:{r.get('score','?')}): {r.get('text_preview','')[:100]}")

    elif args.command == "build":
        build_from_quality_log()

    elif args.command == "stats":
        meta, embeddings = load_index()
        print(f"Entries: {len(meta)}")
        if embeddings is not None:
            print(f"Embedding dim: {embeddings.shape[1] if len(embeddings.shape) > 1 else 'N/A'}")
            print(f"Index size: {embeddings.nbytes / 1024:.1f} KB")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
