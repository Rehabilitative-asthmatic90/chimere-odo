"""semantic_fewshot.py — FAISS-backed semantic few-shot retrieval

Replaces keyword-based few-shot matching with embedding similarity.
Uses Qwen3-Embedding-0.6B (already cached for ChromaDB RAG) + FAISS
for ~1ms semantic lookup over quality-gated training responses.

Architecture:
  training_pairs.jsonl → embed prompts → FAISS IVF index (RAM)
  user query → embed → FAISS search → top-K similar → inject as few-shot

Auto-reloads when training_pairs.jsonl changes (mtime check).
Thread-safe for ODO's concurrent enrichment pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("semantic_fewshot")

TRAINING_PAIRS = Path.home() / ".chimere" / "logs" / "training_pairs.jsonl"
QUALITY_SCORES = Path.home() / ".chimere" / "logs" / "quality_scores.jsonl"
MIN_QUALITY_SCORE = 3  # Was 4, too strict — only 3 entries in FAISS index
MIN_RESPONSE_LEN = 100
MIN_SIMILARITY = 0.30  # Cosine similarity threshold (was 0.35, too strict for sparse index)

# Lazy-loaded globals
_index = None
_entries: list[dict[str, Any]] = []
_embedder = None
_lock = threading.Lock()
_last_mtime: float = 0.0
_last_check: float = 0.0
CHECK_INTERVAL = 30.0  # Check file mtime every 30s


def _get_embedder():
    """Lazy-load Qwen3-Embedding (shared with ChromaDB RAG)."""
    global _embedder
    if _embedder is not None:
        return _embedder
    try:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        log.info("[SEMANTIC_FS] Qwen3-Embedding-0.6B loaded (CPU)")
    except Exception as e:
        log.warning("[SEMANTIC_FS] Failed to load embedder: %s", e)
    return _embedder


def _load_quality_scores() -> dict[str, int]:
    """Load quality scores indexed by prompt_hash."""
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


def _rebuild_index() -> bool:
    """Rebuild FAISS index from training_pairs.jsonl.

    Only includes quality-gated entries (score >= 4).
    Returns True if successful.
    """
    global _index, _entries, _last_mtime

    embedder = _get_embedder()
    if embedder is None:
        return False

    if not TRAINING_PAIRS.exists():
        log.warning("[SEMANTIC_FS] %s not found", TRAINING_PAIRS)
        return False

    _last_mtime = TRAINING_PAIRS.stat().st_mtime

    # Load quality scores for filtering
    quality_scores = _load_quality_scores()

    # Load and filter training pairs
    entries: list[dict[str, Any]] = []
    with TRAINING_PAIRS.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                prompt = entry.get("prompt", "")
                response = entry.get("response", "")
                reasoning = entry.get("reasoning", "")
                prompt_hash = entry.get("prompt_hash", "")

                if len(response) < MIN_RESPONSE_LEN:
                    continue

                # Quality gate
                score = quality_scores.get(prompt_hash, 3)
                if score < MIN_QUALITY_SCORE:
                    continue

                entries.append({
                    "prompt": prompt,
                    "response": response,
                    "reasoning": reasoning,
                    "route": entry.get("route", ""),
                    "score": score,
                })
            except json.JSONDecodeError:
                continue

    if not entries:
        log.info("[SEMANTIC_FS] No quality-gated entries found")
        _entries = []
        _index = None
        return False

    # Embed all prompts
    t0 = time.monotonic()
    prompts = [e["prompt"] for e in entries]
    embeddings = embedder.encode(prompts, normalize_embeddings=True,
                                  show_progress_bar=False)
    embed_ms = (time.monotonic() - t0) * 1000

    # Build FAISS index
    try:
        import faiss
        import numpy as np

        dim = embeddings.shape[1]
        # For small collections (<1000), use flat index (exact search)
        # For larger, use IVF
        if len(entries) < 500:
            index = faiss.IndexFlatIP(dim)  # Inner product = cosine (normalized)
        else:
            nlist = min(len(entries) // 10, 100)
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist,
                                        faiss.METRIC_INNER_PRODUCT)
            index.train(np.array(embeddings, dtype=np.float32))
            index.nprobe = min(nlist, 10)

        index.add(np.array(embeddings, dtype=np.float32))
        _index = index
        _entries = entries

        log.info(
            "[SEMANTIC_FS] Built FAISS index: %d entries, dim=%d, %.0fms embed",
            len(entries), dim, embed_ms,
        )
        return True

    except ImportError:
        log.warning("[SEMANTIC_FS] faiss-cpu not installed, falling back to numpy")
        # Fallback: store embeddings as numpy array, use dot product search
        import numpy as np
        _index = np.array(embeddings, dtype=np.float32)
        _entries = entries
        log.info(
            "[SEMANTIC_FS] Built numpy index: %d entries, dim=%d, %.0fms",
            len(entries), embeddings.shape[1], embed_ms,
        )
        return True


def _maybe_refresh():
    """Check if training_pairs.jsonl changed and rebuild if needed."""
    global _last_check
    now = time.monotonic()
    if now - _last_check < CHECK_INTERVAL:
        return
    _last_check = now

    if not TRAINING_PAIRS.exists():
        return
    mtime = TRAINING_PAIRS.stat().st_mtime
    if mtime > _last_mtime:
        log.info("[SEMANTIC_FS] training_pairs.jsonl changed, rebuilding...")
        _rebuild_index()


def find_semantic_fewshot(
    user_text: str,
    route_id: str = "",
    max_examples: int = 1,
    min_similarity: float = MIN_SIMILARITY,
) -> list[dict[str, str]]:
    """Find semantically similar training responses.

    Returns list of {"input": prompt, "output": response} dicts,
    sorted by descending similarity.

    Thread-safe. Auto-rebuilds on data change.
    Typical latency: ~1ms (FAISS) + ~5ms (embedding).
    """
    with _lock:
        _maybe_refresh()
        if _index is None:
            _rebuild_index()

    if _index is None or not _entries:
        return []

    embedder = _get_embedder()
    if embedder is None:
        return []

    # Encode query
    t0 = time.monotonic()
    query_emb = embedder.encode([user_text], normalize_embeddings=True,
                                 show_progress_bar=False)

    # Search
    import numpy as np
    query_np = np.array(query_emb, dtype=np.float32)

    try:
        import faiss
        if isinstance(_index, faiss.Index):
            k = min(max_examples * 3, len(_entries))
            scores, indices = _index.search(query_np, k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or score < min_similarity:
                    continue
                entry = _entries[idx]
                # Optional: filter by route if specified
                if route_id and entry["route"] and entry["route"] != route_id:
                    score *= 0.7  # Penalize cross-domain but don't exclude
                results.append((score, entry))
        else:
            raise ImportError  # Use numpy fallback
    except (ImportError, TypeError):
        # Numpy fallback
        sims = query_np @ _index.T
        top_k = min(max_examples * 3, len(_entries))
        top_indices = np.argsort(sims[0])[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(sims[0][idx])
            if score < min_similarity:
                break
            entry = _entries[idx]
            if route_id and entry["route"] and entry["route"] != route_id:
                score *= 0.7
            results.append((score, entry))

    search_ms = (time.monotonic() - t0) * 1000
    results.sort(key=lambda x: -x[0])

    output = []
    for score, entry in results[:max_examples]:
        output.append({
            "input": entry["prompt"],
            "output": entry["response"],
            "score": round(score, 4),
        })
        log.debug(
            "[SEMANTIC_FS] Match (%.3f, %.1fms): %s",
            score, search_ms, entry["prompt"][:60],
        )

    return output


def warmup():
    """Pre-load embedder and build index. Call at ODO startup."""
    with _lock:
        _rebuild_index()


# Self-test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== Semantic Few-Shot Self-Test ===")

    print("\n[1] Building index...")
    t0 = time.monotonic()
    warmup()
    print(f"    Index built in {(time.monotonic() - t0)*1000:.0f}ms")
    print(f"    Entries: {len(_entries)}")

    if not _entries:
        print("    No quality-gated entries. Need training_pairs.jsonl + quality_scores.jsonl")
        print("    Exiting.")
        exit(0)

    test_queries = [
        ("Protocole LCA retour au sport", "kine"),
        ("Implémente un binary search tree en Python", "code"),
        ("Analyse CVE-2024-1234 criticality", "cyber"),
        ("Quelle est la capitale de la France ?", ""),
    ]

    print(f"\n[2] Testing {len(test_queries)} queries...")
    for query, route in test_queries:
        t0 = time.monotonic()
        results = find_semantic_fewshot(query, route_id=route, max_examples=1)
        ms = (time.monotonic() - t0) * 1000
        if results:
            r = results[0]
            print(f"    Q: {query[:50]}...")
            print(f"    → sim={r['score']:.3f}, {ms:.1f}ms: {r['input'][:60]}...")
        else:
            print(f"    Q: {query[:50]}... → no match ({ms:.1f}ms)")

    print("\n=== Done ===")
