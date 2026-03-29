"""dynamic_engram.py — Build per-query Engram from web search results

NOVEL CONCEPT: Instead of static n-gram tables, build a TEMPORARY
Engram at query time from web search results. The search extracts
factual content → tokenize → build mini n-gram table → bias logits
during generation.

Pipeline:
  1. User asks question
  2. Web search finds relevant pages (deep_search_sota)
  3. Extract key factual sentences from results
  4. Tokenize → build temporary .engr file
  5. chimere-server loads it for this generation
  6. After generation, discard temporary table

This gives the model "web-grounded" factual recall via logit biasing,
complementing the system prompt injection (RAG) with token-level guidance.

Integration: called from ODO enricher for "quality" and "ultra" modes.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

ENGRAM_DIR = Path.home() / ".chimere" / "data" / "engram"
INGEST_SCRIPT = Path.home() / ".chimere" / "bin" / "engram_ingest.py"
TOKENIZER_PATH = Path.home() / ".chimere" / "models" / "Qwen3.5-35B-A3B-BF16" / "tokenizer.json"

# Cache: avoid rebuilding for the same query
_cache: dict[str, tuple[str, float]] = {}  # query_hash → (engr_path, timestamp)
CACHE_TTL = 600  # 10 minutes


def build_dynamic_engram(
    search_results: list[dict],
    query: str,
    max_sentences: int = 50,
) -> Optional[str]:
    """Build a temporary .engr file from web search results.

    Args:
        search_results: list of {"text": ..., "title": ..., "url": ...} chunks
        query: the user's query (for cache key)
        max_sentences: max sentences to include

    Returns:
        Path to temporary .engr file, or None if build failed.
    """
    if not search_results:
        return None

    # Cache check
    qhash = hashlib.md5(query.encode()).hexdigest()[:12]
    if qhash in _cache:
        path, ts = _cache[qhash]
        if time.time() - ts < CACHE_TTL and os.path.exists(path):
            return path

    # Extract factual sentences from search results
    sentences = []
    for chunk in search_results:
        text = chunk.get("text", "")
        if not text:
            continue
        # Split into sentences and filter
        import re
        sents = re.split(r'(?<=[.!?])\s+', text)
        for s in sents:
            s = s.strip()
            # Keep only informative sentences (not too short, not boilerplate)
            if 30 < len(s) < 500 and not _is_boilerplate(s):
                sentences.append(s)

    if len(sentences) < 3:
        return None

    # Limit and deduplicate
    seen = set()
    unique = []
    for s in sentences:
        key = s[:80].lower()
        if key not in seen:
            seen.add(key)
            unique.append(s)
    sentences = unique[:max_sentences]

    # Write corpus to temp file
    corpus_file = tempfile.NamedTemporaryFile(
        mode='w', suffix='.txt', delete=False, prefix='engram_dyn_'
    )
    for s in sentences:
        corpus_file.write(s + "\n\n")
    corpus_file.close()

    # Build .engr file
    engr_path = str(ENGRAM_DIR / f"_dynamic_{qhash}.engr")

    try:
        result = subprocess.run(
            ["python3", str(INGEST_SCRIPT),
             "--input", corpus_file.name,
             "--output", engr_path,
             "--order", "5",
             "--min-freq", "1"],  # Low freq threshold for small corpus
            capture_output=True, text=True, timeout=30,
        )
        os.unlink(corpus_file.name)

        if result.returncode != 0 or not os.path.exists(engr_path):
            return None

        size_kb = os.path.getsize(engr_path) / 1024
        print(f"[DYNAMIC_ENGRAM] Built {engr_path} ({size_kb:.0f} KB, "
              f"{len(sentences)} sentences)", flush=True)

        _cache[qhash] = (engr_path, time.time())
        return engr_path

    except Exception as e:
        print(f"[DYNAMIC_ENGRAM] Build failed: {e}", flush=True)
        try:
            os.unlink(corpus_file.name)
        except OSError:
            pass
        return None


def cleanup_old_dynamic():
    """Remove dynamic .engr files older than 1 hour."""
    for f in ENGRAM_DIR.glob("_dynamic_*.engr"):
        if time.time() - f.stat().st_mtime > 3600:
            f.unlink(missing_ok=True)


def _is_boilerplate(s: str) -> bool:
    """Detect boilerplate sentences to filter out."""
    import re
    patterns = [
        r"^(cookie|accept|privacy|terms|subscribe|sign up|log in)",
        r"^(click|tap|scroll|share|follow|like|comment)",
        r"^(copyright|©|all rights reserved)",
        r"^\d{1,2}/\d{1,2}/\d{2,4}",  # dates
        r"^(advertisement|sponsored|promoted)",
    ]
    s_lower = s.lower().strip()
    return any(re.search(p, s_lower) for p in patterns)


# Self-test
if __name__ == "__main__":
    # Simulate search results
    fake_results = [
        {"text": "Les recommandations HAS 2024 pour la lombalgie chronique incluent l'exercice physique comme traitement de première intention. La kinésithérapie active est préférée à la kinésithérapie passive. Les anti-inflammatoires non stéroïdiens sont recommandés en courte durée.", "title": "HAS 2024", "url": "https://has-sante.fr"},
        {"text": "Le protocole d'Alfredson consiste en 3 séries de 15 répétitions d'exercices excentriques, genou tendu puis genou fléchi, 2 fois par jour pendant 12 semaines. La douleur est acceptée jusqu'à un seuil de 5/10 sur l'EVA.", "title": "Alfredson", "url": "https://pubmed.ncbi.nlm.nih.gov"},
    ]

    path = build_dynamic_engram(fake_results, "recommandations HAS lombalgie")
    if path:
        print(f"Built: {path} ({os.path.getsize(path)/1024:.0f} KB)")
    else:
        print("Build failed")
