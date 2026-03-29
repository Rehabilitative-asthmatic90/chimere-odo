#!/usr/bin/env python3
"""Web Deep Fetch — fetch, extract, chunk, and re-rank web pages for RAG.

Takes SearXNG/Brave search results (snippets), fetches the top URLs,
extracts full content via trafilatura, chunks at ~512 tokens,
and re-ranks chunks by cosine similarity with the query embedding.

Returns enriched results with full content chunks ready for LLM synthesis.
"""
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import trafilatura

# Optional PDF support
try:
    import pymupdf
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# Optional embedding-based re-ranking
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CACHE_DIR = Path.home() / ".chimere" / ".deepfetch_cache"
CACHE_TTL = 86400  # 24h for full page content
MAX_FETCH_PAGES = 3
MAX_CHUNKS_PER_PAGE = 10
CHUNK_SIZE = 512  # tokens (~380 words)
CHUNK_OVERLAP = 50  # tokens
FETCH_TIMEOUT = 15  # seconds per page
USER_AGENT = "Chimère-Research/1.0"

# Embedding model (same as ChromaDB RAG pipeline)
_embedder = None
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def _get_embedder():
    global _embedder
    if _embedder is None and HAS_EMBEDDINGS:
        _embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return _embedder


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------
def _fetch_html(url: str) -> str | None:
    """Fetch HTML with browser-like User-Agent. Tries trafilatura first, then requests."""
    import requests as req
    # Try trafilatura's built-in fetcher first
    html = trafilatura.fetch_url(url)
    if html:
        return html
    # Fallback: requests with browser UA (some sites block trafilatura's default UA)
    try:
        resp = req.get(url, timeout=FETCH_TIMEOUT, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
        })
        resp.raise_for_status()
        if len(resp.text) > 500:
            return resp.text
    except Exception:
        pass
    return None


def _offpunk_extract(url: str) -> str | None:
    """Fallback extraction via offpunk (handles JS-heavy sites)."""
    try:
        import subprocess
        venv_python = str(Path.home() / ".chimere/venvs/pipeline/bin/python")
        result = subprocess.run(
            [venv_python, "-m", "offpunk", url, "--command", "quit"],
            capture_output=True, text=True, timeout=20,
            env={**os.environ, "TERM": "dumb"},
        )
        if result.returncode == 0 and result.stdout:
            # Strip ANSI codes and offpunk UI noise
            text = re.sub(r'\x1b\[[0-9;]*m', '', result.stdout)
            # Remove offpunk boilerplate lines
            lines = [l for l in text.splitlines()
                     if not any(skip in l for skip in (
                         'render images', 'Welcome to Offpunk',
                         'Type `help`', 'close your screen',
                         'ERROR', 'Bad SSL', 'accept_bad_ssl',
                         'moving from tofu', 'Creating config',
                     ))]
            clean = "\n".join(lines).strip()
            if len(clean) > 100:
                return clean
    except Exception:
        pass
    return None


def _fetch_and_extract(url: str) -> str | None:
    """Fetch URL and extract clean text content.
    Uses trafilatura first, offpunk as fallback for JS-heavy sites."""
    try:
        if url.lower().endswith(".pdf"):
            return _extract_pdf(url)

        html = _fetch_html(url)
        if not html:
            # No HTML at all — try offpunk as last resort
            return _offpunk_extract(url)

        text = trafilatura.extract(
            html,
            output_format="txt",
            include_links=False,
            include_tables=True,
            include_comments=False,
            favor_precision=True,
        )
        if text and len(text) > 100:
            return text

        # Trafilatura returned too little — try offpunk
        return _offpunk_extract(url)

    except Exception:
        return None


def _extract_pdf(url: str) -> str | None:
    """Download and extract text from a PDF URL."""
    if not HAS_PYMUPDF:
        return None
    try:
        import requests
        resp = requests.get(url, timeout=FETCH_TIMEOUT,
                            headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        doc = pymupdf.open(stream=resp.content, filetype="pdf")
        pages = []
        for page in doc:
            text = page.get_text()
            if text.strip():
                pages.append(text)
        doc.close()
        return "\n\n".join(pages) if pages else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~0.75 tokens per character for French/English."""
    return int(len(text) * 0.75 / 4)  # ~4 chars per token


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into chunks of ~chunk_size tokens with overlap.

    Splits on paragraph boundaries first, then sentences, then fixed size.
    """
    if not text:
        return []

    # Split on paragraphs first
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current = []
    current_tokens = 0
    chars_per_token = 4  # approximate

    target_chars = chunk_size * chars_per_token
    overlap_chars = overlap * chars_per_token

    for para in paragraphs:
        para_len = len(para)

        if current_tokens + para_len / chars_per_token > chunk_size * 1.2 and current:
            # Flush current chunk
            chunk_text_str = "\n\n".join(current)
            chunks.append(chunk_text_str)

            # Keep overlap from end of current chunk
            if overlap_chars > 0 and chunk_text_str:
                overlap_text = chunk_text_str[-overlap_chars:]
                current = [overlap_text]
                current_tokens = len(overlap_text) / chars_per_token
            else:
                current = []
                current_tokens = 0

        # If single paragraph is too large, split on sentences
        if para_len / chars_per_token > chunk_size * 1.5:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                sent_tokens = len(sent) / chars_per_token
                if current_tokens + sent_tokens > chunk_size * 1.2 and current:
                    chunks.append("\n\n".join(current))
                    current = []
                    current_tokens = 0
                current.append(sent)
                current_tokens += sent_tokens
        else:
            current.append(para)
            current_tokens += para_len / chars_per_token

    if current:
        chunks.append("\n\n".join(current))

    return chunks[:MAX_CHUNKS_PER_PAGE * 3]  # safety limit


# ---------------------------------------------------------------------------
# Re-ranking
# ---------------------------------------------------------------------------
def rerank_chunks(query: str, chunks: list[dict],
                  top_k: int = 5) -> list[dict]:
    """Re-rank chunks by cosine similarity to query.

    Each chunk dict must have 'text' key. Adds 'relevance_score' key.
    """
    if not chunks:
        return []

    embedder = _get_embedder()
    if embedder is None:
        # No embeddings available, return first top_k
        for i, c in enumerate(chunks):
            c["relevance_score"] = 1.0 - i * 0.05
        return chunks[:top_k]

    texts = [c["text"] for c in chunks]
    query_emb = embedder.encode([query], normalize_embeddings=True)
    chunk_embs = embedder.encode(texts, normalize_embeddings=True,
                                 batch_size=32)

    # Cosine similarity (already normalized)
    scores = (chunk_embs @ query_emb.T).flatten()

    for i, chunk in enumerate(chunks):
        chunk["relevance_score"] = float(scores[i])

    # Sort by relevance
    chunks.sort(key=lambda c: c["relevance_score"], reverse=True)
    return chunks[:top_k]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def _cache_key(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def _get_cached_content(url: str) -> str | None:
    cache_file = CACHE_DIR / f"{_cache_key(url)}.txt"
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < CACHE_TTL:
            try:
                return cache_file.read_text(encoding="utf-8")
            except OSError:
                pass
    return None


def _set_cached_content(url: str, content: str):
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = CACHE_DIR / f"{_cache_key(url)}.txt"
        cache_file.write_text(content, encoding="utf-8")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def deep_fetch(query: str, search_results: list[dict],
               max_pages: int = MAX_FETCH_PAGES,
               top_chunks: int = 5,
               use_cache: bool = True,
               use_reranking: bool = False) -> list[dict]:
    """Fetch full content from top search result URLs, chunk, and optionally re-rank.

    Args:
        query: Original search query
        search_results: List of {title, url, content} from SearXNG/Brave
        max_pages: Max pages to fetch (default 3)
        top_chunks: Number of top chunks to return after re-ranking
        use_cache: Use content cache
        use_reranking: Whether to use embedding-based re-ranking (slow first load)

    Returns:
        List of {title, url, text, relevance_score, source_rank}
        sorted by relevance_score descending.
    """
    # Filter URLs worth fetching
    urls_to_fetch = []
    url_meta = {}  # url -> {title, rank}
    for i, result in enumerate(search_results[:max_pages * 2]):
        url = result.get("url", "")
        if not url or not url.startswith("http"):
            continue
        # Skip known low-value domains
        if any(skip in url for skip in [
            "youtube.com", "twitter.com", "x.com", "facebook.com",
            "instagram.com", "linkedin.com", "tiktok.com",
        ]):
            continue
        url_meta[url] = {"title": result.get("title", ""), "rank": i}
        urls_to_fetch.append(url)
        if len(urls_to_fetch) >= max_pages:
            break

    if not urls_to_fetch:
        return []

    # Parallel fetch
    contents = {}
    def _fetch_one(url):
        if use_cache:
            cached = _get_cached_content(url)
            if cached:
                return url, cached
        content = _fetch_and_extract(url)
        if content and use_cache:
            _set_cached_content(url, content)
        return url, content

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(_fetch_one, url): url
                   for url in urls_to_fetch}
        for future in as_completed(futures, timeout=FETCH_TIMEOUT + 5):
            try:
                url, content = future.result(timeout=FETCH_TIMEOUT + 5)
                if content:
                    contents[url] = content
            except Exception:
                pass

    if not contents:
        return []

    # Chunk all fetched content
    all_chunks = []
    for url, content in contents.items():
        meta = url_meta.get(url, {})
        text_chunks = chunk_text(content)
        for chunk in text_chunks[:MAX_CHUNKS_PER_PAGE]:
            all_chunks.append({
                "text": chunk,
                "url": url,
                "title": meta.get("title", ""),
                "source_rank": meta.get("rank", 99),
            })

    if use_reranking:
        # Embedding-based re-ranking (slow first load ~60s, then fast)
        ranked = rerank_chunks(query, all_chunks, top_k=top_chunks)
        return ranked
    else:
        # Fast mode: return top chunks sorted by source rank (SearXNG ordering)
        # Distribute chunks across sources for diversity
        by_url = {}
        for c in all_chunks:
            by_url.setdefault(c["url"], []).append(c)
        result = []
        idx = 0
        while len(result) < top_chunks:
            added = False
            for url in sorted(by_url, key=lambda u: url_meta.get(u, {}).get("rank", 99)):
                chunks_list = by_url[url]
                if idx < len(chunks_list):
                    c = chunks_list[idx]
                    c["relevance_score"] = 1.0 - c["source_rank"] * 0.1
                    result.append(c)
                    added = True
                    if len(result) >= top_chunks:
                        break
            idx += 1
            if not added:
                break
        return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Web Deep Fetch")
    parser.add_argument("query", help="Search query for re-ranking")
    parser.add_argument("--urls", nargs="+", help="URLs to fetch directly")
    parser.add_argument("--search-json", help="JSON file with search results")
    parser.add_argument("--top", type=int, default=5, help="Top chunks")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    if args.urls:
        results = [{"url": u, "title": "", "content": ""} for u in args.urls]
    elif args.search_json:
        with open(args.search_json) as f:
            results = json.load(f)
    else:
        print("Provide --urls or --search-json", file=sys.stderr)
        sys.exit(1)

    chunks = deep_fetch(args.query, results, top_chunks=args.top)

    if args.json:
        print(json.dumps(chunks, ensure_ascii=False, indent=2))
    else:
        for i, c in enumerate(chunks, 1):
            score = c.get("relevance_score", 0)
            print(f"\n--- Chunk {i} (score: {score:.3f}) [{c['title'][:50]}] ---")
            print(f"URL: {c['url']}")
            print(c["text"][:500])
