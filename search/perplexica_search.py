#!/usr/bin/env python3
"""Perplexica Search API wrapper. Standalone CLI + importable module.
Includes local file cache and rate limiting (1 req/s)."""
import hashlib
import json
import os
import sys
import time
import argparse
from pathlib import Path

import requests

PERPLEXICA_URL = "http://127.0.0.1:3000"
CACHE_DIR = Path.home() / ".chimere" / ".perplexica_cache"
CACHE_TTL_DEFAULT = 3600  # 1 hour
RATE_LIMIT_INTERVAL = 1.0  # seconds between requests
_last_request_time = 0

# Provider IDs matching config.json
CHAT_PROVIDER_ID = "qwen35-local"
CHAT_MODEL_KEY = "qwen3.5-35b"
EMBEDDING_PROVIDER_ID = "transformers-local"
EMBEDDING_MODEL_KEY = "Xenova/all-MiniLM-L6-v2"


def _cache_key(query: str, mode: str) -> str:
    return hashlib.md5(f"{query}:{mode}".encode()).hexdigest()


def search(query: str, mode: str = "speed", sources: list[str] | None = None,
           cache_ttl: int | None = None, use_cache: bool = True) -> list[dict]:
    """Search Perplexica and return list of {title, url, content}.

    Args:
        query: Search query string
        mode: Optimization mode: "speed", "balanced", or "quality" (default "speed")
        sources: Source types to search. Default ["web"].
                 Options: "web", "academic", "discussions"
        cache_ttl: Cache TTL in seconds (default 3600)
        use_cache: Whether to use cache (default True)

    Returns:
        List of dicts with keys: title, url, content
    """
    global _last_request_time
    ttl = cache_ttl if cache_ttl is not None else CACHE_TTL_DEFAULT
    if sources is None:
        sources = ["web"]

    if mode not in ("speed", "balanced", "quality"):
        print(f"Warning: Invalid mode '{mode}', using 'speed'", file=sys.stderr)
        mode = "speed"

    # Check cache
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = CACHE_DIR / f"{_cache_key(query, mode)}.json"
        if cache_file.exists():
            age = time.time() - cache_file.stat().st_mtime
            if age < ttl:
                try:
                    return json.loads(cache_file.read_text())
                except (json.JSONDecodeError, OSError):
                    pass

    # Rate limit
    elapsed = time.time() - _last_request_time
    if elapsed < RATE_LIMIT_INTERVAL:
        time.sleep(RATE_LIMIT_INTERVAL - elapsed)

    # Perplexica API request
    search_data = {
        "sources": sources,
        "query": query,
        "chatModel": {
            "providerId": CHAT_PROVIDER_ID,
            "key": CHAT_MODEL_KEY,
        },
        "embeddingModel": {
            "providerId": EMBEDDING_PROVIDER_ID,
            "key": EMBEDDING_MODEL_KEY,
        },
        "optimizationMode": mode,
        "history": [],
    }

    try:
        resp = requests.post(
            f"{PERPLEXICA_URL}/api/search",
            headers={"Content-Type": "application/json"},
            json=search_data,
            timeout=600,  # 10 min max for full Perplexica synthesis via Qwen3.5
        )
        _last_request_time = time.time()
        resp.raise_for_status()

        result_data = resp.json()
        sources_list = result_data.get("sources", [])

        out = []
        for source in sources_list:
            if isinstance(source, dict):
                meta = source.get("metadata", {})
                out.append({
                    "title": meta.get("title", source.get("title", "Untitled")),
                    "url": meta.get("url", source.get("url", "")),
                    "content": source.get("content", ""),
                })

        # Include Perplexica's synthesized answer as first result
        message = result_data.get("message", "")
        if message:
            out.insert(0, {
                "title": "Perplexica Answer",
                "url": "",
                "content": message,
            })

        # Save to cache
        if use_cache:
            try:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache_file = CACHE_DIR / f"{_cache_key(query, mode)}.json"
                cache_file.write_text(json.dumps(out, ensure_ascii=False))
            except OSError:
                pass

        return out

    except requests.RequestException as e:
        print(f"Error calling Perplexica API: {e}", file=sys.stderr)
        return []
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing Perplexica response: {e}", file=sys.stderr)
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perplexica Search CLI")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--mode", choices=["speed", "balanced", "quality"],
                        default="speed", help="Optimization mode (default: speed)")
    parser.add_argument("--sources", nargs="+", default=["web"],
                        choices=["web", "academic", "discussions"],
                        help="Source types (default: web)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--cache-ttl", type=int, default=None,
                        help="Cache TTL in seconds (default 3600)")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache")
    args = parser.parse_args()

    results = search(args.query, mode=args.mode, sources=args.sources,
                     cache_ttl=args.cache_ttl, use_cache=not args.no_cache)

    if args.json:
        print(json.dumps(results, ensure_ascii=False))
    else:
        for r in results:
            content_preview = r["content"][:200] if r["content"] else ""
            print(f"- {r['title']}: {content_preview}")
            if r["url"]:
                print(f"  URL: {r['url']}")
