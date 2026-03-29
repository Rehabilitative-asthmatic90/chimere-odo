#!/usr/bin/env python3
"""SearXNG direct search — fast multi-engine search without LLM synthesis.
Uses Perplexica's built-in SearXNG. Returns raw results for the main LLM to synthesize."""
import hashlib
import json
import os
import sys
import time
import argparse
from pathlib import Path

import requests

# SearXNG inside Perplexica container, exposed via Docker port mapping
SEARXNG_URL = "http://127.0.0.1:3000"
# Fallback: direct SearXNG if available
SEARXNG_DIRECT_URL = "http://127.0.0.1:8080"
CACHE_DIR = Path.home() / ".chimere" / ".searxng_cache"
CACHE_TTL_DEFAULT = 3600
RATE_LIMIT_INTERVAL = 1.0
_last_request_time = 0


def search(query: str, count: int = 10, lang: str = "fr",
           categories: str = "general", cache_ttl: int | None = None,
           use_cache: bool = True) -> list[dict]:
    """Search SearXNG directly and return list of {title, url, content}.

    Args:
        query: Search query
        count: Max results to return (default 10)
        lang: Search language (default "fr")
        categories: SearXNG categories (default "general")
        cache_ttl: Cache TTL in seconds
        use_cache: Whether to use file cache
    """
    global _last_request_time
    ttl = cache_ttl if cache_ttl is not None else CACHE_TTL_DEFAULT

    # Cache check
    cache_key = hashlib.md5(f"{query}:{count}:{lang}".encode()).hexdigest()
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = CACHE_DIR / f"{cache_key}.json"
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

    # Try direct SearXNG first (faster), then Perplexica's built-in
    for base_url in [SEARXNG_DIRECT_URL, SEARXNG_URL]:
        try:
            resp = requests.get(
                f"{base_url}/search",
                params={
                    "q": query,
                    "format": "json",
                    "language": lang,
                    "categories": categories,
                },
                timeout=15,
            )
            _last_request_time = time.time()

            if resp.status_code == 403:
                continue  # JSON format not enabled, try next
            resp.raise_for_status()

            data = resp.json()
            results = data.get("results", [])

            out = []
            for r in results[:count]:
                out.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                    "engine": r.get("engine", ""),
                })

            # Cache
            if use_cache and out:
                try:
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    (CACHE_DIR / f"{cache_key}.json").write_text(
                        json.dumps(out, ensure_ascii=False))
                except OSError:
                    pass

            return out

        except requests.RequestException:
            continue

    return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SearXNG Direct Search")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--lang", default="fr")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results = search(args.query, count=args.count, lang=args.lang)
    if args.json:
        print(json.dumps(results, ensure_ascii=False))
    else:
        for r in results:
            print(f"- {r['title']}")
            print(f"  {r['content'][:150]}")
            if r['url']:
                print(f"  {r['url']}")
