#!/usr/bin/env python3
"""Brave Search API wrapper. Standalone CLI + importable module.
Includes local file cache and rate limiting (1 req/s free tier)."""
import hashlib
import json
import os
import sys
import time
import argparse
from pathlib import Path

import threading

import requests

API_URL = "https://api.search.brave.com/res/v1/web/search"
CACHE_DIR = Path.home() / ".chimere" / ".brave_cache"
CACHE_TTL_DEFAULT = 3600  # 1 hour
RATE_LIMIT_INTERVAL = 1.1  # seconds between requests (free tier: 1 req/s)
_last_request_time = 0
_rate_lock = threading.Lock()  # serialize Brave calls across threads

# Auto-load env on import (needed when used as module from search_router)
def _load_env():
    """Load BRAVE_API_KEY from ~/.chimere/.env if not already set."""
    env_file = os.path.expanduser("~/.chimere/.env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#") and not line.startswith("export "):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
                elif line.startswith("export "):
                    rest = line[7:]
                    if "=" in rest:
                        k, v = rest.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())

_load_env()


def _cache_key(query: str, count: int) -> str:
    return hashlib.md5(f"{query}:{count}".encode()).hexdigest()


def search(query: str, count: int = 5, lang: str = "fr-FR", cache_ttl: int | None = None) -> list[dict]:
    """Search Brave and return list of {title, snippet, url}.

    Args:
        query: Search query string
        count: Number of results (default 5)
        lang: UI language locale (default fr-FR)
        cache_ttl: Cache TTL in seconds (default 3600, use 300 for weather/news)
    """
    global _last_request_time
    ttl = cache_ttl if cache_ttl is not None else CACHE_TTL_DEFAULT

    # Check cache first
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{_cache_key(query, count)}.json"
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < ttl:
            try:
                return json.loads(cache_file.read_text())
            except (json.JSONDecodeError, OSError):
                pass  # corrupted cache, re-fetch

    key = os.environ.get("BRAVE_API_KEY", "")
    if not key:
        return []

    # Thread-safe rate limiting (Brave free tier: 1 req/s)
    with _rate_lock:
        global _last_request_time
        elapsed = time.time() - _last_request_time
        if elapsed < RATE_LIMIT_INTERVAL:
            time.sleep(RATE_LIMIT_INTERVAL - elapsed)
        _last_request_time = time.time()

    resp = requests.get(
        API_URL,
        headers={
            "X-Subscription-Token": key,
            "Accept": "application/json",
        },
        params={
            "q": query,
            "count": count,
            "search_lang": "fr",
            "ui_lang": lang,
        },
        timeout=10,
    )
    resp.raise_for_status()
    results = resp.json().get("web", {}).get("results", [])
    out = [
        {"title": r["title"], "snippet": r.get("description", ""), "url": r["url"]}
        for r in results[:count]
    ]

    # Save to cache
    try:
        cache_file.write_text(json.dumps(out, ensure_ascii=False))
    except OSError:
        pass  # non-critical

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brave Search CLI")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--count", type=int, default=5, help="Number of results")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--cache-ttl", type=int, default=None, help="Cache TTL in seconds (default 3600)")
    args = parser.parse_args()

    _load_env()
    results = search(args.query, args.count, cache_ttl=args.cache_ttl)

    if args.json:
        print(json.dumps(results, ensure_ascii=False))
    else:
        for r in results:
            print(f"- {r['title']}: {r['snippet'][:150]}")
