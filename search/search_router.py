#!/usr/bin/env python3
"""Smart Search Router — routes queries to Brave (fast) or Perplexica (deep).

Standalone CLI + importable module.

Components:
  - QueryClassifier: heuristic regex + length-based routing
  - BudgetManager: monthly Brave API budget with sliding daily limits
  - SearchCache: SQLite with exact + normalized + fuzzy token matching
  - SearchRouter: orchestrator with fallback cascade

Usage:
  search_router.py "what is the capital of France"           # auto-classify
  search_router.py "compare React vs Vue" --backend deep     # force deep
  search_router.py "weather Paris" --backend fast             # force fast
  search_router.py --budget                                  # show budget
  search_router.py --cache-stats                             # show cache stats
"""
import argparse
import calendar
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CHIMERE_HOME = Path.home() / ".chimere"
CACHE_DB = CHIMERE_HOME / "cache" / "search_router.db"
BUDGET_FILE = CHIMERE_HOME / "cache" / "brave_budget.json"
LOG_FILE = CHIMERE_HOME / "cache" / "search_router.log"
ENV_FILE = CHIMERE_HOME / ".env"

# Append bin dir so we can import the backends
sys.path.insert(0, str(CHIMERE_HOME / "bin"))

# ---------------------------------------------------------------------------
# Stopwords (FR + EN) for normalized matching
# ---------------------------------------------------------------------------
_STOPWORDS_RAW = (
    "a an the de du des le la les un une et and or ou is est are sont "
    "was were be been to a en dans for pour with avec of que qui quoi "
    "ce cette ces il elle on je tu nous vous ils elles ne pas plus "
    "what which how where when who why comment ou quand pourquoi quel "
    "quelle quels quelles it its this that these those do does did "
    "have has had will would can could shall should may might"
)
STOPWORDS = frozenset(_STOPWORDS_RAW.split())

# ---------------------------------------------------------------------------
# QueryClassifier
# ---------------------------------------------------------------------------
# Patterns that signal a DEEP research query
DEEP_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(compar\w+|versus|vs\.?)\b",
        r"\b(analyz\w+|analy[sz]\w+|évaluer?|evaluat\w+)\b",
        r"\b(expliqu\w+|explain\w*|mechanism\w*|mécanis\w+)\b",
        r"\b(research|recherche|étude|study|paper|article scientifique)\b",
        r"\b(comprehensive|thorough|detailed|in.depth|approfondi\w*|exhausti\w+)\b",
        r"\b(pros?\s+and\s+cons?|avantages?\s+et\s+inconvénients?)\b",
        r"\b(histor\w+|évolution|origins?|origines?)\b",
        r"\b(best\s+practices?|bonnes?\s+pratiques?|architecture|design\s+pattern)\b",
        r"\b(state.of.the.art|état\s+de\s+l.art|benchmark\w*)\b",
        r"\b(deep\s*search|deep\s*research|recherche\s+approfondie)\b",
        r"\b(thès[eé]|dissertation|revue\s+de\s+littérature|literature\s+review)\b",
        r"\b(implément\w+|implement\w+|how\s+to\s+build|comment\s+construire)\b",
    ]
]

# Patterns that signal a FAST factual query
FAST_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(météo|weather|temp[eé]rature|forecast|prévision)\b",
        r"\b(prix|price|stock|cours|bourse|cotation)\b",
        r"\b(define|définit?ion|what\s+is|qu.est.ce\s+qu|c.est\s+quoi)\b",
        r"\b(who\s+is|qui\s+est|when\s+did|quand\s+est)\b",
        r"\b(where\s+is|où\s+(est|se\s+trouve))\b",
        r"\b(convert|calculer?|calculate|combien)\b",
        r"\b(latest\s+news|dernières?\s+nouvelles?|actua?lité)\b",
        r"\b(score|résultat|result\b)\b",
        r"\b(heure|time\s+in|fuseau)\b",
        r"\b(recette|recipe|tuto|tutorial|how\s+to\s+\w+$)\b",
        r"\b(traduction|translate|tradui[st])\b",
    ]
]

# Temporal markers → short cache TTL
TEMPORAL_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(today|aujourd.hui|maintenant|now|current)\b",
        r"\b(latest|derniè\w*|dernier|récent\w*|recent)\b",
        r"\b(this\s+week|cette\s+semaine|ce\s+mois)\b",
        r"\b(météo|weather|forecast|prévision)\b",
        r"\b(score|match|résultat\s+de)\b",
        r"\b(bourse|stock|cours|prix)\b",
    ]
]


def classify_query(query: str) -> str:
    """Classify a query as 'fast' or 'deep'. Returns the classification string."""
    deep_score = sum(1 for p in DEEP_PATTERNS if p.search(query))
    fast_score = sum(1 for p in FAST_PATTERNS if p.search(query))

    # Prefer Perplexica (deep) by default — only use Brave for clear fast patterns
    deep_score += 1

    # Length heuristics
    words = query.split()
    word_count = len(words)
    if word_count > 20:
        deep_score += 2
    elif word_count > 12:
        deep_score += 1
    elif word_count < 5:
        fast_score += 1

    # Multiple questions → research
    if query.count("?") > 1:
        deep_score += 1

    if deep_score >= fast_score:
        return "deep"
    return "fast"


def is_temporal(query: str) -> bool:
    """Check if query references time-sensitive data."""
    return any(p.search(query) for p in TEMPORAL_PATTERNS)


# ---------------------------------------------------------------------------
# BudgetManager
# ---------------------------------------------------------------------------
class BudgetManager:
    """Track monthly Brave API usage with sliding daily budget."""

    MONTHLY_LIMIT = 2000
    RESERVE_THRESHOLD = 0.10   # force Perplexica below 10%
    THROTTLE_THRESHOLD = 0.30  # start preferring Perplexica below 30%

    def __init__(self, state_file: Path = BUDGET_FILE):
        self._file = state_file
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load()

    def _load(self) -> dict:
        month_key = time.strftime("%Y-%m")
        if self._file.exists():
            try:
                data = json.loads(self._file.read_text())
                if data.get("month") == month_key:
                    return data
            except (json.JSONDecodeError, OSError):
                pass
        return {"month": month_key, "used": 0, "daily_log": {}}

    def _save(self):
        self._file.write_text(json.dumps(self._state, ensure_ascii=False))

    @property
    def remaining(self) -> int:
        return max(0, self.MONTHLY_LIMIT - self._state["used"])

    @property
    def used(self) -> int:
        return self._state["used"]

    @property
    def daily_budget(self) -> int:
        """Dynamic daily budget based on remaining days."""
        now = time.localtime()
        days_in_month = calendar.monthrange(now.tm_year, now.tm_mon)[1]
        days_remaining = max(1, days_in_month - now.tm_mday + 1)
        return max(1, self.remaining // days_remaining)

    @property
    def today_used(self) -> int:
        today = time.strftime("%Y-%m-%d")
        return self._state.get("daily_log", {}).get(today, 0)

    def can_use(self) -> bool:
        """True if budget allows a Brave call."""
        return self.remaining > 0

    def should_prefer_perplexica(self) -> bool:
        """True when budget is critically low (<10% remaining)."""
        return self.remaining < (self.MONTHLY_LIMIT * self.RESERVE_THRESHOLD)

    def should_throttle(self) -> bool:
        """True when budget is getting low (<30% remaining)."""
        return self.remaining < (self.MONTHLY_LIMIT * self.THROTTLE_THRESHOLD)

    def record(self, count: int = 1):
        """Record Brave API usage."""
        self._state["used"] += count
        today = time.strftime("%Y-%m-%d")
        daily = self._state.setdefault("daily_log", {})
        daily[today] = daily.get(today, 0) + count
        self._save()

    def status(self) -> dict:
        return {
            "month": self._state["month"],
            "used": self.used,
            "remaining": self.remaining,
            "daily_budget": self.daily_budget,
            "today_used": self.today_used,
            "throttling": self.should_throttle(),
            "reserve_mode": self.should_prefer_perplexica(),
        }


# ---------------------------------------------------------------------------
# SearchCache (SQLite)
# ---------------------------------------------------------------------------
def _normalize_query(query: str) -> str:
    """Normalize query for cache matching."""
    q = query.lower().strip()
    q = re.sub(r"[''`]", "'", q)
    q = re.sub(r"[?!.,;:…]+$", "", q)        # trailing punctuation
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _strip_accents(s: str) -> str:
    """Remove diacritics: météo → meteo, français → francais."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def _tokenize(query: str) -> set[str]:
    """Tokenize, strip accents, remove stopwords for fuzzy matching."""
    q = _strip_accents(_normalize_query(query))
    q = re.sub(r"[^\w\s-]", " ", q)
    tokens = {w for w in q.split() if w not in STOPWORDS and len(w) > 1}
    return tokens


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class SearchCache:
    """SQLite-backed 3-layer cache: exact → normalized → fuzzy token match."""

    FUZZY_THRESHOLD = 0.75

    def __init__(self, db_path: Path = CACHE_DB):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_raw TEXT NOT NULL,
                query_norm TEXT NOT NULL,
                query_tokens TEXT NOT NULL,
                backend TEXT NOT NULL,
                results TEXT NOT NULL,
                answer TEXT,
                ttl INTEGER NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_norm ON cache(query_norm, backend)"
        )
        self._conn.commit()

    def lookup(self, query: str, backend: str = None) -> dict | None:
        """Look up cache. Returns {results, answer, backend, cache_layer} or None."""
        norm = _normalize_query(query)
        now = time.time()

        # Layer 1: exact normalized match
        where = "query_norm = ? AND (created_at + ttl) > ?"
        params: list = [norm, now]
        if backend:
            where += " AND backend = ?"
            params.append(backend)

        row = self._conn.execute(
            f"SELECT results, answer, backend, 'exact' as layer FROM cache "
            f"WHERE {where} ORDER BY created_at DESC LIMIT 1",
            params,
        ).fetchone()
        if row:
            return self._unpack(row)

        # Layer 2: fuzzy token match (scan recent entries)
        query_tokens = _tokenize(query)
        if not query_tokens:
            return None

        where2 = "(created_at + ttl) > ?"
        params2: list = [now]
        if backend:
            where2 += " AND backend = ?"
            params2.append(backend)

        rows = self._conn.execute(
            f"SELECT results, answer, backend, query_tokens FROM cache "
            f"WHERE {where2} ORDER BY created_at DESC LIMIT 200",
            params2,
        ).fetchall()

        best_sim = 0.0
        best_row = None
        for r in rows:
            cached_tokens = set(json.loads(r[3]))
            sim = _jaccard(query_tokens, cached_tokens)
            if sim > best_sim:
                best_sim = sim
                best_row = r

        if best_row and best_sim >= self.FUZZY_THRESHOLD:
            return {
                "results": json.loads(best_row[0]),
                "answer": best_row[1],
                "backend": best_row[2],
                "cache_layer": f"fuzzy({best_sim:.2f})",
            }

        return None

    def store(self, query: str, backend: str, results: list, answer: str | None,
              ttl: int):
        """Store search results in cache."""
        norm = _normalize_query(query)
        tokens_json = json.dumps(sorted(_tokenize(query)))
        self._conn.execute(
            "INSERT INTO cache (query_raw, query_norm, query_tokens, backend, "
            "results, answer, ttl, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (query, norm, tokens_json, backend, json.dumps(results, ensure_ascii=False),
             answer, ttl, time.time()),
        )
        self._conn.commit()

    def stats(self) -> dict:
        now = time.time()
        total = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        alive = self._conn.execute(
            "SELECT COUNT(*) FROM cache WHERE (created_at + ttl) > ?", (now,)
        ).fetchone()[0]
        by_backend = dict(self._conn.execute(
            "SELECT backend, COUNT(*) FROM cache WHERE (created_at + ttl) > ? "
            "GROUP BY backend", (now,)
        ).fetchall())
        return {"total_entries": total, "alive": alive, "by_backend": by_backend}

    def prune(self):
        """Remove expired entries."""
        now = time.time()
        deleted = self._conn.execute(
            "DELETE FROM cache WHERE (created_at + ttl) <= ?", (now,)
        ).rowcount
        self._conn.commit()
        self._conn.execute("VACUUM")
        return deleted

    @staticmethod
    def _unpack(row) -> dict:
        return {
            "results": json.loads(row[0]),
            "answer": row[1],
            "backend": row[2],
            "cache_layer": row[3],
        }


# ---------------------------------------------------------------------------
# SearchRouter
# ---------------------------------------------------------------------------
class SearchRouter:
    """Smart search orchestrator with classification, caching, budget, fallback."""

    def __init__(self):
        self.cache = SearchCache()
        self.budget = BudgetManager()
        self._brave = None
        self._perplexica = None
        self._searxng = None

    def _get_brave(self):
        if self._brave is None:
            import brave_search
            self._brave = brave_search
        return self._brave

    def _get_perplexica(self):
        if self._perplexica is None:
            import perplexica_search
            self._perplexica = perplexica_search
        return self._perplexica

    def _get_searxng(self):
        if self._searxng is None:
            import searxng_search
            self._searxng = searxng_search
        return self._searxng

    def _get_deep_fetch(self):
        if not hasattr(self, "_deep_fetch") or self._deep_fetch is None:
            try:
                import web_deep_fetch
                self._deep_fetch = web_deep_fetch
            except ImportError:
                self._deep_fetch = False  # mark as unavailable
        return self._deep_fetch if self._deep_fetch is not False else None

    def search(self, query: str, force_backend: str | None = None,
               perplexica_mode: str = "speed",
               perplexica_sources: list[str] | None = None,
               no_cache: bool = False) -> dict:
        """Route and execute a search query.

        Args:
            query: The search query
            force_backend: "fast"/"brave" or "deep"/"perplexica" to override classifier
            perplexica_mode: "speed", "balanced", or "quality" (default "balanced")
            perplexica_sources: Perplexica source types (default ["web"])
            no_cache: Skip cache lookup (still stores result)

        Returns:
            {query, backend, classification, cached, cache_layer, results, answer,
             elapsed, budget_remaining}
        """
        start = time.time()

        # --- Classification ---
        classification = classify_query(query)
        temporal = is_temporal(query)

        if force_backend:
            backend = "perplexica" if force_backend in ("deep", "perplexica") else "brave"
        else:
            backend = "perplexica" if classification == "deep" else "brave"

        # --- Budget override ---
        budget_note = None
        if backend == "brave" and self.budget.should_prefer_perplexica():
            backend = "perplexica"
            budget_note = "reserve_mode"
        elif backend == "brave" and self.budget.should_throttle():
            # Probabilistic throttle: the lower the budget, the more likely to use Perplexica
            ratio = self.budget.remaining / self.budget.MONTHLY_LIMIT
            # At 30% → 0% chance to redirect, at 10% → 66% chance
            import random
            if random.random() > (ratio / self.budget.THROTTLE_THRESHOLD):
                backend = "perplexica"
                budget_note = "throttled"

        # --- Cache lookup ---
        cache_hit = None
        if not no_cache:
            cache_hit = self.cache.lookup(query, backend=backend)
            # If no hit on selected backend, try the other one
            if not cache_hit:
                other = "perplexica" if backend == "brave" else "brave"
                cache_hit = self.cache.lookup(query, backend=other)

        if cache_hit:
            return {
                "query": query,
                "backend": cache_hit["backend"],
                "classification": classification,
                "cached": True,
                "cache_layer": cache_hit["cache_layer"],
                "results": cache_hit["results"],
                "answer": cache_hit.get("answer"),
                "elapsed": time.time() - start,
                "budget_remaining": self.budget.remaining,
                "budget_note": budget_note,
            }

        # --- Execute search with fallback ---
        results, answer, actual_backend = self._execute_with_fallback(
            query, backend, perplexica_mode, perplexica_sources or ["web"]
        )

        # --- Compute TTL ---
        if temporal:
            ttl = 900  # 15 min for time-sensitive
        elif actual_backend == "perplexica":
            ttl = 86400  # 24h for deep research (expensive to reproduce)
        else:
            ttl = 3600  # 1h for Brave snippets

        # --- Store in cache ---
        if results:
            self.cache.store(query, actual_backend, results, answer, ttl)

        return {
            "query": query,
            "backend": actual_backend,
            "classification": classification,
            "cached": False,
            "cache_layer": None,
            "results": results,
            "answer": answer,
            "elapsed": time.time() - start,
            "budget_remaining": self.budget.remaining,
            "budget_note": budget_note,
        }

    def _execute_with_fallback(
        self, query: str, primary: str, perp_mode: str, perp_sources: list[str]
    ) -> tuple[list, str | None, str]:
        """Execute search on primary backend, fallback to secondary on failure.

        Returns: (results, answer, actual_backend)
        """
        if primary == "brave":
            return self._try_brave_then_perplexica(query, perp_mode, perp_sources)
        else:
            return self._try_perplexica_then_brave(query, perp_mode, perp_sources)

    def _try_brave_then_perplexica(
        self, query: str, perp_mode: str, perp_sources: list[str]
    ) -> tuple[list, str | None, str]:
        """Try Brave first, fallback to Perplexica."""
        if not self.budget.can_use():
            # No budget, go straight to Perplexica
            return self._call_perplexica(query, perp_mode, perp_sources) + ("perplexica",)

        try:
            results = self._call_brave(query)
            if results:
                return results, None, "brave"
        except Exception as e:
            self._log(f"Brave failed: {e}")

        # Fallback to Perplexica
        try:
            results, answer = self._call_perplexica(query, perp_mode, perp_sources)
            if results:
                return results, answer, "perplexica"
        except Exception as e:
            self._log(f"Perplexica fallback also failed: {e}")

        return [], None, "none"

    @staticmethod
    def _detect_lang(query: str) -> str:
        """Simple heuristic: if query looks English/code, use 'en', else 'fr'."""
        # Code patterns: camelCase, snake_case, ::, ->, common keywords
        if re.search(r'[a-z][A-Z]|_[a-z]|::|->|fn |impl |async |await |import |def |class |const |let |var |function |return |std::', query):
            return "en"
        # Mostly ASCII + common English words → English
        ascii_ratio = sum(1 for c in query if c.isascii()) / max(len(query), 1)
        en_markers = re.findall(r'\b(?:how|what|why|where|which|the|with|from|into|about|using|between|does|should|would|could|workaround|performance|implementation|error|bug|fix|deploy|config)\b', query, re.I)
        fr_markers = re.findall(r'\b(?:comment|pourquoi|quand|quel|quelle|avec|dans|pour|des|les|une|est|sont|faire|entre|aussi|cette|exercice|protocole|traitement|muscle|rééducation|culture|sol|plante)\b', query, re.I)
        if len(en_markers) > len(fr_markers) and ascii_ratio > 0.9:
            return "en"
        return "fr"

    @staticmethod
    def _detect_domain(query: str) -> str:
        """Detect query domain for SearXNG category selection."""
        q = query.lower()
        if re.search(r'[a-z][A-Z]|_[a-z]|::|->|fn |impl |async |await |import |def |class |const |let |var |function |return |std::|api |docker|kubernetes|nginx|linux|git |npm |pip |cargo |cmake', q):
            return "code"
        if re.search(r'\b(?:bug|error|stack\s*overflow|compile|runtime|deploy|debug|refactor|library|framework|package|module|crate)\b', q, re.I):
            return "code"
        if re.search(r'\b(?:étude|study|pubmed|doi|méta.analyse|meta.analysis|essai\s+clinique|clinical\s+trial|revue\s+systématique|systematic\s+review|p\s*[<>=]\s*0|cohorte|cohort|randomis|RCT|evidence.based|arxiv|paper|journal|abstract|cite|citation)\b', q, re.I):
            return "science"
        if re.search(r'\b(?:muscle|tendin|ligament|articulai|ménisque|rééducation|kiné|orthop|pathologi|syndrome|fascia|biomécani|neurolog|physiolog|anatomie|anatomy|clinique|médical|pathophysiologie)\b', q, re.I):
            return "science"
        if re.search(r'\b(?:agro|culture[s]?\b|sol[s]?\b|plante|semence|compost|engrais|rotation|couvert|maraîch|phytosanitaire|mycorhize|azote|phosphore|rendement|irrigat|pédolog)\b', q, re.I):
            return "science"
        return "general"

    def _try_perplexica_then_brave(
        self, query: str, perp_mode: str, perp_sources: list[str]
    ) -> tuple[list, str | None, str]:
        """Try SearXNG multi-category search, then Brave fallback.

        Research-grade: searches 'general' + domain-specific category
        (science for medical/agro, it for code), merges and deduplicates.
        """
        lang = self._detect_lang(query)
        domain = self._detect_domain(query)
        try:
            searxng = self._get_searxng()
            results = searxng.search(query, count=10, lang=lang, categories="general")
            # Add domain-specific academic/code results
            extra_cat = {"science": "science", "code": "it"}.get(domain)
            if extra_cat:
                extra = searxng.search(query, count=10, lang=lang, categories=extra_cat)
                seen_urls = {r["url"] for r in results if r.get("url")}
                for r in extra:
                    if r.get("url") and r["url"] not in seen_urls:
                        results.append(r)
                        seen_urls.add(r["url"])
                self._log(f"SearXNG multi-cat: general={len(results)-len([r for r in extra if r.get('url') and r['url'] not in seen_urls])}, {extra_cat}={len(extra)}, merged={len(results)}")
            if results:
                # Deep fetch: extract full content from top URLs
                deep = self._get_deep_fetch()
                if deep and domain in ("science", "code"):
                    try:
                        chunks = deep.deep_fetch(query, results,
                                                 max_pages=3, top_chunks=5)
                        if chunks:
                            # Prepend enriched chunks to results
                            enriched = [{
                                "title": f"[Deep] {c['title']}",
                                "url": c["url"],
                                "content": c["text"],
                                "relevance_score": c.get("relevance_score", 0),
                            } for c in chunks]
                            self._log(f"Deep fetch: {len(chunks)} chunks from {len(set(c['url'] for c in chunks))} pages")
                            # CRAG: evaluate and filter enriched chunks
                            try:
                                from crag_evaluator import crag_pipeline
                                enriched_before = len(enriched)
                                enriched = crag_pipeline(query, enriched, min_score=0.3)
                                self._log(f"CRAG: {enriched_before} → {len(enriched)} chunks after filtering")
                                if len(enriched) < 2:
                                    # Not enough relevant content — trigger Brave fallback
                                    self._log("CRAG: insufficient relevant chunks, adding Brave results")
                                    if self.budget.can_use():
                                        brave_results = self._call_brave(query)
                                        enriched.extend(brave_results)
                            except ImportError:
                                pass  # crag_evaluator not available, skip
                            results = enriched + results
                    except Exception as e:
                        self._log(f"Deep fetch failed (non-blocking): {e}")
                return results, None, "searxng"
            self._log("SearXNG returned empty results")
        except Exception as e:
            self._log(f"SearXNG failed: {e}")

        # Fallback to Brave
        if self.budget.can_use():
            try:
                results = self._call_brave(query)
                if results:
                    return results, None, "brave"
            except Exception as e:
                self._log(f"Brave fallback also failed: {e}")

        return [], None, "none"

    def _call_brave(self, query: str) -> list[dict]:
        """Call Brave Search API and record budget usage."""
        brave = self._get_brave()
        ttl = 300 if is_temporal(query) else 3600
        results = brave.search(query, count=5, cache_ttl=ttl)
        if results:
            self.budget.record(1)
        # Normalize output format to {title, url, content}
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("snippet", r.get("content", "")),
            }
            for r in results
        ]

    def _call_perplexica(self, query: str, mode: str,
                         sources: list[str]) -> tuple[list, str | None]:
        """Call Perplexica and return (results, synthesized_answer)."""
        perp = self._get_perplexica()
        raw = perp.search(query, mode=mode, sources=sources, use_cache=False)

        answer = None
        results = []
        for r in raw:
            if r.get("title") == "Perplexica Answer":
                answer = r.get("content", "")
            else:
                results.append(r)

        return results, answer

    def _log(self, message: str):
        """Append to log file."""
        try:
            LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(LOG_FILE, "a") as f:
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{ts}] {message}\n")
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Environment loader
# ---------------------------------------------------------------------------
def _load_env():
    """Load secrets from ~/.chimere/.env."""
    if ENV_FILE.exists():
        with open(ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:]
                if "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Smart Search Router — Brave (fast) + Perplexica (deep)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s "météo Paris"                         # auto → Brave (fast)
  %(prog)s "compare React vs Vue en 2026"        # auto → Perplexica (deep)
  %(prog)s "quantum computing" --backend deep     # force deep research
  %(prog)s "what is DNS" --backend fast            # force Brave
  %(prog)s --budget                               # show Brave budget
  %(prog)s --cache-stats                          # show cache statistics
  %(prog)s --cache-prune                          # remove expired cache entries
""",
    )
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument(
        "--backend", choices=["fast", "brave", "deep", "perplexica", "sota", "research"],
        help="Force backend (sota = single-pass SOTA pipeline, research = iterative deep research)",
    )
    parser.add_argument(
        "--depth", choices=["quick", "standard", "deep"], default="standard",
        help="Depth for --backend sota (default: standard)",
    )
    parser.add_argument(
        "--profile", choices=["quick", "standard", "deep", "marathon"],
        default="standard",
        help="Profile for --backend research (default: standard)",
    )
    parser.add_argument(
        "--mode", choices=["speed", "balanced", "quality"], default="balanced",
        help="Perplexica mode when using deep (default: balanced)",
    )
    parser.add_argument(
        "--sources", nargs="+", default=["web"],
        choices=["web", "academic", "discussions"],
        help="Perplexica sources (default: web)",
    )
    parser.add_argument("--no-cache", action="store_true", help="Skip cache lookup")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--budget", action="store_true", help="Show Brave API budget")
    parser.add_argument("--cache-stats", action="store_true", help="Show cache stats")
    parser.add_argument("--cache-prune", action="store_true", help="Prune expired cache")
    parser.add_argument("--classify-only", action="store_true",
                        help="Only classify query, don't search")

    args = parser.parse_args()

    # --- Utility commands ---
    if args.budget:
        budget = BudgetManager()
        s = budget.status()
        if args.json:
            print(json.dumps(s))
        else:
            print(f"Month:          {s['month']}")
            print(f"Used:           {s['used']}/{BudgetManager.MONTHLY_LIMIT}")
            print(f"Remaining:      {s['remaining']}")
            print(f"Daily budget:   {s['daily_budget']}")
            print(f"Today used:     {s['today_used']}")
            print(f"Throttling:     {'YES' if s['throttling'] else 'no'}")
            print(f"Reserve mode:   {'YES' if s['reserve_mode'] else 'no'}")
        return

    if args.cache_stats:
        cache = SearchCache()
        s = cache.stats()
        if args.json:
            print(json.dumps(s))
        else:
            print(f"Total entries:  {s['total_entries']}")
            print(f"Alive (valid):  {s['alive']}")
            for b, c in s.get("by_backend", {}).items():
                print(f"  {b}: {c}")
        return

    if args.cache_prune:
        cache = SearchCache()
        n = cache.prune()
        print(f"Pruned {n} expired entries.")
        return

    if not args.query:
        parser.error("query is required (unless using --budget/--cache-stats/--cache-prune)")

    # --- Classify only ---
    if args.classify_only:
        cls = classify_query(args.query)
        temporal = is_temporal(args.query)
        if args.json:
            print(json.dumps({"classification": cls, "temporal": temporal}))
        else:
            backend = "Perplexica (deep)" if cls == "deep" else "Brave (fast)"
            print(f"Classification: {cls} → {backend}")
            print(f"Temporal:       {'yes' if temporal else 'no'}")
        return

    # --- Search ---
    _load_env()

    # SOTA backend: delegate to deep_search_sota.py full pipeline
    if args.backend == "sota":
        import sys as _sys
        _bin_dir = os.path.dirname(os.path.abspath(__file__))
        if _bin_dir not in _sys.path:
            _sys.path.insert(0, _bin_dir)
        try:
            import deep_search_sota
            sota_result = deep_search_sota.deep_search(
                args.query, domain="auto", depth=args.depth
            )
            if args.json:
                print(json.dumps(sota_result, ensure_ascii=False))
            else:
                elapsed = sota_result.get("elapsed", 0)
                depth = sota_result.get("depth", args.depth)
                n_chunks = len(sota_result.get("chunks", []))
                synthesis = sota_result.get("answer", sota_result.get("synthesis", ""))
                queries = sota_result.get("expanded_queries", sota_result.get("queries_used", []))
                chunks = sota_result.get("chunks", [])
                print(f"--- SOTA 2026 | depth={depth} | {elapsed:.1f}s | {n_chunks} chunks ---")
                print()
                if queries:
                    print(f"Queries: {', '.join(queries[:3])}")
                    print()
                if synthesis:
                    print("SYNTHESIS:")
                    print(synthesis[:4000])
                    if len(synthesis) > 4000:
                        print(f"... ({len(synthesis)} chars total)")
                    print()
                if chunks:
                    seen_urls = set()
                    sources = []
                    for c in chunks:
                        url = c.get("url", "")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            sources.append(c)
                    n = min(10, len(sources))
                    print(f"SOURCES ({len(seen_urls)} unique, showing {n}):")
                    for i, s in enumerate(sources[:n], 1):
                        title = s.get("title", "?")[:70]
                        url = s.get("url", "")
                        text = s.get("text", "")[:200]
                        print(f"  {i}. {title}")
                        if url:
                            print(f"     {url}")
                        if text:
                            print(f"     {text}")
            return
        except Exception as exc:
            print(f"[SOTA] Failed ({exc}), falling back to deep backend", file=sys.stderr)
            args.backend = "deep"

    # RESEARCH backend: delegate to deep_research.py iterative agent
    if args.backend == "research":
        _bin_dir = os.path.dirname(os.path.abspath(__file__))
        if _bin_dir not in sys.path:
            sys.path.insert(0, _bin_dir)
        try:
            import deep_research
            result = deep_research.deep_research(
                args.query, domain="auto", profile=args.profile
            )
            if args.json:
                out = {k: v for k, v in result.items() if k != "report"}
                print(json.dumps(out, ensure_ascii=False, indent=2))
            else:
                print(f"\n{'='*70}")
                print(f"CHIMERE DEEP RESEARCH — {result['profile'].upper()}")
                print(f"Query    : {result['query']}")
                print(f"Domain   : {result['domain']}")
                print(f"Iters    : {result['iterations']}")
                print(f"Sources  : {result['total_sources']}")
                print(f"Chunks   : {result['total_chunks']}")
                print(f"Time     : {result['elapsed']}s")
                print(f"Report   : {result['report_path']}")
                print(f"{'='*70}")
        except Exception as exc:
            print(f"[RESEARCH] Failed: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
        return

    router = SearchRouter()
    result = router.search(
        args.query,
        force_backend=args.backend,
        perplexica_mode=args.mode,
        perplexica_sources=args.sources,
        no_cache=args.no_cache,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False))
        return

    # Human-readable output
    elapsed = result["elapsed"]
    backend = result["backend"]
    cached = result["cached"]
    layer = result["cache_layer"]
    answer = result.get("answer")
    results = result["results"]
    cls = result["classification"]
    budget_note = result.get("budget_note", "")

    cache_info = f" (cache: {layer})" if cached else ""
    note = f" [{budget_note}]" if budget_note else ""
    print(f"--- {backend}{note} | {cls} | {elapsed:.1f}s{cache_info} | "
          f"budget: {result['budget_remaining']} ---")
    print()

    if answer:
        print("ANSWER:")
        print(answer[:3000])
        if len(answer) > 3000:
            print(f"... ({len(answer)} chars total)")
        print()

    if results:
        n = min(10, len(results))
        print(f"SOURCES ({len(results)} total, showing {n}):")
        for i, r in enumerate(results[:n], 1):
            title = r.get("title", "?")[:70]
            url = r.get("url", "")
            content = r.get("content", "")
            is_deep = title.startswith("[Deep]")
            # Show more content for deep-fetched results
            max_content = 800 if is_deep else 120
            print(f"  {i}. {title}")
            if url:
                print(f"     {url}")
            if content:
                print(f"     {content[:max_content]}")
    elif not answer:
        print("No results found.")


if __name__ == "__main__":
    main()
