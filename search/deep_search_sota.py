#!/usr/bin/env python3
"""Deep Search SOTA 2026 — pipeline de recherche web avancé.

Architecture :
  1. Query Expansion     — Qwen génère 3-5 requêtes diversifiées
  2. Parallel Search     — SearXNG + Brave simultanément (ThreadPoolExecutor)
  3. RRF Fusion          — Reciprocal Rank Fusion pour merger les résultats
  4. Deep Fetch          — trafilatura + chunking (max_pages URLs)
  5. Neural Reranking    — Qwen3-Embedding-0.6B (sentence_transformers)
  6. CRAG Filter         — filtrage crédibilité avec seuil adapté
  7. LLM Synthesis       — réponse Qwen citée et structurée

Usage CLI:
  deep_search_sota.py "query"
  deep_search_sota.py "query" --domain medical --depth deep --json

Importable:
  from deep_search_sota import deep_search
  result = deep_search("query", domain="auto", depth="standard")
"""

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

_BIN = Path.home() / ".chimere" / "bin"
if str(_BIN) not in sys.path:
    sys.path.insert(0, str(_BIN))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LLAMA_URL  = "http://127.0.0.1:8084/v1/chat/completions"
LLM_TIMEOUT = 120
CACHE_DIR  = Path.home() / ".chimere" / ".sota_cache"
CACHE_TTL  = {"quick": 3600, "standard": 7200, "deep": 21600}

DEPTH_CONFIG = {
    "quick":    {"n_queries": 2, "max_pages": 2, "top_chunks": 5,  "max_results": 8},
    "standard": {"n_queries": 3, "max_pages": 3, "top_chunks": 8,  "max_results": 12},
    "deep":     {"n_queries": 5, "max_pages": 5, "top_chunks": 12, "max_results": 20},
}

# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _llm_call(messages: list, max_tokens: int = 512, temperature: float = 0.3,
              nothink: bool = True) -> Optional[str]:
    """Call Qwen3.5 via think-router."""
    payload = {
        "model": "qwen3.5-35b",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "top_k": 20,
    }
    if nothink:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    data = json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(
        LLAMA_URL, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
            body = json.loads(resp.read())
        content = body["choices"][0]["message"]["content"].strip()
        # Strip residual <think>…</think>
        import re
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content
    except Exception as exc:
        print(f"[SOTA] LLM error: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Step 1: Query Expansion
# ---------------------------------------------------------------------------

def expand_query(query: str, domain: str = "general", n: int = 3) -> list[str]:
    """Generate n diverse search queries from the original query.

    Falls back to [query] on LLM failure.
    """
    domain_hint = {
        "medical": "médical, physiologique, clinique",
        "code": "programmation, technique, implémentation",
        "agronomy": "agricole, botanique, agronomique",
        "general": "général",
    }.get(domain, "général")

    prompt = (
        f"Tu es un expert en recherche documentaire.\n"
        f"Génère exactement {n} requêtes de recherche web DIFFÉRENTES et COMPLÉMENTAIRES "
        f"pour explorer tous les aspects de cette question.\n"
        f"Domaine : {domain_hint}\n"
        f"Question originale : \"{query}\"\n\n"
        f"Règles :\n"
        f"- Chaque requête doit couvrir un angle différent (ex: définition, mécanisme, benchmark, cas d'usage)\n"
        f"- Varie les formulations (anglais/français si pertinent)\n"
        f"- Évite les redondances\n"
        f"- Format : JSON array de strings uniquement, sans explication\n\n"
        f'Exemple: ["requête 1", "requête 2", "requête 3"]'
    )
    # max_tokens=4096: think_router FORCE_THINK=True consomme des tokens pour le think block
    content = _llm_call([{"role": "user", "content": prompt}], max_tokens=4096)
    if not content:
        return [query]

    import re
    match = re.search(r"\[.*?\]", content, re.DOTALL)
    if not match:
        return [query]
    try:
        queries = json.loads(match.group(0))
        if isinstance(queries, list) and queries:
            # Always include original query
            result = [query] + [str(q).strip() for q in queries if str(q).strip() != query]
            return result[:n + 1]
    except json.JSONDecodeError:
        pass
    return [query]


# ---------------------------------------------------------------------------
# Pipeline cache
# ---------------------------------------------------------------------------

def _sota_cache_get(query: str, depth: str) -> Optional[dict]:
    try:
        key = hashlib.md5((query + depth).encode()).hexdigest()
        f = CACHE_DIR / f"{key}.json"
        if not f.exists():
            return None
        data = json.loads(f.read_text(encoding="utf-8"))
        if time.time() - data.get("ts", 0) > CACHE_TTL.get(depth, 7200):
            f.unlink(missing_ok=True)
            return None
        return data.get("result")
    except Exception:
        return None


def _sota_cache_set(query: str, depth: str, result: dict):
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        key = hashlib.md5((query + depth).encode()).hexdigest()
        (CACHE_DIR / f"{key}.json").write_text(
            json.dumps({"ts": time.time(), "result": result}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Step 0: Local knowledge (ChromaDB)
# ---------------------------------------------------------------------------

def query_local_knowledge(query: str, domain: str, max_results: int = 5) -> list[dict]:
    """Query local ChromaDB knowledge base. Returns web-compatible chunks."""
    try:
        import knowledge_rag_query
        collection = domain if domain in ("medical", "code") else "auto"
        results = knowledge_rag_query.query_rag(
            query, collection=collection, max_results=max_results,
            min_score=0.25, rerank=False,
        )
        out = []
        for r in results:
            coll = r.get("collection", domain)
            fp   = r.get("file_path", "")
            out.append({
                "text":            r.get("text", ""),
                "title":           r.get("title", ""),
                "url":             f"local://{coll}/{fp}",
                "relevance_score": float(r.get("score", 0.0)),
                "_rag":            True,
                "score":           float(r.get("score", 0.0)),
            })
        return out
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Source diversity
# ---------------------------------------------------------------------------

def enforce_source_diversity(chunks: list[dict], max_per_url_ratio: float = 0.5) -> list[dict]:
    """Ensure no single URL represents more than max_per_url_ratio of chunks.

    Only applies if there are chunks from multiple distinct URLs.
    Uses round-robin per URL to redistribute excess chunks.
    """
    if not chunks:
        return []
    unique_urls = {c.get("url", "") for c in chunks}
    if len(unique_urls) <= 1:
        return chunks  # single source, nothing to diversify
    max_per_url = max(1, int(len(chunks) * max_per_url_ratio))

    # Group by URL preserving order
    by_url: dict[str, list[dict]] = {}
    for c in chunks:
        url = c.get("url", "")
        by_url.setdefault(url, []).append(c)

    # Round-robin across URLs, up to max_per_url each
    result: list[dict] = []
    idx = 0
    while True:
        added = False
        for url, cs in by_url.items():
            if idx < len(cs) and idx < max_per_url:
                result.append(cs[idx])
                added = True
        idx += 1
        if not added or idx >= max_per_url:
            break

    return result


# ---------------------------------------------------------------------------
# Contradiction detection
# ---------------------------------------------------------------------------

def detect_contradictions(query: str, chunks: list[dict]) -> Optional[str]:
    """Ask Qwen to identify contradictions between top chunks. Returns text or None."""
    if len(chunks) < 2:
        return None
    parts = [f"Source {i+1}: {c.get('text','')[:300]}" for i, c in enumerate(chunks[:6])]
    prompt = (
        f"Question : {query}\n\n"
        + "\n\n".join(parts)
        + "\n\nY a-t-il des contradictions importantes entre ces sources concernant la question ci-dessus ?\n"
        "Si oui, liste-les brièvement (2-3 lignes max).\n"
        "Si non, réponds exactement : NULL"
    )
    resp = _llm_call([{"role": "user", "content": prompt}], max_tokens=4096, temperature=0.3)
    if not resp or "NULL" in resp.upper():
        return None
    return resp.strip()


# ---------------------------------------------------------------------------
# Step 2: Parallel Search (SearXNG + Brave)
# ---------------------------------------------------------------------------

def _search_brave(query: str, count: int = 10) -> list[dict]:
    """Brave API search. Returns {title, url, content, _source: brave, _rank}."""
    try:
        import brave_search
        raw = brave_search.search(query, count=count)
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""),
             "content": r.get("snippet", r.get("content", "")),
             "_source": "brave", "_query": query}
            for r in raw
        ]
    except Exception as exc:
        print(f"[SOTA] Brave error: {exc}", file=sys.stderr)
        return []


def _search_searxng(query: str, count: int = 10, domain: str = "general") -> list[dict]:
    """SearXNG multi-category search."""
    try:
        import searxng_search
        from search_router import SearchRouter
        lang = SearchRouter._detect_lang(query)
        cat_map = {"science": "science", "code": "it", "medical": "science"}
        extra_cat = cat_map.get(domain)

        results = searxng_search.search(query, count=count, lang=lang, categories="general")
        seen_urls = {r["url"] for r in results if r.get("url")}
        if extra_cat:
            extra = searxng_search.search(query, count=count, lang=lang, categories=extra_cat)
            for r in extra:
                if r.get("url") and r["url"] not in seen_urls:
                    results.append(r)
                    seen_urls.add(r["url"])

        out = []
        for r in results:
            out.append({
                "title": r.get("title", ""), "url": r.get("url", ""),
                "content": r.get("content", r.get("snippet", "")),
                "_source": "searxng", "_query": query,
            })
        return out
    except Exception as exc:
        print(f"[SOTA] SearXNG error: {exc}", file=sys.stderr)
        return []


def _search_academic(query: str, count: int = 8) -> list[dict]:
    """SearXNG science category, filtered to academic domains."""
    _ACADEMIC_DOMAINS = (
        "arxiv.org", "pubmed.ncbi", "hal.science", "cairn.info",
        "scholar.google", "ncbi.nlm.nih.gov", "pmc", "semanticscholar.org",
        "researchgate.net", "sciencedirect.com", "springer.com", "nature.com",
    )
    try:
        import searxng_search
        results = searxng_search.search(query, count=count * 2, lang="en", categories="science")
        out = []
        for r in results:
            url = r.get("url", "")
            if any(d in url for d in _ACADEMIC_DOMAINS):
                out.append({
                    "title":    r.get("title", ""),
                    "url":      url,
                    "content":  r.get("content", r.get("snippet", "")),
                    "_source":  "academic",
                    "_query":   query,
                })
                if len(out) >= count:
                    break
        return out
    except Exception:
        return []


def parallel_search(queries: list[str], domain: str = "general",
                    max_results_per_query: int = 10,
                    include_academic: bool = False) -> list[list[dict]]:
    """Run all queries against SearXNG + Brave (+ academic if requested) in parallel.

    Returns list-of-lists: one list per query (merged results).
    """
    tasks = []
    for q in queries:
        tasks.append(("brave",   q, domain))
        tasks.append(("searxng", q, domain))
    if include_academic:
        # One academic query per unique query (not per domain variant)
        for q in queries[:2]:  # limit to 2 to avoid too many requests
            tasks.append(("academic", q, domain))

    def _run(task):
        backend, q, dom = task
        if backend == "brave":
            return q, backend, _search_brave(q, max_results_per_query)
        elif backend == "academic":
            return q, backend, _search_academic(q, max_results_per_query)
        else:
            return q, backend, _search_searxng(q, max_results_per_query, dom)

    raw_by_query: dict[str, list[dict]] = {q: [] for q in queries}

    with ThreadPoolExecutor(max_workers=min(len(tasks), 6)) as ex:
        futs = {ex.submit(_run, t): t for t in tasks}
        for fut in as_completed(futs, timeout=30):
            try:
                q, backend, results = fut.result(timeout=30)
                raw_by_query[q].extend(results)
            except Exception as exc:
                print(f"[SOTA] parallel search task failed: {exc}", file=sys.stderr)

    # Return one list per query, maintaining order
    return [raw_by_query[q] for q in queries]


# ---------------------------------------------------------------------------
# Step 3: RRF Fusion
# ---------------------------------------------------------------------------

def rrf_fusion(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion across multiple result lists.

    Score = sum(1 / (k + rank_i)) for each list where the doc appears.
    Higher = more relevant / appears in more lists at higher positions.
    """
    scores: dict[str, float] = {}
    docs: dict[str, dict]    = {}

    for result_list in result_lists:
        seen_urls: set[str] = set()
        rank = 0
        for doc in result_list:
            url = doc.get("url", "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            rank += 1
            scores[url] = scores.get(url, 0.0) + 1.0 / (k + rank)
            if url not in docs:
                docs[url] = doc

    merged = []
    for url, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        doc = dict(docs[url])
        doc["rrf_score"] = round(score, 5)
        doc["relevance_score"] = round(score * 20, 3)  # normalize to ~0-1 range
        merged.append(doc)

    return merged


# ---------------------------------------------------------------------------
# Step 4: Deep Fetch + Neural Reranking
# ---------------------------------------------------------------------------

_cross_encoder = None

def _get_cross_encoder():
    """Lazy-load cross-encoder reranker (CPU, ~300MB, first call ~5s)."""
    global _cross_encoder
    if _cross_encoder is None:
        import os
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder = CrossEncoder(
                "Alibaba-NLP/gte-reranker-modernbert-base",
                max_length=8192, device="cpu"
            )
            print("[SOTA] Cross-encoder reranker loaded (CPU)", file=sys.stderr)
        except Exception as e:
            print(f"[SOTA] Cross-encoder unavailable: {e}", file=sys.stderr)
    return _cross_encoder


def cross_encoder_rerank(query: str, chunks: list[dict], top_k: int = 8) -> list[dict]:
    """Rerank chunks using cross-encoder (CPU). Returns top-k sorted by relevance."""
    ce = _get_cross_encoder()
    if not ce or not chunks:
        return chunks[:top_k]

    pairs = [(query, c.get("text", c.get("content", ""))[:4000]) for c in chunks]
    try:
        scores = ce.predict(pairs)
        for i, c in enumerate(chunks):
            c["ce_score"] = float(scores[i])
        chunks.sort(key=lambda c: c.get("ce_score", 0), reverse=True)
        print(f"[SOTA] Cross-encoder reranked {len(chunks)} → top {top_k} "
              f"(best={chunks[0].get('ce_score',0):.3f})", file=sys.stderr)
    except Exception as e:
        print(f"[SOTA] Cross-encoder error: {e}", file=sys.stderr)
    return chunks[:top_k]


def fetch_and_rerank(query: str, results: list[dict],
                     max_pages: int = 3, top_chunks: int = 8,
                     use_neural_rerank: bool = True) -> list[dict]:
    """Fetch top URLs, extract content, chunk, then cross-encoder rerank.

    Always uses cross-encoder reranking (CPU, ~200ms for 20 chunks).
    Falls back to keyword-based ranking if cross-encoder unavailable.
    """
    try:
        import web_deep_fetch
        chunks = web_deep_fetch.deep_fetch(
            query, results,
            max_pages=max_pages,
            top_chunks=top_chunks * 2,  # Fetch more, let cross-encoder pick best
            use_reranking=False,  # Skip old embedding reranker
        )
        # Cross-encoder rerank (CPU, ~200ms)
        if chunks:
            chunks = cross_encoder_rerank(query, chunks, top_k=top_chunks)
        return chunks
    except Exception as exc:
        print(f"[SOTA] deep_fetch error: {exc}", file=sys.stderr)
        return [
            {"text": r.get("content", r.get("snippet", ""))[:1000],
             "title": r.get("title", ""),
             "url": r.get("url", ""),
             "relevance_score": r.get("rrf_score", 0.5)}
            for r in results[:top_chunks]
            if r.get("content") or r.get("snippet")
        ]


# ---------------------------------------------------------------------------
# Step 5: CRAG Filter
# ---------------------------------------------------------------------------

def crag_filter(query: str, chunks: list[dict], threshold: float = 0.20) -> list[dict]:
    """CRAG filter with web-adapted threshold and score normalization."""
    if not chunks:
        return []

    # Normalize scores
    normalized = []
    for c in chunks:
        nc = dict(c)
        if "score" not in nc:
            nc["score"] = float(nc.get("relevance_score", nc.get("rrf_score", 0.45)))
        normalized.append(nc)

    try:
        from crag_evaluator import crag_pipeline
        result = crag_pipeline(query, normalized, threshold=threshold, verbose=False)
        if not result:
            print("[SOTA] CRAG filtered everything, using fallback", file=sys.stderr)
            return sorted(normalized, key=lambda x: x.get("score", 0), reverse=True)[:15]
        return result
    except Exception as exc:
        print(f"[SOTA] CRAG error: {exc}", file=sys.stderr)
        return sorted(normalized, key=lambda x: x.get("score", 0), reverse=True)[:15]


# ---------------------------------------------------------------------------
# Step 6: LLM Synthesis
# ---------------------------------------------------------------------------

def synthesize(query: str, chunks: list[dict], domain: str = "general",
               contradiction_note: Optional[str] = None) -> str:
    """Generate a grounded, cited answer from chunks via Qwen."""
    if not chunks:
        return "Aucun résultat pertinent trouvé."

    # Build context with numbered sources
    context_parts = []
    for i, c in enumerate(chunks[:12], 1):
        title = c.get("title", f"Source {i}")[:80]
        url   = c.get("url", "")
        text  = c.get("text", c.get("content", ""))[:600]
        ref   = f"[{i}] {title}"
        if url and not url.startswith("local://"):
            ref += f" ({url})"
        elif url.startswith("local://"):
            ref += " [Base locale]"
        context_parts.append(f"{ref}\n{text}")

    context = "\n\n---\n\n".join(context_parts)

    domain_inst = {
        "medical":  "Utilise un langage médical précis. Cite les études si disponibles.",
        "code":     "Fournis des exemples de code si pertinent. Cite les documentations.",
        "agronomy": "Contextualise pour la pratique agricole. Cite les sources agronomiques.",
        "general":  "Sois précis et factuel. Cite tes sources.",
    }.get(domain, "Sois précis et factuel. Cite tes sources.")

    contradiction_section = ""
    if contradiction_note:
        contradiction_section = f"\n⚠️ CONTRADICTIONS DÉTECTÉES entre sources : {contradiction_note}\n"

    prompt = (
        f"Tu es un assistant de recherche expert. Réponds à la question suivante "
        f"en utilisant UNIQUEMENT les sources fournies ci-dessous.\n"
        f"{domain_inst}\n"
        f"Cite les sources avec [1], [2], etc. Mentionne les incertitudes si nécessaire.\n"
        f"{contradiction_section}\n"
        f"Question : {query}\n\n"
        f"Sources :\n{context}\n\n"
        f"Réponse structurée et citée :"
    )

    content = _llm_call(
        [{"role": "user", "content": prompt}],
        # max_tokens=8192: think_router min(8192, 16384)=8192, laisse de la place après le think block
        # nothink ignoré par think_router FORCE_THINK=True, d'où le max_tokens élevé
        max_tokens=8192, temperature=0.5, nothink=True,
    )
    return content or "Synthèse indisponible (erreur LLM)."


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def deep_search(query: str, domain: str = "auto", depth: str = "standard",
                use_cache: bool = True) -> dict:
    """Full SOTA 2026 deep search pipeline.

    Étapes :
      0. ChromaDB local (en parallèle avec 2)
      1. Query Expansion
      2. Parallel Search (SearXNG + Brave + académique si deep/medical)
      3. RRF Fusion (web + RAG local)
      4. Deep Fetch + Neural Reranking
      5. Source Diversity enforcement
      6. CRAG Filter
      7. Contradiction Detection (standard/deep)
      8. LLM Synthesis

    Returns: {query, domain, depth, expanded_queries, total_raw_results,
              rag_chunks_found, chunks_used, unique_sources, answer,
              contradictions, chunks, sources, elapsed, steps_timing}
    """
    start = time.time()
    timings: dict = {}

    # --- Cache check ---
    if use_cache:
        cached = _sota_cache_get(query, depth)
        if cached is not None:
            print(f"[SOTA] cache hit ({depth})", file=sys.stderr)
            cached["elapsed"] = 0
            return cached

    config = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["standard"])

    # Auto-detect domain
    if domain == "auto":
        try:
            from search_router import SearchRouter
            domain = SearchRouter._detect_domain(query)
        except Exception:
            domain = "general"

    print(f"[SOTA] Query: {query!r} | domain={domain} | depth={depth}", file=sys.stderr)

    # --- Step 1: Query Expansion ---
    t0 = time.time()
    queries = expand_query(query, domain=domain, n=config["n_queries"])
    timings["query_expansion"] = round(time.time() - t0, 2)
    print(f"[SOTA] {len(queries)} queries expanded", file=sys.stderr)

    # --- Steps 0 + 2 in parallel: ChromaDB + Web Search ---
    include_academic = (depth == "deep") or (domain in ("medical", "agronomy"))
    t0 = time.time()

    rag_chunks: list[dict] = []
    result_lists: list[list[dict]] = []

    def _run_web():
        return parallel_search(
            queries, domain=domain,
            max_results_per_query=config["max_results"],
            include_academic=include_academic,
        )

    def _run_rag():
        return query_local_knowledge(query, domain=domain, max_results=5)

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_web = ex.submit(_run_web)
        fut_rag = ex.submit(_run_rag)
        result_lists = fut_web.result(timeout=40)
        rag_chunks   = fut_rag.result(timeout=10)

    timings["search_and_rag"] = round(time.time() - t0, 2)
    total_raw_web = sum(len(r) for r in result_lists)
    print(f"[SOTA] {total_raw_web} web results | {len(rag_chunks)} RAG chunks", file=sys.stderr)

    # --- Step 3: RRF Fusion (web only first, then merge RAG) ---
    t0 = time.time()
    fused_web = rrf_fusion(result_lists)
    # Include RAG chunks as an extra source list in RRF so they get scored
    if rag_chunks:
        fused = rrf_fusion([fused_web, rag_chunks])
    else:
        fused = fused_web
    timings["rrf_fusion"] = round(time.time() - t0, 2)
    print(f"[SOTA] {len(fused)} unique results after RRF", file=sys.stderr)

    # --- Step 4: Deep Fetch + Neural Reranking ---
    use_neural = (depth == "deep")
    t0 = time.time()
    web_chunks = fetch_and_rerank(
        query, fused_web,
        max_pages=config["max_pages"],
        top_chunks=config["top_chunks"],
        use_neural_rerank=use_neural,
    )
    # Fallback if fetch returned nothing
    if not web_chunks:
        web_chunks = [
            {"text": r.get("content", "")[:800], "title": r.get("title", ""),
             "url": r.get("url", ""), "relevance_score": r.get("rrf_score", 0.5)}
            for r in fused_web[:config["top_chunks"]] if r.get("content")
        ]
    timings["deep_fetch_rerank"] = round(time.time() - t0, 2)
    print(f"[SOTA] {len(web_chunks)} web chunks fetched", file=sys.stderr)

    # Merge RAG (prepend) + web chunks, deduplicate
    import re as _re
    chunks: list[dict] = []
    seen_keys: set = set()
    for c in (rag_chunks + web_chunks):
        key = (c.get("url", "")[:80]) + _re.sub(r"\s+", " ", c.get("text", "")[:60])
        if key not in seen_keys:
            seen_keys.add(key)
            chunks.append(c)

    # --- Step 5: Source Diversity ---
    chunks = enforce_source_diversity(chunks, max_per_url_ratio=0.5)
    print(f"[SOTA] {len(chunks)} chunks after diversity enforcement", file=sys.stderr)

    # --- Step 6: CRAG Filter ---
    t0 = time.time()
    threshold = 0.25 if rag_chunks else 0.20
    filtered = crag_filter(query, chunks, threshold=threshold)
    timings["crag_filter"] = round(time.time() - t0, 2)
    print(f"[SOTA] {len(filtered)} chunks after CRAG", file=sys.stderr)

    # --- Step 7: Contradiction Detection ---
    contradictions: Optional[str] = None
    if depth in ("standard", "deep") and len(filtered) >= 2:
        t0 = time.time()
        contradictions = detect_contradictions(query, filtered)
        timings["contradictions"] = round(time.time() - t0, 2)
        if contradictions:
            print(f"[SOTA] Contradictions detected", file=sys.stderr)

    # --- Step 8: Synthesis ---
    t0 = time.time()
    answer = synthesize(query, filtered, domain=domain, contradiction_note=contradictions)
    timings["synthesis"] = round(time.time() - t0, 2)

    # --- Build sources ---
    seen_urls: set = set()
    sources = []
    for c in filtered:
        url = c.get("url", "").strip()
        title = c.get("title", "").strip()
        if url and url not in seen_urls:
            seen_urls.add(url)
            label = "[LOCAL]" if url.startswith("local://") else ""
            sources.append({"title": title or url, "url": url, "label": label})

    elapsed = round(time.time() - start, 2)
    print(f"[SOTA] Done in {elapsed}s | {len(sources)} sources | {len(rag_chunks)} RAG", file=sys.stderr)

    result = {
        "query":            query,
        "domain":           domain,
        "depth":            depth,
        "expanded_queries": queries,
        "total_raw_results": total_raw_web + len(rag_chunks),
        "rag_chunks_found": len(rag_chunks),
        "chunks_used":      len(filtered),
        "unique_sources":   len(sources),
        "answer":           answer,
        "contradictions":   contradictions,
        "chunks":           filtered,
        "sources":          sources,
        "elapsed":          elapsed,
        "steps_timing":     timings,
    }

    if use_cache:
        _sota_cache_set(query, depth, result)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Deep Search SOTA 2026",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s "transformer neural networks 2025 benchmark"
  %(prog)s "lombalgie chronique traitement kinésithérapie" --domain medical --depth deep
  %(prog)s "Qwen3.5 MoE quantization benchmark" --depth quick --json
""",
    )
    parser.add_argument("query", help="Search query")
    parser.add_argument("--domain", default="auto",
                        choices=["auto", "general", "medical", "code", "agronomy"])
    parser.add_argument("--depth", default="standard",
                        choices=["quick", "standard", "deep"])
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--no-synthesis", action="store_true",
                        help="Skip LLM synthesis (faster, return chunks only)")
    parser.add_argument("--no-cache", action="store_true", help="Bypass cache")
    args = parser.parse_args()

    # Load env for API keys
    env_file = Path.home() / ".chimere" / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:]
                if "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

    result = deep_search(args.query, domain=args.domain, depth=args.depth,
                         use_cache=not args.no_cache)

    if args.no_synthesis:
        result.pop("answer", None)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # Human-readable
    print(f"\n{'='*70}")
    print(f"DEEP SEARCH SOTA 2026 — {result['depth'].upper()}")
    print(f"Query  : {result['query']}")
    print(f"Domain : {result['domain']} | Time: {result['elapsed']}s")
    rag_n = result.get("rag_chunks_found", 0)
    print(f"Queries: {len(result['expanded_queries'])} | Sources: {result['unique_sources']} | Chunks: {result['chunks_used']} | RAG: {rag_n}")
    print(f"Timings: {result['steps_timing']}")
    print(f"{'='*70}\n")

    if result.get("contradictions"):
        print("⚠️  CONTRADICTIONS:")
        print(result["contradictions"])
        print()

    if result.get("answer"):
        print("RÉPONSE SYNTHÉTISÉE:")
        print(result["answer"])
        print()

    if result["sources"]:
        print(f"SOURCES ({len(result['sources'])}):")
        for i, s in enumerate(result["sources"][:10], 1):
            label = s.get("label", "")
            print(f"  [{i}] {label} {s['title'][:70]}")
            if s["url"] and not s["url"].startswith("local://"):
                print(f"       {s['url']}")


if __name__ == "__main__":
    main()
