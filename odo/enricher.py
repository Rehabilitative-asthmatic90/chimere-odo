#!/usr/bin/env python3
"""
ODO Enricher — Context enrichment layer.

Called between CLASSIFY and GENERATE. For each route, runs the appropriate
tool (search, RAG, CSV analysis, IoC lookup) and injects results into the
system prompt BEFORE forwarding to the LLM.

Absorbs the enrichment logic from message_router.py (port 8083).
All enrichment is async-safe (subprocess with timeout).
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dynamic_engram import build_dynamic_engram

# ── Paths ────────────────────────────────────────────────────────────────────

BIN = Path.home() / ".openclaw/bin"
VENV_RAG = Path.home() / ".openclaw/venvs/kine-rag/bin/python3"
PYTHON = str(VENV_RAG) if VENV_RAG.exists() else sys.executable

# ── Pattern detection (absorbed from message_router.py) ──────────────────────

CSV_PATH_RE = re.compile(r'/[\w./-]+\.csv\b')

IPV4_RE = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
HASH_SHA256_RE = re.compile(r'\b[a-fA-F0-9]{64}\b')
IOC_KEYWORDS = re.compile(
    r'\b(?:ioc|IoC|indicat(?:eur|or)|menace|threat|malware|malicious'
    r'|suspicious|suspect|analyse.*(?:ip|adresse|hash|domaine))\b', re.I
)

FACTUAL_RE = re.compile(
    r'\b(?:population|combien d\'habitants|superficie|pib|gdp'
    r'|capitale de|president de|nombre de|quel(?:le)? est (?:la|le|l\')'
    r'|en (?:quelle année|quel pays))\b', re.I
)

DEEP_RESEARCH_RE = re.compile(
    r'\b(?:méta.analyse|meta.analysis|revue\s+systématique|systematic\s+review'
    r'|état\s+de\s+l.art|state.of.the.art|recherche\s+approfondie|deep\s+research'
    r'|compare[rz]?\s+(?:les|différent|plusieurs)|efficacité\s+(?:de|des|du)'
    r'|evidence.based|niveau\s+de\s+preuve|benchmark|exhausti[fv])\b', re.I
)

URL_YOUTUBE_RE = re.compile(r'https?://(www\.)?(youtube\.com/watch|youtu\.be/)\S+')

INGEST_CMD_RE = re.compile(r'/ingest\s+(https?://\S+)', re.I)


# ── Tool runners ─────────────────────────────────────────────────────────────

def _run_script(args: list, timeout: int = 60, max_chars: int = 8000) -> str | None:
    """Run a script with timeout. Returns stdout or None."""
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()[:max_chars]
        if r.returncode != 0:
            print(f"[enricher] script failed ({r.returncode}): {' '.join(str(a) for a in args[:3])} "
                  f"stderr={r.stderr[:200]}", file=sys.stderr, flush=True)
    except subprocess.TimeoutExpired:
        print(f"[enricher] timeout ({timeout}s): {' '.join(str(a) for a in args[:3])}",
              file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[enricher] error: {e}", file=sys.stderr, flush=True)
    return None


def run_rag_search(query: str, collection: str = "auto", max_results: int = 3) -> str | None:
    """Query ChromaDB knowledge base. Forces CPU to avoid GPU OOM."""
    script = BIN / "knowledge_rag_query.py"
    if not script.exists():
        return None
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "SENTENCE_TRANSFORMERS_HOME": str(Path.home() / ".cache/sentence_transformers")}
    try:
        r = subprocess.run(
            [PYTHON, str(script), query, "--collection", collection,
             "--max", str(max_results), "--no-rerank"],
            capture_output=True, text=True, timeout=60, env=env
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()[:6000]
        if r.returncode != 0:
            print(f"[enricher] RAG failed ({r.returncode}): {r.stderr[:200]}",
                  file=sys.stderr, flush=True)
    except subprocess.TimeoutExpired:
        print("[enricher] RAG timeout (60s)", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[enricher] RAG error: {e}", file=sys.stderr, flush=True)
    return None


def run_web_search(query: str, depth: str = "quick") -> str | None:
    """Run deep_search_sota.py with specified depth."""
    script = BIN / "deep_search_sota.py"
    if not script.exists():
        return None
    return _run_script([PYTHON, str(script), query, "--depth", depth],
                       timeout=120, max_chars=6000)


def run_csv_analysis(csv_path: str) -> str | None:
    """Run analyze_csv.sh on a CSV file."""
    script = BIN / "analyze_csv.sh"
    if not script.exists() or not os.path.exists(csv_path):
        return None
    return _run_script(["bash", str(script), csv_path], timeout=240)


def run_cyberbro(observable: str) -> str | None:
    """Run CyberBro IoC analysis."""
    tools_dir = Path.home() / ".openclaw/agents/cyber/tools"
    code = (
        f"import sys, json, time; sys.path.insert(0, '{tools_dir}'); "
        f"from cyberbro_tool import analyze_observables, get_analysis_results; "
        f"r = analyze_observables('{observable}'); "
        f"time.sleep(15) if not r.get('error') else None; "
        f"print(json.dumps(get_analysis_results(r['analysis_id']) if not r.get('error') else r, indent=2, default=str))"
    )
    return _run_script([sys.executable, "-c", code], timeout=60, max_chars=3000)


def run_research(query: str, depth: str = "standard") -> str | None:
    """Run deep_search_sota.py for research enrichment.

    Previously called research_orchestrator.py (600s timeout, too slow for
    inline enrichment). Now uses deep_search_sota.py directly which has its
    own internal caching and is bounded to ~60s for standard depth.
    """
    script = BIN / "deep_search_sota.py"
    if not script.exists():
        return None
    return _run_script([PYTHON, str(script), query, "--depth", depth],
                       timeout=90, max_chars=12000)


# ── Dynamic Engram (Option A: system prompt injection) ───────────────────────

def inject_dynamic_engram_context(
    search_results_text: str,
    query: str,
    top_k: int = 20,
) -> str | None:
    """Build dynamic engram from web search text, extract top n-gram predictions.

    Option A approach: instead of per-request Engram path override in chimere-server,
    extract the most confident n-gram predictions and format them as system prompt
    context that complements RAG injection with token-level factual associations.

    Args:
        search_results_text: raw text output from web search (deep_search_sota)
        query: user's original query
        top_k: number of top predictions to extract

    Returns:
        Formatted text for system prompt injection, or None if build failed.
    """
    if not search_results_text or len(search_results_text) < 100:
        return None

    # Convert raw text into the list-of-dicts format expected by build_dynamic_engram
    # Split into paragraph-sized chunks as pseudo search results
    paragraphs = [p.strip() for p in search_results_text.split("\n\n") if p.strip()]
    chunks = []
    for para in paragraphs:
        if len(para) > 30:  # Skip tiny fragments
            chunks.append({"text": para, "title": "", "url": ""})

    if len(chunks) < 2:
        return None

    # Build the .engr file from search result text
    engr_path = build_dynamic_engram(chunks, query)
    if not engr_path or not os.path.exists(engr_path):
        return None

    # Load the .engr table and extract top predictions for the query
    try:
        # Import locally to avoid circular imports and keep startup fast
        sys.path.insert(0, str(BIN))
        from engram_query import EngramTable, load_tokenizer, format_token

        table = EngramTable(engr_path)
        tokenizer = load_tokenizer()

        # Tokenize the query
        encoding = tokenizer.encode(query, add_special_tokens=False)
        token_ids = list(encoding.ids)

        if len(token_ids) < table.order:
            return None

        # Collect all predictions across n-gram windows of the query
        all_predictions: list[tuple[str, str, float]] = []  # (context_str, predicted_str, prob)

        for i in range(len(token_ids) - table.order + 1):
            window = token_ids[i:i + table.order]
            predictions = table.lookup(window)
            if not predictions:
                continue

            # Build human-readable context string from the window
            context_str = tokenizer.decode(window).strip()

            # Take top-3 predictions per window
            for tok_id, prob in predictions[:3]:
                if prob < 0.05:  # Skip low-confidence predictions
                    break
                predicted_str = format_token(tokenizer, tok_id)
                all_predictions.append((context_str, predicted_str, prob))

        if not all_predictions:
            return None

        # Sort by confidence and deduplicate
        all_predictions.sort(key=lambda x: -x[2])
        seen = set()
        unique_preds = []
        for ctx, pred, prob in all_predictions:
            key = (ctx, pred)
            if key not in seen:
                seen.add(key)
                unique_preds.append((ctx, pred, prob))
            if len(unique_preds) >= top_k:
                break

        if not unique_preds:
            return None

        # Format as system prompt context
        lines = []
        for ctx, pred, prob in unique_preds:
            lines.append(f"  \"{ctx}\" -> \"{pred}\" ({prob:.0%})")

        engram_text = (
            "## Factual associations from sources\n\n"
            "The following token-level patterns were extracted from search results. "
            "Use these to ground factual claims:\n\n"
            + "\n".join(lines)
        )

        print(f"[DYNAMIC_ENGRAM] Injected {len(unique_preds)} predictions "
              f"for query: {query[:60]!r}", flush=True)

        return engram_text

    except Exception as e:
        print(f"[DYNAMIC_ENGRAM] Injection failed: {e}", file=sys.stderr, flush=True)
        return None


# ── Detection helpers ────────────────────────────────────────────────────────

def detect_csv(text: str) -> str | None:
    """Extract CSV path from message."""
    m = CSV_PATH_RE.search(text)
    return m.group(0) if m else None


def detect_ioc(text: str) -> tuple | None:
    """Detect IoC observable. Returns (type, value) or None."""
    has_kw = IOC_KEYWORDS.search(text) is not None
    m = IPV4_RE.search(text)
    if m and (has_kw or not m.group(0).startswith(('127.', '192.168.', '10.'))):
        return ("ip", m.group(0))
    m = HASH_SHA256_RE.search(text)
    if m and (has_kw or len(text.strip()) < 80):
        return ("hash", m.group(0))
    return None


def needs_web_search(text: str) -> bool:
    """Check if query would benefit from web search."""
    return bool(FACTUAL_RE.search(text))


def needs_deep_research(text: str) -> bool:
    """Check if query warrants full research pipeline."""
    return bool(DEEP_RESEARCH_RE.search(text))


def detect_ingest(text: str) -> str | None:
    """Detect /ingest URL command."""
    m = INGEST_CMD_RE.search(text)
    return m.group(1) if m else None


# ── Few-shot example store ───────────────────────────────────────────────────

FEW_SHOT_DIR = Path(__file__).parent / "few_shot"
_few_shot_cache: dict[str, list] = {}
_few_shot_mtime: dict[str, float] = {}


def _load_few_shot(route_id: str) -> list:
    """Load few-shot examples for a route. Cached with mtime check."""
    json_path = FEW_SHOT_DIR / f"{route_id}.json"
    if not json_path.exists():
        return []
    try:
        mtime = json_path.stat().st_mtime
        if route_id in _few_shot_cache and mtime <= _few_shot_mtime.get(route_id, 0):
            return _few_shot_cache[route_id]
        with open(json_path) as f:
            examples = json.load(f)
        _few_shot_cache[route_id] = examples
        _few_shot_mtime[route_id] = mtime
        return examples
    except Exception:
        return []


def find_few_shot(route_id: str, user_text: str, max_examples: int = 1) -> list:
    """Find the most relevant few-shot examples.

    Uses semantic FAISS matching (tier 1) with keyword fallback (tier 2).
    Returns list of {"input": ..., "output": ...} dicts.
    """
    # Tier 1: Semantic FAISS matching (~5ms)
    try:
        from semantic_fewshot import find_semantic_fewshot
        semantic = find_semantic_fewshot(user_text, route_id=route_id,
                                         max_examples=max_examples)
        if semantic:
            return [{"input": s["input"], "output": s["output"],
                     "_source": "semantic"} for s in semantic]
    except Exception:
        pass  # Fall through to keyword matching

    # Tier 2: Keyword matching (0ms fallback)
    examples = _load_few_shot(route_id)
    if not examples:
        return []

    query_tokens = set(user_text.lower().split())

    scored = []
    for ex in examples:
        # Score by tag match + input keyword overlap
        tags = set(t.lower() for t in ex.get("tags", []))
        input_tokens = set(ex.get("input", "").lower().split())
        tag_hits = len(query_tokens & tags)
        token_hits = len(query_tokens & input_tokens)
        score = tag_hits * 3 + token_hits  # tags are worth more
        if score > 0:
            scored.append((score, ex))

    scored.sort(key=lambda x: -x[0])
    return [{"input": s[1]["input"], "output": s[1]["output"]}
            for s in scored[:max_examples]]


# ── Main enrichment function ────────────────────────────────────────────────

def enrich(payload: dict, route_id: str, user_text: str,
           pipeline: dict) -> tuple[dict, dict]:
    """Enrich payload with context from tools + few-shot examples.

    Returns (enriched_payload, enrich_info).
    enrich_info contains: tools_used, enrich_ms, context_chars.
    """
    t0 = time.time()
    result = dict(payload)
    tools_used = []
    context_parts = []

    # Few-shot examples — ONLY for ambiguous/open queries, NOT factual recall
    # Ablation showed few-shot on kine factual questions DEGRADES recall (63%→11%)
    # because the injected example confuses the model into mimicking the example
    # instead of answering the specific question.
    is_factual = bool(re.search(
        r"(quels? (sont|est)|combien|liste|critères?|protocole|score|dosage|posologie)",
        user_text, re.IGNORECASE
    ))
    few_shots = [] if is_factual else find_few_shot(route_id, user_text, max_examples=1)
    if few_shots:
        # Inject as user/assistant message pair BEFORE the current user message
        msgs = result.get("messages", [])
        # Find the last user message index
        last_user_idx = None
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") == "user":
                last_user_idx = i
                break
        if last_user_idx is not None:
            for fs in reversed(few_shots):
                msgs.insert(last_user_idx, {"role": "assistant", "content": fs["output"]})
                msgs.insert(last_user_idx, {"role": "user", "content": fs["input"]})
        tools_used.append(f"few_shot({len(few_shots)})")

    # Get enrichment config from pipeline YAML
    enrich_cfg = pipeline.get("enrich", {})
    if not isinstance(enrich_cfg, dict):
        enrich_cfg = {}

    # Auto-detect enrichment needs based on route + content
    do_rag = enrich_cfg.get("rag", route_id in ("kine", "research", "cyber"))
    do_deep = needs_deep_research(user_text) and route_id == "research"
    rag_collection = enrich_cfg.get("rag_collection", "auto")

    # Confidence-based RAG trigger: probe the model's uncertainty
    # If model is uncertain (high entropy / hedge phrases) → enable web search
    explicit_web = enrich_cfg.get("web")
    if explicit_web is True:
        do_web = True
        web_depth = "standard" if route_id == "research" else "quick"
    elif explicit_web is False:
        # Pipeline says no web — only probe confidence for queries that
        # explicitly ask for recent/updated info (avoid 15s overhead on every request)
        needs_recent = bool(re.search(
            r"(2024|2025|2026|récent|dernier|nouveau|dernières|actuali|mise à jour|latest|recent|updated)",
            user_text, re.IGNORECASE
        ))
        if needs_recent and len(user_text) >= 40:
            try:
                from confidence_rag_trigger import probe_confidence
                probe = probe_confidence(user_text, max_tokens=48, timeout=10)
                if not probe["confident"]:
                    do_web = True
                    web_depth = "standard" if probe["recommendation"] == "deep_rag" else "quick"
                    tools_used.append(f"confidence_probe({probe['mean_entropy']:.2f})")
                else:
                    do_web = False
                    web_depth = "skip"
            except Exception:
                do_web = False
                web_depth = "skip"
        else:
            do_web = False
            web_depth = "skip"
    else:
        do_web = explicit_web if explicit_web is not None else (
            route_id == "research" or needs_web_search(user_text)
        )
        web_depth = "standard" if route_id == "research" else "quick"

    # CSV detection (any route)
    csv_path = detect_csv(user_text)
    if csv_path:
        csv_out = run_csv_analysis(csv_path)
        if csv_out:
            context_parts.append(f"## Analyse CSV ({csv_path})\n\n{csv_out}")
            tools_used.append("csv_analysis")

    # IoC detection (any route, but prioritize cyber)
    ioc = detect_ioc(user_text)
    if ioc:
        ioc_type, ioc_val = ioc
        cb_out = run_cyberbro(ioc_val)
        if cb_out:
            context_parts.append(f"## Analyse CTI ({ioc_type}: {ioc_val})\n\n{cb_out}")
            tools_used.append("cyberbro")

    # Ingest detection
    ingest_url = detect_ingest(user_text)
    if ingest_url:
        # Don't enrich — just flag for the caller to handle
        pass

    # Deep research (slow, only for explicit research queries)
    if do_deep:
        research_out = run_research(user_text, depth="standard")
        if research_out:
            context_parts.append(f"## Recherche approfondie\n\n{research_out}")
            tools_used.append("research_orchestrator")
            do_rag = False  # research_orchestrator already includes RAG
            do_web = False
    else:
        # Parallel RAG + web search
        futures = {}
        web_text = None  # Capture for dynamic engram
        with ThreadPoolExecutor(max_workers=2) as pool:
            if do_rag:
                futures["rag"] = pool.submit(run_rag_search, user_text, rag_collection)
            if do_web:
                futures["web"] = pool.submit(run_web_search, user_text, web_depth)

            for key, future in futures.items():
                try:
                    timeout = 120 if key == "web" and route_id == "research" else 60
                    out = future.result(timeout=timeout)
                    if out:
                        if key == "rag":
                            context_parts.append(f"## Base de connaissances\n\n{out}")
                            tools_used.append("knowledge_rag")
                        elif key == "web":
                            context_parts.append(f"## Recherche web\n\n{out}")
                            tools_used.append("web_search")
                            web_text = out
                except Exception:
                    pass

        # Dynamic Engram injection (quality/ultra modes only)
        do_engram = enrich_cfg.get("dynamic_engram", False)
        if do_engram and web_text:
            engram_context = inject_dynamic_engram_context(web_text, user_text)
            if engram_context:
                context_parts.append(engram_context)
                tools_used.append("dynamic_engram")

    # Inject context into system prompt
    if context_parts:
        context_block = "\n\n---\n\n".join(context_parts)

        # Build reasoning instructions based on route
        if route_id == "kine":
            instructions = (
                "Utilise les sources ci-dessus pour fonder ta réponse. "
                "Identifie les recommandations HAS de grade A/B. "
                "Cite [Source: titre] pour chaque fait."
            )
        elif route_id == "research":
            instructions = (
                "Analyse les sources ci-dessus. Recoupe entre sources multiples. "
                "Cite [Source: titre/auteur] pour chaque fait. "
                "Mentionne les contradictions s'il y en a."
            )
        elif route_id == "cyber":
            instructions = (
                "Analyse les résultats CTI ci-dessus. Map sur MITRE ATT&CK. "
                "Classifie par sévérité. Propose des contre-mesures."
            )
        else:
            instructions = (
                "Utilise les sources ci-dessus pour enrichir ta réponse. "
                "Cite tes sources."
            )

        enrichment = f"{context_block}\n\n## Instructions\n\n{instructions}"

        # Inject as system message content (append to existing)
        msgs = result.get("messages", [])
        if msgs and msgs[0].get("role") == "system":
            msgs[0]["content"] = f"{msgs[0]['content']}\n\n{enrichment}"
        else:
            msgs.insert(0, {"role": "system", "content": enrichment})

    enrich_ms = int((time.time() - t0) * 1000)
    total_chars = sum(len(p) for p in context_parts)

    return result, {
        "tools_used": tools_used,
        "enrich_ms": enrich_ms,
        "context_chars": total_chars,
    }
