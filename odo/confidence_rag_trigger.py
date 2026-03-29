"""confidence_rag_trigger.py — Trigger RAG/web search based on model confidence

Instead of deciding BEFORE generation if RAG is needed (keyword heuristics),
this module monitors the model's response DURING streaming and triggers
retrieval when confidence drops.

Two strategies:

1. PRE-GENERATION (fast, heuristic):
   - Send a SHORT probe to the model (max 64 tokens, no-think)
   - Analyze logprobs: if entropy > threshold → model is uncertain → trigger RAG
   - This is a "should we search?" check, ~2s overhead

2. POST-GENERATION (accurate, slower):
   - After initial generation, check if response contains hedge phrases
   - If hedging detected → search and regenerate with context
   - This is the "reflection" approach (like Self-RAG)

Integration: called from ODO enricher to decide web search depth.
"""

from __future__ import annotations

import http.client
import json
import math
import os
import re
import time
from typing import Optional

CHIMERE_URL = os.environ.get("CHIMERE_BACKEND", "http://127.0.0.1:8081")

# Hedge phrases indicating model uncertainty (French + English)
HEDGE_PHRASES = [
    # French
    r"je ne (suis|sais) pas (certain|sûr|sure)",
    r"il (est|serait) (possible|probable) que",
    r"je n.ai pas (d.informations?|de données)",
    r"à ma connaissance",
    r"je ne (dispose|possède) pas",
    r"(difficile|impossible) de (dire|confirmer|vérifier)",
    r"(mes|ma) (données|connaissance|information)s? (s.arrêtent|datent|limitées?)",
    r"je (recommande|suggère) de (vérifier|consulter)",
    r"(sans|en l.absence de) (sources?|données|preuves?)",
    r"(actuellement )?(impossible|pas possible) de (fournir|donner|confirmer)",
    r"(pas|aucune) (de |d.)?(informations?|données) (récentes?|disponibles?|spécifiques?|à jour)",
    r"(je |il )?(ne |n.)?(?:est |suis )?(pas en mesure|dans l.incapacité)",
    # English
    r"I('m| am) not (sure|certain)",
    r"I don.t have (specific|recent|updated) (information|data|knowledge)",
    r"my (knowledge|training) (cutoff|data|has a cutoff)",
    r"(may|might|could) (not be|be inaccurate)",
    r"I (cannot|can.t) (verify|confirm)",
    r"(please|you should) (verify|check|consult)",
]
_HEDGE_RE = re.compile("|".join(HEDGE_PHRASES), re.IGNORECASE)


def probe_confidence(
    user_text: str,
    max_tokens: int = 64,
    timeout: float = 15.0,
) -> dict:
    """Send a quick probe to the model and analyze confidence via logprobs.

    Returns:
      {
        "confident": bool,         # True if model seems confident
        "mean_entropy": float,     # Average normalized entropy across tokens
        "hedge_detected": bool,    # True if response contains hedge phrases
        "probe_text": str,         # The short probe response
        "probe_ms": float,         # Time taken
        "recommendation": str,     # "skip_rag" | "quick_rag" | "deep_rag"
      }
    """
    from urllib.parse import urlparse
    parsed = urlparse(CHIMERE_URL)

    payload = {
        "model": "qwen3.5",
        "messages": [{"role": "user", "content": user_text}],
        "max_tokens": max_tokens,
        "temperature": 0.0,  # Greedy for confidence measurement
        "logprobs": True,
        "top_logprobs": 5,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    t0 = time.monotonic()
    try:
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
        body = json.dumps(payload).encode()
        conn.request("POST", "/v1/chat/completions", body, {
            "Content-Type": "application/json",
        })
        resp = conn.getresponse()
        data = json.loads(resp.read())
        conn.close()
    except Exception as e:
        return {
            "confident": True,  # Default to confident (skip RAG) on probe failure
            "mean_entropy": 0.0,
            "hedge_detected": False,
            "probe_text": "",
            "probe_ms": (time.monotonic() - t0) * 1000,
            "recommendation": "skip_rag",
            "error": str(e),
        }

    probe_ms = (time.monotonic() - t0) * 1000

    # Extract response
    choice = data.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content", "")
    logprobs_data = choice.get("logprobs", {})
    token_logprobs = logprobs_data.get("content", []) if logprobs_data else []

    # Analyze entropy from logprobs
    entropies = []
    for tok_entry in token_logprobs:
        top_lps = tok_entry.get("top_logprobs", [])
        if not top_lps:
            continue
        probs = [math.exp(lp.get("logprob", -10)) for lp in top_lps]
        total = sum(probs)
        if total <= 0:
            continue
        probs = [p / total for p in probs]
        entropy = -sum(p * math.log(p + 1e-10) for p in probs if p > 0)
        max_entropy = math.log(len(probs) + 1e-10)
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        entropies.append(norm_entropy)

    mean_entropy = sum(entropies) / len(entropies) if entropies else 0.0

    # Check for hedge phrases
    hedge_detected = bool(_HEDGE_RE.search(content))

    # Decision logic — hedge detection is primary (logprobs may not be available)
    if hedge_detected:
        recommendation = "deep_rag"
        confident = False
    elif mean_entropy > 0.6:
        recommendation = "deep_rag"
        confident = False
    elif mean_entropy > 0.35:
        recommendation = "quick_rag"
        confident = False
    elif mean_entropy == 0.0 and len(content) > 20:
        # No logprobs available (chimere-server FFI) — fallback to text analysis
        # Check for short/evasive responses or knowledge cutoff mentions
        short_evasive = len(content.strip()) < 80
        cutoff_mention = bool(re.search(
            r"(connaissance|knowledge|cutoff|training|data).*(limit|arrêt|date|2024|2025)",
            content, re.IGNORECASE
        ))
        if short_evasive or cutoff_mention:
            recommendation = "quick_rag"
            confident = False
        else:
            recommendation = "skip_rag"
            confident = True
    else:
        recommendation = "skip_rag"
        confident = True

    return {
        "confident": confident,
        "mean_entropy": round(mean_entropy, 4),
        "hedge_detected": hedge_detected,
        "probe_text": content[:200],
        "probe_ms": round(probe_ms, 1),
        "recommendation": recommendation,
    }


def should_trigger_rag(
    user_text: str,
    route_id: str,
    pipeline: dict,
) -> tuple[bool, str]:
    """Decide if RAG/web search should be triggered for this request.

    Combines:
    1. Pipeline YAML config (explicit web: true/false)
    2. Confidence probe (entropy + hedge detection)
    3. Route-specific rules

    Returns: (should_search: bool, depth: "skip"|"quick"|"standard"|"deep")
    """
    enrich_cfg = pipeline.get("enrich", {})
    if not isinstance(enrich_cfg, dict):
        enrich_cfg = {}

    # If pipeline explicitly enables web → always search
    explicit_web = enrich_cfg.get("web")
    if explicit_web is True:
        return True, "standard" if route_id == "research" else "quick"

    # If pipeline explicitly disables web → check confidence
    if explicit_web is False:
        # Even with web:false, probe the model — if it's uncertain, search anyway
        if len(user_text) < 40:
            return False, "skip"

        probe = probe_confidence(user_text)

        if probe["recommendation"] == "deep_rag":
            return True, "standard"
        elif probe["recommendation"] == "quick_rag":
            return True, "quick"
        else:
            return False, "skip"

    # No explicit config → use probe
    probe = probe_confidence(user_text)
    return not probe["confident"], probe["recommendation"].replace("_rag", "").replace("skip", "skip")


# Self-test
if __name__ == "__main__":
    import sys
    queries = [
        "Quels sont les critères de retour au sport après LCA ?",  # kine, model should know
        "Quelles sont les dernières avancées en IA pour l'agronomie en 2026 ?",  # uncertain, needs web
        "Bonjour, ça va ?",  # trivial, no RAG needed
        "Compare les performances de Qwen3.5 vs Llama 4 sur MMLU-Pro",  # needs fresh data
    ]

    for q in queries:
        print(f"\nQ: {q[:60]}...")
        result = probe_confidence(q)
        print(f"  Confident: {result['confident']}")
        print(f"  Entropy: {result['mean_entropy']:.3f}")
        print(f"  Hedge: {result['hedge_detected']}")
        print(f"  Recommendation: {result['recommendation']}")
        print(f"  Probe: {result['probe_text'][:100]}...")
        print(f"  Time: {result['probe_ms']:.0f}ms")
