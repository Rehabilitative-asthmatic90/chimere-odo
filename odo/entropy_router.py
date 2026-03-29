#!/usr/bin/env python3
"""
Entropy Router — Pre-generation entropy classification for ODO.

Classifies requests by expected response entropy BEFORE generation.
Uses a lightweight heuristic approach (no ML model):

  1. Query complexity heuristics (length, technical keywords, ambiguity markers)
  2. Route confidence from classifier (low confidence = ambiguous = high entropy)
  3. Historical quality scores for the route (low avg score = hard domain = high entropy)

Returns an entropy_class and recommended generation parameters:
  - "low"    → no-think mode (factual, greetings, short)
  - "medium" → standard think (most requests)
  - "high"   → DVTS K=2 or extended ABF budget (complex, ambiguous, multi-step)

Integration: called from odo.py after classification, before _decide_thinking().
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────

QUALITY_SCORES_PATH = Path.home() / ".chimere/logs/quality_scores.jsonl"

# Lookback window for historical quality scores (max lines to scan from tail)
HISTORY_LOOKBACK = 200

# Weights for the composite entropy score (sum to 1.0)
W_COMPLEXITY = 0.45    # query complexity heuristics
W_CONFIDENCE = 0.30    # classifier confidence (inverted)
W_HISTORY = 0.25       # historical quality for route

# Thresholds for entropy classes
THRESHOLD_LOW = 0.28    # below this → low entropy
THRESHOLD_HIGH = 0.52   # above this → high entropy (was 0.65, never triggered)

# ── Complexity Heuristics ───────────────────────────────────────────────────

# Ambiguity markers: words/patterns that signal the query has no clear-cut answer
AMBIGUITY_RE = re.compile(
    r'(?i)\b('
    r'ou bien|ou alors|soit.*soit|alternative|entre .+ et .+'
    r'|est-ce que .+ ou .+'
    r'|what if|should i|would you|trade-?off|pros? (?:and|et) cons?'
    r'|avantages? (?:et|and) inconv[eé]nients?'
    r'|opinion|avis|recommand|conseil|suggest|choisir|d[eé]cid'
    r'|controversial|d[eé]bat|discuss|argument|nuanc'
    r'|it depends|[cç]a d[eé]pend'
    r')\b'
)

# Multi-step markers: queries that require structured multi-step reasoning
MULTISTEP_RE = re.compile(
    r'(?i)\b('
    r'[eé]tape|step|d\'abord.*ensuite|first.*then|plan|pipeline|workflow'
    r'|proc[eé]dure|processus|protocol|m[eé]thod'
    r'|compare[rz]?\s+.+\s+(?:et|and|vs|avec)\s+'
    r'|analyser?\b|d[eé]composer|d[eé]taill|exhausti'
    r'|1[\.\)]\s|a[\.\)]\s'
    r'|point par point|item by item|liste|list\b'
    r')\b'
)

# Technical depth markers: domain-specific jargon that implies depth
TECHNICAL_RE = re.compile(
    r'(?i)\b('
    r'algorithm|complexit[eé]|asymptoti|O\(|NP-|heuristique'
    r'|kernel|mutex|semaphore|syscall|interrupt|register'
    r'|backpropagation|gradient|loss\s+function|epoch|transformer'
    r'|r[eé]gression|bayesi[ea]n|markov|stochasti|eigenvalu'
    r'|protocole|RFC\s*\d|handshake|TCP|UDP|TLS|certifica'
    r'|CVE-\d|exploit|payload|shellcode|buffer\s+overflow'
    r'|quantification|GGUF|VRAM|tensor|CUDA|sm_\d'
    r'|biomec?anique|arthro|ligament|propri?oception|MTP|apon[eé]vrose'
    r')\b'
)

# Factual / simple patterns: queries with clear single-answer expectations
FACTUAL_RE = re.compile(
    r'(?i)('
    r'^(?:quel(?:le)?|what|who|when|where|combien|how (?:many|much|old|long|far))\s'
    r'|^(?:est-ce que|is|are|was|were|did|does|do)\s.{5,40}\??\s*$'
    r'|^(?:tradui[st]|translate|convertir?|convert)\s'
    r'|^(?:d[eé]fini[st]|define)\s'
    r'|^(?:quelle? (?:est|heure|date|temp[eé]rature))'
    r'|^(?:capital[e]? (?:de|of))\b'
    r')'
)

# Greeting / trivial (already in odo.py, replicated here for self-containment)
GREETING_RE = re.compile(
    r'^\s*(?:bonjour|salut|hello|hi|hey|coucou|bonsoir|merci|thanks|ok|okay'
    r"|d'accord|[cç]a va|comment (?:[cç]a va|vas-tu|allez-vous)"
    r'|good (?:morning|evening|night|afternoon)|bonne (?:nuit|journ[eé]e|soir[eé]e)'
    r'|au revoir|bye|[aà]\s*\+|bisous?|ciao)\s*[!?.\s]*$',
    re.I
)


def _query_complexity(text: str) -> float:
    """Score query complexity from 0.0 (trivial) to 1.0 (very complex).

    Combines length, pattern matching, and structural indicators.
    """
    score = 0.0

    # ── Length component (0-0.25) ──
    n = len(text)
    if n < 20:
        score += 0.0
    elif n < 80:
        score += 0.05
    elif n < 200:
        score += 0.10
    elif n < 500:
        score += 0.18
    else:
        score += 0.25

    # ── Greeting / trivial → floor immediately ──
    if GREETING_RE.match(text):
        return 0.0

    # ── Factual / single-answer → low ──
    if FACTUAL_RE.search(text):
        return min(score + 0.05, 0.20)

    # ── Question marks: multiple questions = higher entropy ──
    qmarks = text.count('?')
    if qmarks >= 3:
        score += 0.15
    elif qmarks >= 2:
        score += 0.08
    elif qmarks >= 1:
        score += 0.03

    # ── Ambiguity markers (0-0.25) ──
    amb_hits = len(AMBIGUITY_RE.findall(text))
    if amb_hits:
        score += min(0.25, 0.10 * amb_hits)

    # ── Multi-step markers (0-0.20) ──
    multi_hits = len(MULTISTEP_RE.findall(text))
    if multi_hits:
        score += min(0.20, 0.08 * multi_hits)

    # ── Technical depth (0-0.20) ──
    tech_hits = len(TECHNICAL_RE.findall(text))
    if tech_hits:
        score += min(0.20, 0.06 * tech_hits)

    # ── Code blocks present → likely implementation task ──
    if '```' in text or 'def ' in text or 'fn ' in text:
        score += 0.05

    # ── Sentence count: more sentences = more context = more entropy ──
    sentences = len(re.split(r'[.!?]+', text))
    if sentences >= 5:
        score += 0.10
    elif sentences >= 3:
        score += 0.05

    return min(1.0, score)


def _confidence_entropy(route_confidence: float) -> float:
    """Convert route classifier confidence to an entropy signal.

    Low confidence → high entropy (ambiguous intent).
    High confidence → low entropy (clear intent).
    Returns 0.0 (clear) to 1.0 (ambiguous).
    """
    # Invert and scale: conf 0.95 → 0.05, conf 0.3 → 0.70
    # Clamp to [0, 1]
    return max(0.0, min(1.0, 1.0 - route_confidence))


# ── Historical Quality Cache ────────────────────────────────────────────────

_quality_cache: dict[str, list[int]] = {}
_quality_cache_ts: float = 0.0
_CACHE_TTL = 300  # 5 minutes


def _load_quality_history() -> dict[str, list[int]]:
    """Load recent quality scores per route from quality_scores.jsonl.

    Returns: {"route": [score1, score2, ...]} for recent entries.
    Cached with 5-minute TTL to avoid repeated file reads.
    """
    global _quality_cache, _quality_cache_ts

    now = time.time()
    if _quality_cache and (now - _quality_cache_ts) < _CACHE_TTL:
        return _quality_cache

    scores_by_route: dict[str, list[int]] = defaultdict(list)

    if not QUALITY_SCORES_PATH.exists():
        _quality_cache = dict(scores_by_route)
        _quality_cache_ts = now
        return _quality_cache

    try:
        # Read last HISTORY_LOOKBACK lines (tail of file)
        with open(QUALITY_SCORES_PATH, 'r') as f:
            lines = f.readlines()
        recent = lines[-HISTORY_LOOKBACK:] if len(lines) > HISTORY_LOOKBACK else lines

        for line in recent:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                route = entry.get("route", "unknown")
                score = entry.get("score")
                if isinstance(score, (int, float)):
                    scores_by_route[route].append(int(score))
            except (json.JSONDecodeError, KeyError):
                continue
    except Exception as e:
        print(f"[entropy_router] warning: failed to read quality history: {e}",
              file=sys.stderr, flush=True)

    _quality_cache = dict(scores_by_route)
    _quality_cache_ts = now
    return _quality_cache


def _history_entropy(route_id: str) -> float:
    """Estimate entropy from historical quality scores for this route.

    Logic:
      - If no history → return neutral 0.5 (unknown)
      - Low avg score (< 3.0) → high entropy (this route is hard, model struggles)
      - High avg score (>= 4.0) → low entropy (model handles this well)
      - High variance → boost entropy (unpredictable quality)

    Returns: 0.0 (easy route) to 1.0 (hard/unreliable route).
    """
    history = _load_quality_history()
    scores = history.get(route_id)

    if not scores or len(scores) < 3:
        return 0.5  # not enough data, be neutral

    avg = sum(scores) / len(scores)
    # Variance component: high spread means unpredictable
    variance = sum((s - avg) ** 2 for s in scores) / len(scores)
    std = math.sqrt(variance)

    # Map avg score (1-5) to entropy:
    #   avg=5 → 0.0, avg=4 → 0.15, avg=3 → 0.50, avg=2 → 0.75, avg=1 → 1.0
    if avg >= 4.5:
        base = 0.0
    elif avg >= 4.0:
        base = 0.15
    elif avg >= 3.0:
        base = 0.35 + 0.15 * (3.0 - (avg - 3.0))  # 0.35 to 0.50
    elif avg >= 2.0:
        base = 0.50 + 0.25 * (1.0 - (avg - 2.0))  # 0.50 to 0.75
    else:
        base = 0.85

    # Add variance bonus (high std → higher entropy, capped at +0.15)
    var_bonus = min(0.15, std * 0.10)

    return min(1.0, base + var_bonus)


# ── Actions ─────────────────────────────────────────────────────────────────

# Actions recommended per entropy class
ACTIONS = {
    "low": {
        "thinking": False,
        "description": "no-think mode (factual, simple, greeting)",
        "dvts": False,
        "abf_threshold": None,  # skip ABF
    },
    "medium": {
        "thinking": True,
        "description": "standard think pass",
        "dvts": False,
        "abf_threshold": None,  # use pipeline default
    },
    "high": {
        "thinking": True,
        "description": "DVTS K=2 or extended ABF budget",
        "dvts": True,
        "dvts_k": 2,
        "abf_threshold": 0.65,  # stricter ABF for high-entropy
    },
}


# ── Public API ──────────────────────────────────────────────────────────────

def estimate_entropy(user_text: str, route_id: str,
                     route_confidence: float) -> dict:
    """Estimate pre-generation entropy for a request.

    Args:
        user_text: The user's message text.
        route_id: The classified route (e.g., "code", "kine", "general").
        route_confidence: Confidence from the classifier (0.0-1.0).

    Returns:
        {
            "entropy_class": "low" | "medium" | "high",
            "entropy_score": float (0.0-1.0),
            "components": {
                "complexity": float,
                "confidence_entropy": float,
                "history_entropy": float,
            },
            "action": { ... },   # recommended generation params
            "estimate_ms": int,  # time taken in ms
        }
    """
    t0 = time.time()

    # Compute the three components
    complexity = _query_complexity(user_text)
    conf_entropy = _confidence_entropy(route_confidence)
    hist_entropy = _history_entropy(route_id)

    # Weighted composite
    composite = (
        W_COMPLEXITY * complexity
        + W_CONFIDENCE * conf_entropy
        + W_HISTORY * hist_entropy
    )

    # Classify
    if composite <= THRESHOLD_LOW:
        entropy_class = "low"
    elif composite >= THRESHOLD_HIGH:
        entropy_class = "high"
    else:
        entropy_class = "medium"

    action = dict(ACTIONS[entropy_class])
    estimate_ms = int((time.time() - t0) * 1000)

    return {
        "entropy_class": entropy_class,
        "entropy_score": round(composite, 4),
        "components": {
            "complexity": round(complexity, 4),
            "confidence_entropy": round(conf_entropy, 4),
            "history_entropy": round(hist_entropy, 4),
        },
        "action": action,
        "estimate_ms": estimate_ms,
    }


# ── CLI test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys as _sys

    test_cases = [
        ("Bonjour !", "general", 0.9),
        ("Quelle est la capitale de la France ?", "general", 0.95),
        ("Explique-moi la backpropagation dans un transformer avec attention multi-tete, "
         "puis compare avec un RNN classique", "tutor", 0.7),
        ("Compare les avantages et inconvenients de Rust vs Go pour un serveur HTTP "
         "haute performance. Detaille etape par etape.", "code", 0.6),
        ("Ecris un Hello World en Python", "code", 0.95),
        ("Analyse la CVE-2024-1234 et propose un plan de remediation en 5 etapes "
         "avec les impacts sur l'infrastructure Kubernetes", "cyber", 0.45),
        ("Merci beaucoup !", "general", 0.9),
         "kine", 0.85),
    ]

    if len(_sys.argv) >= 2:
        # CLI mode: entropy_router.py "message" [route] [confidence]
        msg = _sys.argv[1]
        route = _sys.argv[2] if len(_sys.argv) > 2 else "general"
        conf = float(_sys.argv[3]) if len(_sys.argv) > 3 else 0.7
        result = estimate_entropy(msg, route, conf)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        # Run built-in test cases
        print("Entropy Router — Test Cases\n" + "=" * 60)
        for text, route, conf in test_cases:
            result = estimate_entropy(text, route, conf)
            cls = result["entropy_class"]
            score = result["entropy_score"]
            action_desc = result["action"]["description"]
            comps = result["components"]
            print(f"\n  [{cls.upper():6s}] score={score:.3f}  "
                  f"(cx={comps['complexity']:.2f} "
                  f"cf={comps['confidence_entropy']:.2f} "
                  f"hx={comps['history_entropy']:.2f})")
            print(f"  Action: {action_desc}")
            print(f"  Query:  {text[:80]}...")
        print()
