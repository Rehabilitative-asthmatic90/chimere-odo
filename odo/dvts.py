"""dvts.py — Diverse Verifier Tree Search

Generates K candidate responses, scores each with ThinkPRM step-level
verification, and returns the best one. For "hard" queries where
single-pass quality is insufficient.

Architecture:
  user query → generate K candidates (chimere-server)
             → score each (ThinkPRM step-level, CPU)
             → return best candidate

Integration: called from ODO for routes marked dvts: true in pipeline YAML.

References:
  - Diverse Verifier Tree Search (DVTS) — arXiv 2024
  - ThinkPRM: step-level PRM with CoT verification
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.parse import urlparse

log = logging.getLogger("dvts")

CHIMERE_URL = os.environ.get("CHIMERE_BACKEND", "http://127.0.0.1:8081")
THINKPRM_URL = os.environ.get("THINKPRM_BACKEND", "http://127.0.0.1:8085")

# Defaults
DEFAULT_K = 4          # Number of candidates
DEFAULT_TIMEOUT = 90   # Per-candidate generation timeout
SCORE_TIMEOUT = 60     # ThinkPRM scoring timeout


def _generate_candidate(
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    candidate_id: int,
) -> dict[str, Any]:
    """Generate a single candidate response from chimere-server."""
    parsed = urlparse(CHIMERE_URL)
    payload = {
        "model": "qwen3.5",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "stream": False,
    }

    try:
        conn = http.client.HTTPConnection(
            parsed.hostname, parsed.port, timeout=DEFAULT_TIMEOUT
        )
        body = json.dumps(payload).encode()
        conn.request("POST", "/v1/chat/completions", body=body, headers={
            "Content-Type": "application/json",
        })
        resp = conn.getresponse()
        data = json.loads(resp.read())
        conn.close()

        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        reasoning = choice.get("message", {}).get("reasoning_content", "")

        return {
            "id": candidate_id,
            "content": content,
            "reasoning": reasoning,
            "finish_reason": choice.get("finish_reason", ""),
            "tokens": data.get("usage", {}).get("completion_tokens", 0),
            "error": None,
        }
    except Exception as e:
        return {
            "id": candidate_id,
            "content": "",
            "reasoning": "",
            "finish_reason": "error",
            "tokens": 0,
            "error": str(e),
        }


def _score_candidate_heuristic(
    user_text: str,
    response_text: str,
) -> dict[str, Any]:
    """Heuristic scoring when ThinkPRM is unavailable.

    Scores based on response length, structure (lists, headers), and
    relevance (keyword overlap with query). Not as good as a real PRM
    but differentiates between empty/short and substantive responses.
    """
    if not response_text:
        return {"score": 0.0, "step_labels": [], "n_steps": 0,
                "correct_steps": 0, "cot_preview": "", "error": "empty"}

    score = 0.0

    # Length score (0-0.3): prefer substantive responses, diminishing returns
    length = len(response_text)
    if length > 100:
        score += min(0.3, 0.1 + 0.2 * min(length / 2000, 1.0))

    # Structure score (0-0.3): lists, headers, citations indicate quality
    import re
    structure_signals = (
        len(re.findall(r'^\s*[-*\d]+[.)]\s', response_text, re.MULTILINE)),  # list items
        len(re.findall(r'^#+\s', response_text, re.MULTILINE)),  # markdown headers
        len(re.findall(r'\[.*?\]', response_text)),  # citations/references
    )
    struct_count = sum(min(s, 5) for s in structure_signals)
    score += min(0.3, struct_count * 0.03)

    # Relevance score (0-0.4): keyword overlap with query
    query_tokens = set(user_text.lower().split())
    resp_tokens = set(response_text.lower().split())
    if query_tokens:
        overlap = len(query_tokens & resp_tokens) / len(query_tokens)
        score += 0.4 * min(overlap, 1.0)

    return {
        "score": round(score, 3),
        "step_labels": [],
        "n_steps": 0,
        "correct_steps": 0,
        "cot_preview": "(heuristic scoring)",
        "error": None,
    }


def _score_candidate(
    user_text: str,
    response_text: str,
    route_id: str = "",
) -> dict[str, Any]:
    """Score a candidate with ThinkPRM step-level verification.

    Falls back to heuristic scoring if ThinkPRM is not enabled or unavailable.
    """
    # Check if ThinkPRM is enabled before attempting to call it
    try:
        from quality_gate import THINKPRM_ENABLED
        if not THINKPRM_ENABLED:
            return _score_candidate_heuristic(user_text, response_text)
    except ImportError:
        return _score_candidate_heuristic(user_text, response_text)

    from quality_gate import _call_thinkprm

    try:
        prefix_score, step_labels, cot = _call_thinkprm(
            user_text, response_text, route_id
        )
        # Guard against degenerate scores (all steps incorrect = 0.0)
        # which provide no differentiation between candidates
        if prefix_score <= 0.0 and not step_labels:
            log.warning("[DVTS] ThinkPRM returned score=%.3f with no step labels, "
                        "falling back to heuristic", prefix_score)
            return _score_candidate_heuristic(user_text, response_text)
        return {
            "score": prefix_score,
            "step_labels": step_labels,
            "n_steps": len(step_labels),
            "correct_steps": sum(step_labels) if step_labels else 0,
            "cot_preview": cot[:200] if cot else "",
            "error": None,
        }
    except Exception as e:
        log.warning("[DVTS] ThinkPRM call failed (%s), using heuristic", e)
        return _score_candidate_heuristic(user_text, response_text)


def dvts_generate(
    messages: list[dict],
    user_text: str,
    route_id: str = "",
    k: int = DEFAULT_K,
    max_tokens: int = 4096,
    base_temperature: float = 0.7,
) -> dict[str, Any]:
    """Generate K candidates, score with ThinkPRM, return the best.

    Candidates are generated SEQUENTIALLY (np=1 constraint) but scored
    in parallel on CPU (ThinkPRM is CPU-only).

    Returns dict with:
      - content: best response text
      - reasoning: best response reasoning
      - score: ThinkPRM score of best candidate
      - candidates: list of all candidates with scores
      - dvts_ms: total time in milliseconds
    """
    t0 = time.monotonic()

    # Phase 1: Generate K candidates sequentially (GPU single-slot)
    log.info("[DVTS] Generating %d candidates for: %s", k, user_text[:60])
    candidates = []
    for i in range(k):
        # Slightly vary temperature for diversity
        temp = base_temperature + (i * 0.05)
        c = _generate_candidate(messages, temp, max_tokens, i)
        if c["error"]:
            log.warning("[DVTS] Candidate %d failed: %s", i, c["error"])
        else:
            log.info("[DVTS] Candidate %d: %d tokens", i, c["tokens"])
        candidates.append(c)

    # Filter out failed candidates
    valid = [c for c in candidates if c["content"] and not c["error"]]
    if not valid:
        log.error("[DVTS] All %d candidates failed", k)
        return {
            "content": "",
            "reasoning": "",
            "score": 0.0,
            "candidates": candidates,
            "dvts_ms": (time.monotonic() - t0) * 1000,
            "error": "All candidates failed",
        }

    if len(valid) == 1:
        # Only one valid candidate — skip scoring
        best = valid[0]
        return {
            "content": best["content"],
            "reasoning": best["reasoning"],
            "score": 0.5,
            "candidates": candidates,
            "dvts_ms": (time.monotonic() - t0) * 1000,
        }

    # Phase 2: Score candidates in parallel (ThinkPRM CPU, no GPU needed)
    log.info("[DVTS] Scoring %d valid candidates with ThinkPRM...", len(valid))
    scores = {}
    with ThreadPoolExecutor(max_workers=min(len(valid), 4)) as pool:
        futures = {}
        for c in valid:
            f = pool.submit(_score_candidate, user_text, c["content"], route_id)
            futures[f] = c["id"]

        for f in as_completed(futures):
            cid = futures[f]
            result = f.result()
            scores[cid] = result
            log.info(
                "[DVTS] Candidate %d: score=%.3f (%d/%d steps correct)",
                cid, result["score"], result["correct_steps"], result["n_steps"],
            )

    # Phase 3: Select best candidate
    best_id = max(scores, key=lambda cid: scores[cid]["score"])
    best_candidate = next(c for c in valid if c["id"] == best_id)
    best_score = scores[best_id]

    # Annotate all candidates with scores
    for c in candidates:
        if c["id"] in scores:
            c["thinkprm_score"] = scores[c["id"]]["score"]
            c["step_labels"] = scores[c["id"]]["step_labels"]
        else:
            c["thinkprm_score"] = None

    dvts_ms = (time.monotonic() - t0) * 1000
    log.info(
        "[DVTS] Best: candidate %d (score=%.3f) in %.0fms total",
        best_id, best_score["score"], dvts_ms,
    )

    return {
        "content": best_candidate["content"],
        "reasoning": best_candidate["reasoning"],
        "score": best_score["score"],
        "candidates": candidates,
        "dvts_ms": dvts_ms,
    }


# Self-test (requires chimere-server + ThinkPRM running)
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    query = sys.argv[1] if len(sys.argv) > 1 else "Quels sont les critères de retour au sport après rupture du LCA ?"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 2  # Default K=2 for quick test

    print(f"=== DVTS Test (K={k}) ===")
    print(f"Query: {query}")
    print()

    messages = [{"role": "user", "content": query}]
    result = dvts_generate(messages, query, route_id="kine", k=k)

    if result.get("error"):
        print(f"ERROR: {result['error']}")
    else:
        print(f"Best score: {result['score']:.3f}")
        print(f"Total time: {result['dvts_ms']:.0f}ms")
        print(f"\n--- Best response (first 500 chars) ---")
        print(result["content"][:500])
        print(f"\n--- All candidates ---")
        for c in result["candidates"]:
            score = c.get("thinkprm_score", "N/A")
            tokens = c.get("tokens", 0)
            err = c.get("error", "")
            print(f"  #{c['id']}: score={score}, tokens={tokens}{f', error={err}' if err else ''}")
