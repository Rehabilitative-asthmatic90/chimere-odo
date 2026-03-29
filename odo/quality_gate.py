#!/usr/bin/env python3
"""
ODO Quality Gate — Post-response micro-evaluation with dual scorer.

Two scoring backends:
  1. Qwen3.5 (legacy): fast heuristic "rate 1-5" via no-think on port 8081
  2. ThinkPRM-1.5B: generative step-level verification via CPU on port 8085

Modes (controlled by env vars):
  - THINKPRM_ENABLED=0: Qwen3.5 only (default)
  - THINKPRM_ENABLED=1 + THINKPRM_SHADOW=1: both run, Qwen3.5 decides
  - THINKPRM_ENABLED=1 + THINKPRM_SHADOW=0: ThinkPRM decides, Qwen3.5 backup

Based on the score:
  - score >= 4: Engram WRITE candidate (auto-ingest good patterns)
  - score <= 2: flag for reflection (caller can retry with self-critique)
  - score 3: neutral, pass through
"""

import json
import http.client
import math
import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

LLAMA_BASE = os.environ.get("ODO_BACKEND", "http://127.0.0.1:8081")
QUALITY_LOG = Path.home() / ".chimere/logs/quality_scores.jsonl"

# ThinkPRM configuration
THINKPRM_BASE = os.environ.get("THINKPRM_BACKEND", "http://127.0.0.1:8085")
THINKPRM_ENABLED = os.environ.get("THINKPRM_ENABLED", "0") == "1"
THINKPRM_SHADOW = os.environ.get("THINKPRM_SHADOW", "1") == "1"

# Minimum response length to bother scoring
MIN_RESPONSE_LEN = 100

# Routes that benefit from quality scoring
SCORED_ROUTES = {"kine", "research", "cyber", "code"}

SCORE_PROMPT = """Rate this AI response on a scale of 1-5. Reply with ONLY a JSON object.

Criteria:
- Accuracy: Are facts correct and well-sourced?
- Completeness: Does it answer the full question?
- Structure: Is it well-organized and clear?
- Safety: No harmful or misleading content?

User question: {question}

AI response (first 1000 chars): {response}

Reply format: {{"score": N, "reason": "one sentence"}}"""

THINKPRM_PROMPT = """You are given a question and a proposed step-by-step answer:

[Question]

{question}

[Answer]

{steps}

Review and critique each step in the proposed answer to determine whether each step is correct. If the answer is incomplete, only verify the provided steps."""


def should_score(route_id: str, response_text: str, is_streaming: bool) -> bool:
    """Decide if this response should be quality-scored."""
    if route_id not in SCORED_ROUTES:
        return False
    if len(response_text) < MIN_RESPONSE_LEN:
        return False
    return True


def score_response_async(user_text: str, response_text: str, route_id: str,
                         callback=None):
    """Score response quality in a background thread. Non-blocking."""
    thread = threading.Thread(
        target=_score_and_log,
        args=(user_text, response_text, route_id, callback),
        daemon=True
    )
    thread.start()


def _score_and_log(user_text: str, response_text: str, route_id: str,
                   callback=None):
    """Internal: call LLM for micro-eval, log result. Dual scorer if ThinkPRM enabled."""
    try:
        import hashlib
        prompt_hash = hashlib.sha256(user_text.encode()).hexdigest()[:16]

        # Legacy Qwen3.5 scorer — skip if ThinkPRM is primary (avoids
        # blocking the single-slot chimere-server with a scorer request)
        score, reason = 3, "scorer_skipped"
        if not (THINKPRM_ENABLED and not THINKPRM_SHADOW):
            try:
                score, reason = _call_scorer(user_text, response_text)
            except Exception as e:
                print(f"[quality] qwen35 scorer failed: {e}",
                      file=sys.stderr, flush=True)

        entry = {
            "ts": datetime.now().isoformat(),
            "route": route_id,
            "score": score,
            "reason": reason,
            "scorer": "qwen35",
            "prompt_len": len(user_text),
            "response_len": len(response_text),
            "prompt_hash": prompt_hash,
        }

        # ThinkPRM shadow/production scoring
        if THINKPRM_ENABLED:
            try:
                v2_score, step_labels, cot = _call_thinkprm(
                    user_text, response_text, route_id)
                v1_mapped = _v2_to_v1(v2_score)
                entry["score_v2"] = round(v2_score, 4)
                entry["score_thinkprm"] = v1_mapped
                entry["step_labels"] = step_labels
                entry["verification_cot"] = cot[:2000]
                entry["scorer_v2"] = "thinkprm"

                if not THINKPRM_SHADOW:
                    # Production: ThinkPRM replaces legacy score
                    entry["score_legacy"] = score
                    entry["score"] = v1_mapped
                    entry["scorer"] = "thinkprm"
                    score = v1_mapped
                    reason = (f"ThinkPRM: {sum(step_labels)}/{len(step_labels)} steps correct"
                              if step_labels else reason)
                    entry["reason"] = reason

                prm_tag = "shadow" if THINKPRM_SHADOW else "primary"
                print(f"[quality] thinkprm({prm_tag}): P={v2_score:.3f} "
                      f"steps={step_labels} mapped={v1_mapped}",
                      file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[quality] thinkprm error: {e}", file=sys.stderr, flush=True)

        QUALITY_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(QUALITY_LOG, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        tag = "GOOD" if score >= 4 else "BAD" if score <= 2 else "OK"
        print(f"[quality] {tag} score={score} route={route_id} reason={reason[:80]}",
              file=sys.stderr, flush=True)

        if callback:
            callback(score, reason, route_id, user_text, response_text)

    except Exception as e:
        print(f"[quality] error: {e}", file=sys.stderr, flush=True)


# ── Qwen3.5 scorer (legacy) ─────────────────────────────────────────────────

def _call_scorer(user_text: str, response_text: str) -> tuple[int, str]:
    """Call Qwen3.5 with no-think for a quick quality score."""
    prompt = SCORE_PROMPT.format(
        question=user_text[:500],
        response=response_text[:1000]
    )

    payload = {
        "model": "q",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.1,
        "top_p": 0.5,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_object",
            "schema": {
                "type": "object",
                "properties": {
                    "score": {"type": "integer", "minimum": 1, "maximum": 5},
                    "reason": {"type": "string"}
                },
                "required": ["score", "reason"],
                "additionalProperties": False
            }
        },
    }

    body = json.dumps(payload).encode()
    parsed = urlparse(LLAMA_BASE)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=30)
    conn.request("POST", "/v1/chat/completions", body=body, headers={
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
    })
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()

    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(content[start:end])
            score = int(result.get("score", 3))
            reason = result.get("reason", "")
            return max(1, min(5, score)), reason
    except (json.JSONDecodeError, ValueError):
        pass

    m = re.search(r'\b([1-5])\b', content)
    if m:
        return int(m.group(1)), content[:100]

    return 3, "parse_failed"


# ── ThinkPRM scorer ─────────────────────────────────────────────────────────

def _extract_steps(response_text: str, route_id: str) -> list[str]:
    """Extract logical steps from a response. 5 strategies by content type."""
    # Strategy 1: numbered lists (math, structured)
    numbered = re.findall(r'(?:^|\n)\s*(\d+[.)]\s*.+)', response_text)
    if len(numbered) >= 2:
        return [s.strip() for s in numbered[:10]]

    # Strategy 2: markdown headers (kine, research — structured protocols)
    headers = re.findall(r'(?:^|\n)(#{1,4}\s+.+)', response_text)
    if len(headers) >= 2:
        # Extract header + first paragraph after it
        sections = re.split(r'\n#{1,4}\s+', response_text)
        steps = []
        for i, section in enumerate(sections[1:], 1):  # skip preamble
            header = headers[min(i-1, len(headers)-1)].strip()
            body = section.strip()[:300]
            steps.append(f"{header}: {body}")
        if len(steps) >= 2:
            return steps[:10]

    # Strategy 3: bullet points (- or * lists, common in kine/medical)
    bullets = re.findall(r'(?:^|\n)\s*[-*]\s+(.{20,})', response_text)
    if len(bullets) >= 3:
        return [b.strip()[:300] for b in bullets[:10]]

    # Strategy 4: code blocks (code route)
    if route_id == "code":
        blocks = re.split(r'(?:```[\w]*\n|```\n?)', response_text)
        steps = []
        for i, block in enumerate(blocks):
            block = block.strip()
            if len(block) > 30:
                label = "Code" if i % 2 == 1 else "Explanation"
                steps.append(f"{label}: {block[:500]}")
        if len(steps) >= 2:
            return steps[:10]

    # Strategy 5: paragraphs (general fallback)
    paragraphs = [p.strip() for p in response_text.split('\n\n')
                  if p.strip() and len(p.strip()) > 30]
    if len(paragraphs) >= 2:
        return paragraphs[:10]

    # Fallback: sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', response_text)
    return [s for s in sentences if len(s) > 20][:10]


def _format_for_thinkprm(user_text: str, response_text: str,
                          route_id: str) -> str:
    """Format response as step-by-step for ThinkPRM verification."""
    steps = _extract_steps(response_text, route_id)
    steps_text = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(steps))
    return THINKPRM_PROMPT.format(
        question=user_text[:1000],
        steps=steps_text
    )


def _call_thinkprm(user_text: str, response_text: str,
                    route_id: str) -> tuple[float, list[int], str]:
    """Call ThinkPRM-1.5B for step-verification scoring.

    Returns: (prefix_score 0.0-1.0, step_labels [0/1,...], verification_cot)
    """
    prompt_text = _format_for_thinkprm(user_text, response_text, route_id)

    payload = {
        "model": "thinkprm",
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": 2048,
        "temperature": 0.0,
        "stream": False,
        "logprobs": True,
        "top_logprobs": 10,
    }

    body = json.dumps(payload).encode()
    parsed = urlparse(THINKPRM_BASE)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=60)
    conn.request("POST", "/v1/chat/completions", body=body, headers={
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
    })
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()

    msg = data.get("choices", [{}])[0].get("message", {})
    reasoning = msg.get("reasoning_content", "")
    content = msg.get("content", "")
    full_cot = reasoning + "\n" + content if reasoning else content

    logprobs_data = data.get("choices", [{}])[0].get("logprobs", {})
    token_logprobs = logprobs_data.get("content", []) if logprobs_data else []

    step_labels = _extract_step_labels(full_cot)
    prefix_score = _extract_prefix_score(token_logprobs)

    # Fallback: if logprob extraction failed, derive from step labels
    if prefix_score < 0 and step_labels:
        prefix_score = sum(step_labels) / len(step_labels)
    elif prefix_score < 0:
        # Last resort: check if content says Yes or No
        if re.search(r'\byes\b', content, re.IGNORECASE):
            prefix_score = 0.9
        elif re.search(r'\bno\b', content, re.IGNORECASE):
            prefix_score = 0.1
        else:
            prefix_score = 0.5

    return prefix_score, step_labels, full_cot


def _extract_step_labels(text: str) -> list[int]:
    r"""Extract step correctness from \boxed{correct}/\boxed{incorrect} or variants."""
    labels = []
    # Pattern 1: \boxed{correct} / \boxed{incorrect}
    for m in re.finditer(r'\\boxed\{(correct|incorrect)\}', text, re.IGNORECASE):
        labels.append(1 if m.group(1).lower() == "correct" else 0)
    if labels:
        return labels

    # Pattern 2: "Step N is correct/incorrect" or "Step N: correct/incorrect"
    for m in re.finditer(
            r'[Ss]tep\s*\d+\s*(?:is\s+|:\s*)(correct|incorrect|right|wrong)',
            text, re.IGNORECASE):
        val = m.group(1).lower()
        labels.append(1 if val in ("correct", "right") else 0)
    if labels:
        return labels

    # Pattern 3: checkmarks / crosses (common in verification CoT)
    for m in re.finditer(r'(✓|✗|✔|✘|☑|☒)', text):
        labels.append(1 if m.group(1) in ('✓', '✔', '☑') else 0)

    return labels


def _extract_prefix_score(token_logprobs: list) -> float:
    """Extract P(Yes) from logprobs at the Yes/No decision point.

    Scans backwards through tokens to find where model decides Yes/No.
    Returns P(Yes) / (P(Yes) + P(No)), or -1.0 if extraction fails.
    """
    YES_TOKENS = {" Yes", "Yes", " yes", "yes", " YES"}
    NO_TOKENS = {" No", "No", " no", "no", " NO"}

    for tok_entry in reversed(token_logprobs):
        top = tok_entry.get("top_logprobs", [])
        yes_lp = None
        no_lp = None
        for lp in top:
            token_text = lp.get("token", "")
            if token_text in YES_TOKENS and yes_lp is None:
                yes_lp = lp.get("logprob", -100)
            elif token_text in NO_TOKENS and no_lp is None:
                no_lp = lp.get("logprob", -100)

        if yes_lp is not None and no_lp is not None:
            p_yes = math.exp(yes_lp)
            p_no = math.exp(no_lp)
            total = p_yes + p_no
            return p_yes / total if total > 0 else 0.5

    return -1.0


def _v2_to_v1(score_v2: float) -> int:
    """Map ThinkPRM float [0,1] to legacy int [1,5]."""
    if score_v2 < 0:
        return 3  # fallback for extraction failure
    if score_v2 >= 0.9:
        return 5
    if score_v2 >= 0.7:
        return 4
    if score_v2 >= 0.5:
        return 3
    if score_v2 >= 0.3:
        return 2
    return 1


# ── Public API (unchanged interface) ─────────────────────────────────────────

def score_response_sync(user_text: str, response_text: str) -> tuple[int, str]:
    """Score response synchronously. Returns (score, reason). ~0.5-2s."""
    try:
        return _call_scorer(user_text, response_text)
    except Exception as e:
        print(f"[quality] sync error: {e}", file=sys.stderr, flush=True)
        return 3, "error"


def reflect_and_retry(user_text: str, bad_response: str, reason: str) -> str | None:
    """Ask the LLM to self-critique and produce a better response.

    Returns improved response text, or None if reflection fails.
    """
    reflect_prompt = (
        f"Your previous response was rated poorly. Reason: {reason}\n\n"
        f"Original question: {user_text[:500]}\n\n"
        f"Your previous answer (first 500 chars): {bad_response[:500]}\n\n"
        f"Please provide a corrected, improved response. "
        f"Address the specific criticism. Be accurate and complete."
    )

    payload = {
        "model": "q",
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": bad_response[:1000]},
            {"role": "user", "content": reflect_prompt},
        ],
        "max_tokens": 8192,
        "temperature": 0.6,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": True},
    }

    try:
        body = json.dumps(payload).encode()
        parsed = urlparse(LLAMA_BASE)
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=120)
        conn.request("POST", "/v1/chat/completions", body=body, headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        })
        resp = conn.getresponse()
        data = json.loads(resp.read())
        conn.close()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if content and len(content) > 50:
            print(f"[quality] reflection produced {len(content)} chars",
                  file=sys.stderr, flush=True)
            return content
    except Exception as e:
        print(f"[quality] reflection error: {e}", file=sys.stderr, flush=True)

    return None


FEW_SHOT_DIR = Path(__file__).parent / "few_shot"
FEW_SHOT_MAX = 10  # max examples per route in few-shot store


def on_quality_score(score: int, reason: str, route_id: str,
                     user_text: str = "", response_text: str = ""):
    """Callback for quality scores. Auto-feeds few-shot store for good responses."""
    if score >= 4 and user_text and response_text:
        _auto_feed_few_shot(route_id, user_text, response_text, score)
    elif score <= 2:
        print(f"[quality] low score={score} route={route_id} — "
              f"candidate for nightly negative example", file=sys.stderr, flush=True)


def _auto_feed_few_shot(route_id: str, user_text: str, response_text: str,
                         score: int):
    """Add high-quality response to few-shot store for the route."""
    try:
        FEW_SHOT_DIR.mkdir(parents=True, exist_ok=True)
        json_path = FEW_SHOT_DIR / f"{route_id}.json"

        # Load existing
        examples = []
        if json_path.exists():
            with open(json_path) as f:
                examples = json.load(f)

        # Check duplicate (by input prefix)
        input_prefix = user_text[:100].lower()
        for ex in examples:
            if ex.get("input", "")[:100].lower() == input_prefix:
                return  # already exists

        # Extract keywords as tags
        words = set(user_text.lower().split())
        tags = [w for w in words if len(w) > 4][:8]

        new_example = {
            "input": user_text[:500],
            "output": response_text[:2000],
            "score": score,
            "tags": tags,
            "ts": datetime.now().isoformat(),
        }
        examples.append(new_example)

        # Keep only top FEW_SHOT_MAX by score (LRU on equal scores)
        if len(examples) > FEW_SHOT_MAX:
            examples.sort(key=lambda x: (x.get("score", 0), x.get("ts", "")))
            examples = examples[-FEW_SHOT_MAX:]

        with open(json_path, "w") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)

        print(f"[quality] few-shot: added to {route_id} "
              f"({len(examples)} examples total)",
              file=sys.stderr, flush=True)

    except Exception as e:
        print(f"[quality] few-shot error: {e}", file=sys.stderr, flush=True)
