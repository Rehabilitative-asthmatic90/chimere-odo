#!/usr/bin/env python3
"""
Think Router — Entropy-based thinking/no-think proxy for Qwen3.5-35B-A3B.

Sits between Chimère gateway and llama-server. Does a 5-token "preflight" probe
with logprobs to measure token entropy, then decides whether to enable thinking mode.

Architecture:
  Gateway (25443) → think_router (8084) → llama-server (8081)

Probe strategy (probe-then-commit):
  1. Send max_tokens=5 probe with logprobs, no-think
  2. Measure mean entropy from top-5 logprobs
  3. High entropy → enable thinking (complex query)
  4. Low entropy → disable thinking (simple query, faster)
  5. llama-server --cache-reuse 256 reuses KV cache from probe

Install:
  systemctl --user enable --now think-router
  # Update chimere.json: qwen35.baseUrl → http://127.0.0.1:8084/v1
"""

import json
import http.client
import math
import os
import re
import sqlite3
import sys
import time
import threading
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
from urllib.parse import urlparse

# ── Config ──────────────────────────────────────────────────────────────────
LISTEN_PORT = 8084
LLAMA_BASE = "http://127.0.0.1:8081"
DB_PATH = Path.home() / ".chimere/logs/think_router.db"

PROBE_MAX_TOKENS = 5
PROBE_TOP_LOGPROBS = 5
PROBE_TIMEOUT = 15       # seconds — abort probe if llama-server is busy
FORWARD_TIMEOUT = 300    # seconds — generous timeout for actual generation
ENTROPY_THRESHOLD = float(os.environ.get("ENTROPY_THRESHOLD", "0.8"))
FORCE_THINK = os.environ.get("FORCE_THINK", "0") == "1"

# ABF (Adaptive Budget Forcing, ICLR 2026) — 2-signal Ct: Conf + (1-Ent), no Coherence
# Replaces naive char-count budget forcing with certainty-scored halting
ABF_ENABLED = os.environ.get("ABF_ENABLED", "1") == "1"
ABF_ALPHA = float(os.environ.get("ABF_ALPHA", "0.625"))   # Confidence weight (reweighted for 2-signal)
ABF_BETA = float(os.environ.get("ABF_BETA", "0.375"))     # (1-Entropy) weight
ABF_THRESHOLD = float(os.environ.get("ABF_THRESHOLD", "0.55"))  # Ct halting threshold
ABF_MIN_THINKING_CHARS = int(os.environ.get("ABF_MIN_THINKING_CHARS", "100"))  # min thinking before ABF can halt
ABF_MAX_RETRIES = int(os.environ.get("ABF_MAX_RETRIES", "3"))
ABF_FALLBACK_MIN_CHARS = int(os.environ.get("ABF_FALLBACK_MIN_CHARS", "500"))  # accept if long enough regardless of Ct
ABF_STREAM_WINDOW = int(os.environ.get("ABF_STREAM_WINDOW", "5"))  # sliding window for inline SSE ABF
LOG_VOCAB_SIZE = 11.93  # log(152064) for Qwen3.5 entropy normalization

# CGRS (Certainty-Guided Reasoning Suppression, AAAI 2026) — suppress reflection triggers
CGRS_ENABLED = os.environ.get("CGRS_ENABLED", "1") == "1"
CGRS_DELTA = float(os.environ.get("CGRS_DELTA", "0.9"))  # suppression threshold
# Qwen reflection trigger token IDs (Qwen2Tokenizer — verify for Qwen3.5)
CGRS_TRIGGER_IDS = {
    "14190": -100, "13824": -100,   # Wait, ␣Wait
    "11489": -100, "3783": -100,    # wait, ␣wait
    "3983": -100, "1988": -100,     # But, ␣But
    "8088": -100, "714": -100,      # but, ␣but
    "38478": -100,                   # Alternatively / ␣Alternatively
    "75763": -100, "41109": -100,   # Alternative, ␣Alternative
    "80022": -100, "88190": -100,   # Hmm, ␣Hmm
}

# Legacy compat (ABF replaces these but kept for env var override)
BUDGET_FORCING_ENABLED = ABF_ENABLED  # alias
BUDGET_FORCING_MIN_CHARS = ABF_FALLBACK_MIN_CHARS
BUDGET_FORCING_MAX_RETRIES = ABF_MAX_RETRIES

# Training pair logging for overnight LoRA pipeline
LOG_TRAINING_PAIRS = os.environ.get("LOG_TRAINING_PAIRS", "1") == "1"
TRAINING_PAIRS_PATH = Path.home() / ".chimere/logs/training_pairs.jsonl"

# Sampling profiles (Qwen3.5 official recommendations)
THINK_MIN_TOKENS = 4096  # minimum tokens when thinking active (guard against overflow)

THINK_PARAMS = {
    "temperature": 1.0, "top_p": 0.95, "top_k": 20,
    "presence_penalty": 0.0, "max_tokens": 16384,  # cap: évite réponse vide si think chain longue
    "chat_template_kwargs": {"enable_thinking": True},
}
NO_THINK_PARAMS = {
    "temperature": 0.7, "top_p": 0.8, "top_k": 20,
    "presence_penalty": 0.0, "max_tokens": 8192,
    "chat_template_kwargs": {"enable_thinking": False},
}
CODE_THINK_PARAMS = {
    "temperature": 0.6, "top_p": 0.95, "top_k": 20,
    "presence_penalty": 0.0, "max_tokens": 16384,  # cap: laisse de la place pour le code réel
    "chat_template_kwargs": {"enable_thinking": True},
}
CODE_NO_THINK_PARAMS = {
    "temperature": 0.6, "top_p": 0.9, "top_k": 20,
    "presence_penalty": 0.0, "max_tokens": 4096,
    "chat_template_kwargs": {"enable_thinking": False},
}

CODE_RE = re.compile(
    r'(code|script|program|fonction|function|class|debug|fix|refactor'
    r'|bug|error|compile|import\s|variable|python|bash|javascript|rust|sql'
    r'|api\b|endpoint|fastapi|django|flask|express|serveur|server|docker'
    r'|fichier\s+\w*\.py|cree\s+un\s+\w*script|html|css|react|vue)',
    re.I
)

# Complex queries that benefit from extended thinking (budget forcing trigger)
COMPLEXITY_RE = re.compile(
    r'((?:résou[dst]|solve|calculer?|compute|prouver?|prove|démontrer?)'
    r'|(?:équation|equation|intégral|integral|dérivé|derivative|matrice|matrix)'
    r'|(?:algorithm|complexité|complexity|optimi[sz]|NP-|O\(n)'
    r'|(?:analys(?:e[rz]?|is)|compar(?:e[rz]?|ison)|explain\s+(?:why|how|the\s+difference))'
    r'|(?:raisonn|logique|paradox|dilemm|stratég|archite?ctur)'
    r'|(?:debug.*complex|refactor.*entir|design.*system|implement.*from\s+scratch)'
    r'|(?:\d{2,}\s*[×x\*÷/]\s*\d{2,}))',
    re.I
)

# Simple greetings / acknowledgments → always no-think (S1 fast path)
GREETING_RE = re.compile(
    r'^\s*(?:bonjour|salut|hello|hi|hey|coucou|bonsoir|merci|thanks|ok|okay'
    r"|d'accord|ça va|comment (?:ça va|vas-tu|allez-vous)"
    r'|good (?:morning|evening|night|afternoon)|bonne (?:nuit|journée|soirée)'
    r'|au revoir|bye|à\s*\+|bisous?|ciao)\s*[!?.\s]*$',
    re.I
)

# ── Database ────────────────────────────────────────────────────────────────
def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS think_decisions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            ts            TEXT NOT NULL DEFAULT (datetime('now')),
            probe_entropy REAL,
            decision      TEXT,
            domain        TEXT,
            probe_ms      INTEGER,
            total_ms      INTEGER,
            prompt_len    INTEGER,
            sample_prompt TEXT,
            budget_retries INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    # Add budget_retries column if upgrading from old schema
    try:
        conn.execute("ALTER TABLE think_decisions ADD COLUMN budget_retries INTEGER DEFAULT 0")
        conn.commit()
    except Exception:
        pass  # Column already exists
    conn.close()


def log_decision(data: dict):
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute(
            "INSERT INTO think_decisions (ts, probe_entropy, decision, domain, probe_ms, total_ms, prompt_len, sample_prompt, budget_retries) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(), data.get("probe_entropy"), data.get("decision"),
             data.get("domain"), data.get("probe_ms"), data.get("total_ms"),
             data.get("prompt_len"), data.get("sample_prompt"), data.get("budget_retries", 0)),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _log_training_pair(user_text: str, reasoning: str, content: str, budget_retries: int):
    """Log prompt/response pair for overnight LoRA training pipeline."""
    if not LOG_TRAINING_PAIRS:
        return
    try:
        entry = {
            "ts": datetime.now().isoformat(),
            "prompt": user_text[:2000],
            "reasoning": reasoning[:8000],
            "response": content[:4000],
            "budget_retries": budget_retries,
        }
        TRAINING_PAIRS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TRAINING_PAIRS_PATH, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def compute_abf_certainty(logprobs_content: list, window: int = 32) -> float:
    """ABF composite certainty: Ct = alpha*Conf + beta*(1-Ent_normalized).

    Uses a sliding window over the last `window` logprob entries.
    Conf = mean of max softmax probability per token.
    Ent = mean Shannon entropy, normalized by log(vocab_size).
    Returns Ct in [0, 1].
    """
    if not logprobs_content:
        return 0.0

    tokens = logprobs_content[-window:]
    confs = []
    ents = []
    for tok in tokens:
        top = tok.get("top_logprobs", [])
        if not top:
            continue
        probs = [math.exp(lp["logprob"]) for lp in top if "logprob" in lp]
        if not probs:
            continue
        total = sum(probs)
        if total <= 0:
            continue
        probs = [p / total for p in probs]
        confs.append(max(probs))
        H = -sum(p * math.log(p + 1e-10) for p in probs if p > 0)
        ents.append(H / LOG_VOCAB_SIZE)

    if not confs:
        return 0.0
    mean_conf = sum(confs) / len(confs)
    mean_ent = sum(ents) / len(ents)
    return ABF_ALPHA * mean_conf + ABF_BETA * (1 - mean_ent)


# ── Entropy ─────────────────────────────────────────────────────────────────
def token_entropy(logprobs_content: list) -> float:
    """Compute mean Shannon entropy from top-k logprobs."""
    entropies = []
    for tok in logprobs_content:
        top = tok.get("top_logprobs", [])
        if not top:
            continue
        probs = [math.exp(lp["logprob"]) for lp in top if "logprob" in lp]
        total = sum(probs)
        if total <= 0:
            continue
        probs = [p / total for p in probs]
        H = -sum(p * math.log(p + 1e-10) for p in probs if p > 0)
        entropies.append(H)
    return sum(entropies) / len(entropies) if entropies else 999.0


VALID_ROLES = {"system", "user", "assistant", "tool"}


def sanitize_messages(messages: list) -> list:
    """Sanitize message roles for Qwen3.5 Jinja template.

    Qwen3.5 only accepts: system, user, assistant, tool.
    Remaps or drops messages with other roles (e.g. tool_result, function, ipython).
    Also consolidates system messages to the beginning (Qwen3.5 Jinja requires this).
    """
    sanitized = []
    for msg in messages:
        role = msg.get("role", "")
        if role in VALID_ROLES:
            sanitized.append(msg)
        elif role in ("tool_result", "tool_response", "function"):
            # Remap tool-like roles to "tool"
            sanitized.append({**msg, "role": "tool"})
        elif role == "ipython":
            # Remap ipython to user
            sanitized.append({**msg, "role": "user"})
        else:
            # Unknown role — log and remap to user as safe fallback
            print(f"[think-router] WARNING: unknown role '{role}', remapping to 'user'",
                  file=sys.stderr, flush=True)
            sanitized.append({**msg, "role": "user"})

    # Consolidate system messages to the beginning.
    # Qwen3.5 Jinja template raises if system message is not at position 0.
    # Aider architect mode and other tools may interleave system messages.
    system_msgs = [m for m in sanitized if m.get("role") == "system"]
    other_msgs = [m for m in sanitized if m.get("role") != "system"]
    if len(system_msgs) > 1:
        # Merge all system messages into one
        merged_content = "\n\n".join(
            m.get("content", "") for m in system_msgs if m.get("content")
        )
        sanitized = [{"role": "system", "content": merged_content}] + other_msgs
    elif system_msgs and sanitized[0].get("role") != "system":
        # Single system message but not at position 0 — move it
        sanitized = system_msgs + other_msgs

    return sanitized


def extract_user_text(payload: dict) -> str:
    """Extract last user message text from OpenAI-format payload."""
    for msg in reversed(payload.get("messages", [])):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                return " ".join(c.get("text", "") for c in content if c.get("type") == "text")
            return str(content)
    return ""


def has_image(payload: dict) -> bool:
    """Check if payload contains image_url content blocks."""
    for msg in payload.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, list):
            for c in content:
                if c.get("type") == "image_url":
                    return True
    return False


def is_code_request(text: str) -> bool:
    return bool(CODE_RE.search(text))


def is_complex_query(text: str) -> bool:
    """Detect queries that benefit from extended thinking / budget forcing."""
    return bool(COMPLEXITY_RE.search(text))


# ── HTTP helpers ────────────────────────────────────────────────────────────
def _send_to_llama(payload: dict, timeout: int = 120) -> dict:
    """Send a request to llama-server and return parsed JSON response."""
    body = json.dumps(payload).encode()
    parsed = urlparse(LLAMA_BASE)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    conn.request("POST", "/v1/chat/completions", body=body, headers={
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
    })
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    return data


def _forward_raw(path: str, body: bytes, headers: dict, timeout: int = 300):
    """Forward a raw request to llama-server. Returns (status, resp_headers, resp_body)."""
    parsed = urlparse(LLAMA_BASE)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    conn.request("POST", path, body=body, headers=headers)
    resp = conn.getresponse()
    return resp, conn


# ── Proxy Handler ───────────────────────────────────────────────────────────
class ThinkRouterHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass  # suppress access logs

    def do_GET(self):
        if self.path == "/health":
            self._json_response(200, {"status": "ok", "threshold": ENTROPY_THRESHOLD})
        elif self.path == "/stats":
            self._send_stats()
        elif self.path.startswith("/v1/models"):
            self._proxy_get()
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        if not self.path.startswith("/v1/chat/completions"):
            # Transparent proxy for non-chat endpoints
            self._proxy_post(body)
            return

        try:
            payload = json.loads(body)
        except Exception:
            self.send_error(400, "Bad JSON")
            return

        # Sanitize message roles for Qwen3.5 Jinja template
        if "messages" in payload:
            original_roles = [m.get("role") for m in payload["messages"]]
            payload["messages"] = sanitize_messages(payload["messages"])
            new_roles = [m.get("role") for m in payload["messages"]]
            if original_roles != new_roles:
                print(f"[think-router] roles sanitized: {original_roles} → {new_roles}",
                      file=sys.stderr, flush=True)

        t0 = time.time()
        user_text = extract_user_text(payload)

        # ── Fast paths (skip probe) ──

        # 1. Tool calls → no-think (thinking in stream breaks Chimère's tool call parser)
        if payload.get("tools") or payload.get("functions"):
            self._forward_with_params(payload, CODE_NO_THINK_PARAMS, "no-think", "tools", t0, user_text, probe_entropy=None)
            return

        # 2. Explicit thinking override from caller
        if "chat_template_kwargs" in payload:
            self._forward_payload(payload, t0)
            return

        # 3. Vision requests → always think (complex)
        if has_image(payload):
            params = CODE_THINK_PARAMS if is_code_request(user_text) else THINK_PARAMS
            self._forward_with_params(payload, params, "think", "vision", t0, user_text, probe_entropy=None)
            return

        # 4. Very short prompts → skip probe, use heuristic
        if len(user_text) < 20:
            self._forward_with_params(payload, NO_THINK_PARAMS, "no-think", "short", t0, user_text, probe_entropy=None)
            return

        # 4b. Simple greetings / acknowledgments → S1 no-think fast path
        if len(user_text) < 80 and GREETING_RE.match(user_text):
            self._forward_with_params(payload, NO_THINK_PARAMS, "no-think", "greeting", t0, user_text, probe_entropy=None)
            return

        # 5. Force think mode — skip probe entirely
        if FORCE_THINK:
            code = is_code_request(user_text)
            params = CODE_THINK_PARAMS if code else THINK_PARAMS
            domain = "code" if code else "general"
            self._forward_with_params(payload, params, "think", domain, t0, user_text, probe_entropy=None)
            return

        # ── Probe phase ──
        probe_payload = {
            **{k: v for k, v in payload.items() if k != "stream"},
            "max_tokens": PROBE_MAX_TOKENS,
            "stream": False,
            "logprobs": True,
            "top_logprobs": PROBE_TOP_LOGPROBS,
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0.7,
            "top_p": 0.8,
        }

        try:
            probe_t0 = time.time()
            probe_result = _send_to_llama(probe_payload, timeout=PROBE_TIMEOUT)
            probe_ms = int((time.time() - probe_t0) * 1000)
        except Exception as e:
            print(f"[think-router] Probe failed ({e}), defaulting to think", file=sys.stderr, flush=True)
            params = CODE_THINK_PARAMS if is_code_request(user_text) else THINK_PARAMS
            self._forward_with_params(payload, params, "think", "probe-fail", t0, user_text, probe_entropy=None)
            return

        # Extract entropy from probe
        choice = probe_result.get("choices", [{}])[0]
        logprobs_data = choice.get("logprobs", {})
        tok_list = logprobs_data.get("content", []) if logprobs_data else []
        entropy = token_entropy(tok_list) if tok_list else 999.0

        # ── Decision ──
        code = is_code_request(user_text)
        if entropy > ENTROPY_THRESHOLD:
            params = CODE_THINK_PARAMS if code else THINK_PARAMS
            decision = "think"
        else:
            params = CODE_NO_THINK_PARAMS if code else NO_THINK_PARAMS
            decision = "no-think"

        domain = "code" if code else "general"
        print(f"[think-router] entropy={entropy:.3f} → {decision} ({domain}) probe={probe_ms}ms",
              file=sys.stderr, flush=True)

        self._forward_with_params(payload, params, decision, domain, t0, user_text, probe_entropy=entropy, probe_ms=probe_ms)

    def _forward_with_params(self, payload, params, decision, domain, t0, user_text, probe_entropy=None, probe_ms=0):
        """Apply sampling params and forward the real request. Integrates budget forcing."""
        real_payload = dict(payload)
        is_thinking = params.get("chat_template_kwargs", {}).get("enable_thinking", False)
        is_streaming = payload.get("stream", False)

        # Apply params but preserve caller's stream setting
        for k, v in params.items():
            if k == "max_tokens":
                caller_max = payload.get("max_tokens")
                merged = min(caller_max, v) if caller_max else v
                if is_thinking:
                    merged = max(merged, THINK_MIN_TOKENS)
                real_payload[k] = merged
            else:
                real_payload[k] = v

        # ABF: complex thinking queries → inline stream monitoring with Ct score
        budget_retries = 0
        complex_q = is_complex_query(user_text)
        needs_abf = (ABF_ENABLED and is_thinking and complex_q and len(user_text) > 30)

        if needs_abf:
            result, budget_retries = self._abf_monitor(real_payload, user_text)
            resp_body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)
        else:
            # For streaming with thinking enabled + ABF: request logprobs from backend
            # and enable inline ABF monitoring in the SSE forwarder
            abf_stream = (ABF_ENABLED and is_thinking and is_streaming
                          and FORCE_THINK and len(user_text) > 20)
            if abf_stream:
                real_payload["logprobs"] = True
                real_payload["top_logprobs"] = PROBE_TOP_LOGPROBS

            body = json.dumps(real_payload).encode()
            try:
                resp, conn = _forward_raw(self.path, body, {
                    "Content-Type": "application/json",
                    "Content-Length": str(len(body)),
                })
            except Exception as e:
                self.send_error(502, str(e))
                return

            if is_streaming:
                self._stream_response(resp, conn, abf_monitor=abf_stream)
            else:
                self._buffer_response(resp, conn, user_text=user_text)

        total_ms = int((time.time() - t0) * 1000)

        # Log decision async
        threading.Thread(target=log_decision, daemon=True, args=({
            "probe_entropy": probe_entropy,
            "decision": decision,
            "domain": domain,
            "probe_ms": probe_ms,
            "total_ms": total_ms,
            "prompt_len": len(user_text),
            "sample_prompt": user_text[:200],
            "budget_retries": budget_retries,
        },)).start()

    def _abf_monitor(self, payload, user_text):
        """ABF inline streaming monitor: stream with logprobs, compute Ct, retry if needed.

        Streams the request internally (zero overhead — same generation, no separate probe).
        Computes composite certainty Ct from logprobs. If Ct < threshold and reasoning
        is too short, retries with "Wait" prefill + CGRS logit_bias.

        Returns (result_dict, retries_count).
        """
        work_payload = dict(payload)
        work_payload["stream"] = True
        work_payload["logprobs"] = True
        work_payload["top_logprobs"] = PROBE_TOP_LOGPROBS
        original_messages = list(payload["messages"])

        for attempt in range(ABF_MAX_RETRIES + 1):
            body = json.dumps(work_payload).encode()
            try:
                resp, conn = _forward_raw(self.path, body, {
                    "Content-Type": "application/json",
                    "Content-Length": str(len(body)),
                }, timeout=FORWARD_TIMEOUT)
            except Exception as e:
                print(f"[think-router] ABF: llama-server error ({e})",
                      file=sys.stderr, flush=True)
                return {"choices": [{"message": {"content": f"Error: {e}", "role": "assistant"}}]}, attempt

            reasoning_buf = []
            content_buf = []
            logprob_entries = []
            finish_reason = None

            try:
                for raw_line in resp:
                    line = raw_line.strip()
                    if not line:
                        continue
                    if line == b'data: [DONE]':
                        break
                    if not line.startswith(b'data: '):
                        continue
                    try:
                        chunk = json.loads(line[6:])
                    except Exception:
                        continue
                    choice = chunk.get("choices", [{}])[0]
                    delta = choice.get("delta", {})
                    # Collect logprobs inline (zero overhead — same generation)
                    lp = choice.get("logprobs", {})
                    if lp:
                        logprob_entries.extend(lp.get("content", []))
                    if "reasoning_content" in delta and delta["reasoning_content"]:
                        reasoning_buf.append(delta["reasoning_content"])
                    if "content" in delta and delta["content"]:
                        content_buf.append(delta["content"])
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
            except Exception:
                pass
            finally:
                conn.close()

            reasoning_text = "".join(reasoning_buf)
            content_text = "".join(content_buf)
            reasoning_len = len(reasoning_text)

            # Compute ABF certainty from inline logprobs
            ct = compute_abf_certainty(logprob_entries) if logprob_entries else 0.0

            # Halting decision
            accepted = False
            if ct >= ABF_THRESHOLD and reasoning_len >= ABF_MIN_THINKING_CHARS:
                accepted = True
            elif reasoning_len >= ABF_FALLBACK_MIN_CHARS:
                accepted = True  # long enough, accept regardless of Ct
            elif attempt >= ABF_MAX_RETRIES:
                accepted = True  # max retries

            if accepted:
                tag = "accepted" if attempt == 0 else f"accepted after {attempt} retries"
                print(f"[think-router] ABF: {tag} (Ct={ct:.3f}, {reasoning_len} chars)",
                      file=sys.stderr, flush=True)
                if reasoning_text:
                    _log_training_pair(user_text, reasoning_text, content_text, attempt)
                return {
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": content_text},
                        "finish_reason": finish_reason or "stop",
                    }]
                }, attempt

            # Reject: retry with "Wait" prefill
            print(f"[think-router] ABF: attempt {attempt+1}/{ABF_MAX_RETRIES}, "
                  f"Ct={ct:.3f} < {ABF_THRESHOLD}, {reasoning_len} chars, injecting Wait",
                  file=sys.stderr, flush=True)

            messages = list(original_messages)
            prefill = f"<think>\n{reasoning_text}\nWait, let me reconsider this step by step.\n"
            messages.append({"role": "assistant", "content": prefill})
            work_payload["messages"] = messages

            # CGRS: suppress reflection triggers on retry when partially confident
            if CGRS_ENABLED and ct > CGRS_DELTA:
                work_payload["logit_bias"] = CGRS_TRIGGER_IDS

        # Safety fallback
        return {
            "choices": [{"message": {"role": "assistant", "content": "".join(content_buf)},
                         "finish_reason": "stop"}]
        }, ABF_MAX_RETRIES

    def _forward_payload(self, payload, t0):
        """Forward payload as-is (explicit override from caller)."""
        body = json.dumps(payload).encode()
        is_streaming = payload.get("stream", False)
        try:
            resp, conn = _forward_raw(self.path, body, {
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            })
        except Exception as e:
            self.send_error(502, str(e))
            return
        if is_streaming:
            self._stream_response(resp, conn)
        else:
            self._buffer_response(resp, conn)

    def _stream_response(self, resp, conn, abf_monitor=False):
        """Stream SSE response from llama-server to client.

        When abf_monitor=True, parses logprobs from SSE events during the
        thinking phase and computes ABF certainty (Ct) over a sliding window.
        Logs when Ct exceeds ABF_THRESHOLD for ABF_STREAM_WINDOW consecutive
        tokens. This is informational — it does not halt the stream.
        """
        self.send_response(resp.status)
        self.send_header("Content-Type", resp.getheader("Content-Type", "text/event-stream"))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        # ABF inline state
        abf_window = []       # sliding window of per-token Ct values
        abf_halted = False    # True once we've logged the halt (log once)
        in_thinking = False   # tracks whether we're in <think> phase
        thinking_chars = 0    # chars of reasoning_content seen so far

        try:
            while True:
                line = resp.readline()
                if not line:
                    break
                # Always forward the line to the client
                self.wfile.write(line)
                self.wfile.flush()

                # ABF monitoring: parse SSE data lines during thinking phase
                if abf_monitor and not abf_halted:
                    stripped = line.strip()
                    if stripped == b'data: [DONE]':
                        continue
                    if not stripped.startswith(b'data: '):
                        continue
                    try:
                        chunk = json.loads(stripped[6:])
                    except Exception:
                        continue
                    choice = chunk.get("choices", [{}])[0]
                    delta = choice.get("delta", {})

                    # Track thinking phase via reasoning_content
                    rc = delta.get("reasoning_content")
                    if rc:
                        in_thinking = True
                        thinking_chars += len(rc)

                    # If content appears, thinking phase is over
                    if delta.get("content"):
                        in_thinking = False

                    # Only compute ABF during thinking phase with enough chars
                    if not in_thinking or thinking_chars < ABF_MIN_THINKING_CHARS:
                        continue

                    # Extract logprobs from this SSE event
                    lp_data = choice.get("logprobs")
                    if not lp_data:
                        continue
                    lp_content = lp_data.get("content", [])
                    if not lp_content:
                        continue

                    # Compute per-token Ct for each logprob entry in this event
                    for tok_entry in lp_content:
                        top_logprobs = tok_entry.get("top_logprobs", [])
                        if not top_logprobs:
                            continue
                        top1_logprob = tok_entry.get("logprob", None)
                        if top1_logprob is None:
                            # Fall back to first entry in top_logprobs
                            top1_logprob = top_logprobs[0].get("logprob", -10.0)

                        probs = [math.exp(lp["logprob"]) for lp in top_logprobs if "logprob" in lp]
                        if not probs:
                            continue
                        total = sum(probs)
                        if total <= 0:
                            continue
                        probs = [p / total for p in probs]

                        # Entropy (normalized)
                        entropy = -sum(p * math.log(p + 1e-10) for p in probs if p > 0)
                        max_entropy = math.log(len(probs) + 1e-10)
                        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

                        # Confidence (from top-1 logprob)
                        confidence = math.exp(top1_logprob)

                        # Composite certainty
                        ct = ABF_ALPHA * confidence + ABF_BETA * (1.0 - norm_entropy)
                        abf_window.append(ct)

                        # Keep only last ABF_STREAM_WINDOW entries
                        if len(abf_window) > ABF_STREAM_WINDOW:
                            abf_window = abf_window[-ABF_STREAM_WINDOW:]

                        # Check if all entries in a full window exceed threshold
                        if len(abf_window) == ABF_STREAM_WINDOW:
                            avg_ct = sum(abf_window) / ABF_STREAM_WINDOW
                            if avg_ct > ABF_THRESHOLD:
                                abf_halted = True
                                print(f"[ABF] Halting thinking: Ct={avg_ct:.3f} > {ABF_THRESHOLD} "
                                      f"(window={ABF_STREAM_WINDOW}, thinking_chars={thinking_chars})",
                                      file=sys.stderr, flush=True)
                                # Informational only — continue streaming

        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            conn.close()

    def _buffer_response(self, resp, conn, user_text=None):
        """Buffer full response, strip reasoning_content, optionally log training pair."""
        try:
            resp_body = resp.read()
            status = resp.status
        finally:
            conn.close()
        # Strip reasoning_content from response so Chimère doesn't store think chains in history
        try:
            data = json.loads(resp_body)
            for choice in data.get("choices", []):
                msg = choice.get("message", {})
                reasoning = msg.get("reasoning_content", "")
                content = msg.get("content", "")
                # Log training pair before stripping
                if reasoning and user_text:
                    _log_training_pair(user_text, reasoning, content, 0)
                if "reasoning_content" in msg:
                    del msg["reasoning_content"]
            resp_body = json.dumps(data).encode()
        except Exception:
            pass  # If parsing fails, forward as-is
        self.send_response(status)
        self.send_header("Content-Type", resp.getheader("Content-Type", "application/json"))
        self.send_header("Content-Length", str(len(resp_body)))
        self.end_headers()
        self.wfile.write(resp_body)

    def _proxy_post(self, body):
        """Transparent POST proxy for non-chat endpoints."""
        try:
            resp, conn = _forward_raw(self.path, body, {
                "Content-Type": self.headers.get("Content-Type", "application/json"),
                "Content-Length": str(len(body)),
            })
            resp_body = resp.read()
            conn.close()
        except Exception as e:
            self.send_error(502, str(e))
            return
        self.send_response(resp.status)
        self.send_header("Content-Type", resp.getheader("Content-Type", "application/json"))
        self.send_header("Content-Length", str(len(resp_body)))
        self.end_headers()
        self.wfile.write(resp_body)

    def _proxy_get(self):
        """Transparent GET proxy."""
        parsed = urlparse(LLAMA_BASE)
        try:
            conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=30)
            conn.request("GET", self.path)
            resp = conn.getresponse()
            resp_body = resp.read()
            conn.close()
        except Exception as e:
            self.send_error(502, str(e))
            return
        self.send_response(resp.status)
        self.send_header("Content-Type", resp.getheader("Content-Type", "application/json"))
        self.send_header("Content-Length", str(len(resp_body)))
        self.end_headers()
        self.wfile.write(resp_body)

    def _json_response(self, status, obj):
        body = json.dumps(obj, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_stats(self):
        try:
            conn = sqlite3.connect(str(DB_PATH))
            rows = conn.execute("""
                SELECT COUNT(*),
                       AVG(probe_entropy),
                       SUM(CASE WHEN decision='think' THEN 1 ELSE 0 END),
                       AVG(probe_ms),
                       AVG(total_ms),
                       SUM(CASE WHEN budget_retries > 0 THEN 1 ELSE 0 END),
                       AVG(CASE WHEN budget_retries > 0 THEN budget_retries END)
                FROM think_decisions WHERE ts > datetime('now', '-24 hours')
            """).fetchone()
            conn.close()
            total = rows[0] or 0
            think_count = rows[2] or 0
            budget_count = rows[5] or 0
            self._json_response(200, {
                "last_24h": {
                    "requests": total,
                    "avg_entropy": round(rows[1] or 0, 3),
                    "think_ratio": round(think_count / max(1, total), 3),
                    "avg_probe_ms": round(rows[3] or 0),
                    "avg_total_ms": round(rows[4] or 0),
                    "budget_forcing_count": budget_count,
                    "avg_budget_retries": round(rows[6] or 0, 1),
                },
                "config": {
                    "threshold": ENTROPY_THRESHOLD,
                    "force_think": FORCE_THINK,
                    "abf_enabled": ABF_ENABLED,
                    "abf_threshold": ABF_THRESHOLD,
                    "abf_alpha": ABF_ALPHA,
                    "abf_beta": ABF_BETA,
                    "abf_stream_window": ABF_STREAM_WINDOW,
                    "cgrs_enabled": CGRS_ENABLED,
                    "cgrs_delta": CGRS_DELTA,
                    "log_training_pairs": LOG_TRAINING_PAIRS,
                },
            })
        except Exception as e:
            self._json_response(500, {"error": str(e)})

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    init_db()
    server = ThreadedHTTPServer(("127.0.0.1", LISTEN_PORT), ThinkRouterHandler)
    print(f"[think-router] listening on 127.0.0.1:{LISTEN_PORT}", flush=True)
    print(f"[think-router] forwarding to {LLAMA_BASE}", flush=True)
    print(f"[think-router] entropy threshold: {ENTROPY_THRESHOLD}", flush=True)
    print(f"[think-router] force_think: {FORCE_THINK}", flush=True)
    print(f"[think-router] ABF: enabled={ABF_ENABLED} threshold={ABF_THRESHOLD} "
          f"alpha={ABF_ALPHA} beta={ABF_BETA} max_retries={ABF_MAX_RETRIES} "
          f"stream_window={ABF_STREAM_WINDOW}", flush=True)
    print(f"[think-router] CGRS: enabled={CGRS_ENABLED} delta={CGRS_DELTA} "
          f"triggers={len(CGRS_TRIGGER_IDS)} token IDs", flush=True)
    print(f"[think-router] training logging: {LOG_TRAINING_PAIRS} → {TRAINING_PAIRS_PATH}", flush=True)
    print(f"[think-router] stats: curl http://127.0.0.1:{LISTEN_PORT}/stats", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
