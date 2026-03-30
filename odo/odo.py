#!/usr/bin/env python3
"""
ODO — One Door Orchestrator (unified).

Single proxy that replaces BOTH the old ODO (8085) and think_router (8084).
No more duplicate forwarding hops.

Architecture:
  Client → ODO (8084) → llama-server (8081)

Features absorbed from think_router:
  - ABF (Adaptive Budget Forcing) with inline SSE logprobs
  - CGRS (Certainty-Guided Reasoning Suppression)
  - FORCE_THINK (always enable thinking, env override)
  - Sampling profiles (think/no-think/code-think/code-no-think)
  - Qwen3.5 system message consolidation & role sanitization
  - Entropy probing (optional, disabled when FORCE_THINK=1)
  - Training pair logging for overnight LoRA

Features absorbed from ODO:
  - Intent classification (keyword → filetype → LLM GBNF cascade)
  - Pipeline YAML loading with hot-reload (mtime cache)
  - Per-route params: temperature, max_tokens, engram, lora, gbnf, system_prefix
  - Per-route thinking override (pipeline can force think/no-think)

Install:
  systemctl --user enable --now odo
  # Port 8084 replaces think-router. Disable old think-router service.
"""

import json
import http.client
import logging
import math
import os
import re
import sqlite3
import sys
import time
import threading

# Enable logging for submodules (semantic_fewshot, entropy_router)
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
from urllib.parse import urlparse

# Import classifier (sibling module)
sys.path.insert(0, str(Path(__file__).parent))
from classifier import classify
from enricher import enrich as enrich_context
from entropy_router import (estimate_entropy, THRESHOLD_LOW, THRESHOLD_HIGH,
                            W_COMPLEXITY, W_CONFIDENCE, W_HISTORY)
from quality_gate import (should_score, score_response_async, score_response_sync,
                          reflect_and_retry, on_quality_score)
from pipeline_executor import execute_pipeline, should_use_pipeline

# ── Config ───────────────────────────────────────────────────────────────────

LISTEN_PORT = int(os.environ.get("ODO_PORT", "8084"))
LLAMA_BASE = os.environ.get("ODO_BACKEND", "http://127.0.0.1:8081")
FORWARD_TIMEOUT = int(os.environ.get("ODO_TIMEOUT", "300"))

PIPELINES_DIR = Path(__file__).parent / "pipelines"
DB_PATH = Path.home() / ".openclaw/logs/odo.db"

# ── Think Router Config ──────────────────────────────────────────────────────

FORCE_THINK = os.environ.get("FORCE_THINK", "0") == "1"
ENTROPY_THRESHOLD = float(os.environ.get("ENTROPY_THRESHOLD", "0.8"))

PROBE_MAX_TOKENS = 5
PROBE_TOP_LOGPROBS = 5
PROBE_TIMEOUT = 15

# ABF (Adaptive Budget Forcing, ICLR 2026)
ABF_ENABLED = os.environ.get("ABF_ENABLED", "1") == "1"
ABF_ALPHA = float(os.environ.get("ABF_ALPHA", "0.625"))
ABF_BETA = float(os.environ.get("ABF_BETA", "0.375"))
ABF_THRESHOLD = float(os.environ.get("ABF_THRESHOLD", "0.55"))
ABF_MIN_THINKING_CHARS = int(os.environ.get("ABF_MIN_THINKING_CHARS", "100"))
ABF_MAX_RETRIES = int(os.environ.get("ABF_MAX_RETRIES", "3"))
ABF_FALLBACK_MIN_CHARS = int(os.environ.get("ABF_FALLBACK_MIN_CHARS", "500"))
ABF_STREAM_WINDOW = int(os.environ.get("ABF_STREAM_WINDOW", "5"))
LOG_VOCAB_SIZE = 11.93  # log(152064) for Qwen3.5

# CGRS (Certainty-Guided Reasoning Suppression, AAAI 2026)
CGRS_ENABLED = os.environ.get("CGRS_ENABLED", "1") == "1"
CGRS_DELTA = float(os.environ.get("CGRS_DELTA", "0.9"))
CGRS_TRIGGER_IDS = {
    "14190": -100, "13824": -100,   # Wait, ␣Wait
    "11489": -100, "3783": -100,    # wait, ␣wait
    "3983": -100, "1988": -100,     # But, ␣But
    "8088": -100, "714": -100,      # but, ␣but
    "38478": -100,                   # Alternatively
    "75763": -100, "41109": -100,   # Alternative
    "80022": -100, "88190": -100,   # Hmm
}

# Training pair logging
LOG_TRAINING_PAIRS = os.environ.get("LOG_TRAINING_PAIRS", "1") == "1"
TRAINING_PAIRS_PATH = Path.home() / ".openclaw/logs/training_pairs.jsonl"

THINK_MIN_TOKENS = 4096

# ── Sampling Profiles ────────────────────────────────────────────────────────

THINK_PARAMS = {
    "temperature": 1.0, "top_p": 0.95, "top_k": 20,
    "presence_penalty": 0.0, "max_tokens": 16384,
    "chat_template_kwargs": {"enable_thinking": True},
}
NO_THINK_PARAMS = {
    "temperature": 0.7, "top_p": 0.8, "top_k": 20,
    "presence_penalty": 0.0, "max_tokens": 8192,
    "chat_template_kwargs": {"enable_thinking": False},
}
CODE_THINK_PARAMS = {
    "temperature": 0.6, "top_p": 0.95, "top_k": 20,
    "presence_penalty": 0.0, "max_tokens": 16384,
    "chat_template_kwargs": {"enable_thinking": True},
}
CODE_NO_THINK_PARAMS = {
    "temperature": 0.6, "top_p": 0.9, "top_k": 20,
    "presence_penalty": 0.0, "max_tokens": 4096,
    "chat_template_kwargs": {"enable_thinking": False},
}

# ── Generation Modes ──────────────────────────────────────────────────────────
# API payload: "mode": "fast" | "quality" | "ultra" (default: "fast")

MODE_CONFIGS = {
    "fast": {
        "thinking_budget": 2048,
        "abf_threshold": 0.55,
        "max_tokens": 2048,
        "dvts_enabled": False,
        "dvts_k": 0,
        "pipeline_auto": False,
        "web_enrich": False,
        "confidence_probe": False,
        "dynamic_engram": False,
        "timeout_guidance": 60,
    },
    "quality": {
        "thinking_budget": 4096,
        "abf_threshold": 0.70,
        "max_tokens": 4096,
        "dvts_enabled": False,   # entropy router decides
        "dvts_k": 2,
        "pipeline_auto": False,
        "web_enrich": False,
        "confidence_probe": True,
        "dynamic_engram": True,
        "timeout_guidance": 180,
    },
    "ultra": {
        "thinking_budget": 8192,
        "abf_threshold": 0.0,
        "max_tokens": 8192,
        "dvts_enabled": True,
        "dvts_k": 4,
        "pipeline_auto": True,
        "web_enrich": True,
        "confidence_probe": True,
        "dynamic_engram": True,
        "timeout_guidance": 600,
    },
}

VALID_MODES = frozenset(MODE_CONFIGS.keys())

# ── Regex ─────────────────────────────────────────────────────────────────────

CODE_RE = re.compile(
    r'(code|script|program|fonction|function|class|debug|fix|refactor'
    r'|bug|error|compile|import\s|variable|python|bash|javascript|rust|sql'
    r'|api\b|endpoint|fastapi|django|flask|express|serveur|server|docker'
    r'|fichier\s+\w*\.py|cree\s+un\s+\w*script|html|css|react|vue)',
    re.I
)

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

GREETING_RE = re.compile(
    r'^\s*(?:bonjour|salut|hello|hi|hey|coucou|bonsoir|merci|thanks|ok|okay'
    r"|d'accord|ça va|comment (?:ça va|vas-tu|allez-vous)"
    r'|good (?:morning|evening|night|afternoon)|bonne (?:nuit|journée|soirée)'
    r'|au revoir|bye|à\s*\+|bisous?|ciao)\s*[!?.\s]*$',
    re.I
)

# ── Pipeline loading ─────────────────────────────────────────────────────────

_pipeline_cache: dict[str, dict] = {}
_pipeline_mtime: dict[str, float] = {}


def _load_yaml(path: Path) -> dict:
    """Load YAML file. Falls back to basic parser if PyYAML unavailable."""
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        result = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    key, _, val = line.partition(":")
                    val = val.strip()
                    if val.lower() in ("true",):
                        val = True
                    elif val.lower() in ("false",):
                        val = False
                    elif val.lower() in ("null", "none", "~", ""):
                        val = None
                    else:
                        try:
                            val = int(val)
                        except ValueError:
                            try:
                                val = float(val)
                            except ValueError:
                                if (val.startswith('"') and val.endswith('"')) or \
                                   (val.startswith("'") and val.endswith("'")):
                                    val = val[1:-1]
                    result[key.strip()] = val
        return result


def load_pipeline(route_id: str) -> dict:
    """Load pipeline config from YAML with mtime-based hot-reload."""
    yaml_path = PIPELINES_DIR / f"{route_id}.yaml"
    if not yaml_path.exists():
        return {}

    try:
        mtime = yaml_path.stat().st_mtime
    except OSError:
        return {}

    if route_id in _pipeline_cache and mtime <= _pipeline_mtime.get(route_id, 0):
        return dict(_pipeline_cache[route_id])

    try:
        data = _load_yaml(yaml_path)
        _pipeline_cache[route_id] = data
        _pipeline_mtime[route_id] = mtime
        return dict(data)
    except Exception as e:
        print(f"[odo] warning: failed to load {yaml_path}: {e}",
              file=sys.stderr, flush=True)
        return {}


# ── Message helpers ──────────────────────────────────────────────────────────

VALID_ROLES = {"system", "user", "assistant", "tool"}


def extract_user_text(payload: dict) -> str:
    """Extract last user message text."""
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


def sanitize_messages(messages: list) -> list:
    """Sanitize roles and consolidate system messages for Qwen3.5 Jinja."""
    sanitized = []
    for msg in messages:
        role = msg.get("role", "")
        if role in VALID_ROLES:
            sanitized.append(msg)
        elif role in ("tool_result", "tool_response", "function"):
            sanitized.append({**msg, "role": "tool"})
        elif role == "ipython":
            sanitized.append({**msg, "role": "user"})
        else:
            print(f"[odo] WARNING: unknown role '{role}' → 'user'",
                  file=sys.stderr, flush=True)
            sanitized.append({**msg, "role": "user"})

    # Consolidate system messages to position 0
    system_msgs = [m for m in sanitized if m.get("role") == "system"]
    other_msgs = [m for m in sanitized if m.get("role") != "system"]
    if len(system_msgs) > 1:
        merged = "\n\n".join(m.get("content", "") for m in system_msgs if m.get("content"))
        sanitized = [{"role": "system", "content": merged}] + other_msgs
    elif system_msgs and sanitized[0].get("role") != "system":
        sanitized = system_msgs + other_msgs

    return sanitized


def is_code_request(text: str) -> bool:
    return bool(CODE_RE.search(text))


def is_complex_query(text: str) -> bool:
    return bool(COMPLEXITY_RE.search(text))


# ── Tool injection ───────────────────────────────────────────────────────────

TOOL_TRIGGER_KEYWORDS = {
    "web_search": ["recherche", "search", "cherche", "trouve", "actualit", "dernier", "récent", "latest"],
    "calculator": ["calcul", "compute", "combien fait", "IMC", "BMI", "pourcentage", "%"],
}

TOOL_DEFINITIONS = {
    "web_search": {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        },
    },
    "calculator": {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a mathematical calculation",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "Math expression"}},
                "required": ["expression"],
            },
        },
    },
}


def _should_inject_tools(user_text: str, tools_allowed: list) -> bool:
    """Check if the user's query warrants tool injection."""
    text_lower = user_text.lower()
    for tool in tools_allowed:
        keywords = TOOL_TRIGGER_KEYWORDS.get(tool, [])
        if any(kw in text_lower for kw in keywords):
            return True
    return False


def _build_tool_definitions(tools_allowed: list) -> list:
    """Build OpenAI-format tool definitions for allowed tools."""
    return [TOOL_DEFINITIONS[t] for t in tools_allowed if t in TOOL_DEFINITIONS]


# ── Pipeline application ────────────────────────────────────────────────────

def apply_pipeline(payload: dict, pipeline: dict, route_id: str) -> dict:
    """Apply pipeline-specific overrides to the payload.

    Pipeline YAML structure (nested):
      params.temperature, params.max_tokens, etc.
      thinking.enabled, thinking.budget
      engram.table, engram.alpha
      lora.path
      system_prompt (prepended to system message)
    """
    result = dict(payload)
    result.pop("odo_metadata", None)

    # Params overrides (nested under 'params' in YAML)
    params = pipeline.get("params", {})
    if isinstance(params, dict):
        for key in ("temperature", "top_p", "top_k", "max_tokens",
                     "min_p", "presence_penalty"):
            if key in params and params[key] is not None:
                result[key] = params[key]

    # Engram table
    engram = pipeline.get("engram", {})
    if isinstance(engram, dict) and engram.get("table"):
        result["engram_table"] = engram["table"]
        if engram.get("alpha") is not None:
            result["engram_alpha"] = engram["alpha"]

    # LoRA
    lora = pipeline.get("lora", {})
    if isinstance(lora, dict) and lora.get("path"):
        result["lora"] = lora["path"]

    # Grammar
    grammar = pipeline.get("grammar")
    if grammar:
        result["grammar"] = grammar

    # System prompt prefix
    sys_prompt = pipeline.get("system_prompt")
    if sys_prompt and "messages" in result:
        msgs = result["messages"]
        if msgs and msgs[0].get("role") == "system":
            msgs[0]["content"] = f"{sys_prompt.strip()}\n\n{msgs[0]['content']}"
        else:
            msgs.insert(0, {"role": "system", "content": sys_prompt.strip()})

    return result


def pipeline_thinking_override(pipeline: dict) -> bool | None:
    """Get thinking override from pipeline config. None = let ODO decide."""
    thinking = pipeline.get("thinking", {})
    if isinstance(thinking, dict) and "enabled" in thinking:
        return bool(thinking["enabled"])
    return None


def pipeline_abf_threshold(pipeline: dict) -> float | None:
    """Get per-route ABF threshold override."""
    thinking = pipeline.get("thinking", {})
    if isinstance(thinking, dict) and "abf_threshold" in thinking:
        return float(thinking["abf_threshold"])
    return None


# ── ABF / Entropy ────────────────────────────────────────────────────────────

def compute_abf_certainty(logprobs_content: list, window: int = 32) -> float:
    """ABF composite certainty: Ct = alpha*Conf + beta*(1-Ent_normalized)."""
    if not logprobs_content:
        return 0.0
    tokens = logprobs_content[-window:]
    confs, ents = [], []
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
    return ABF_ALPHA * (sum(confs) / len(confs)) + ABF_BETA * (1 - sum(ents) / len(ents))


def token_entropy(logprobs_content: list) -> float:
    """Mean Shannon entropy from top-k logprobs."""
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


# ── Database ─────────────────────────────────────────────────────────────────

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            ts             TEXT NOT NULL DEFAULT (datetime('now')),
            route          TEXT,
            strategy       TEXT,
            confidence     REAL,
            decision       TEXT,
            domain         TEXT,
            probe_entropy  REAL,
            probe_ms       INTEGER,
            total_ms       INTEGER,
            prompt_len     INTEGER,
            sample_prompt  TEXT,
            budget_retries INTEGER DEFAULT 0,
            entropy_class  TEXT,
            entropy_score  REAL
        )
    """)
    # Migrate existing DB: add entropy columns if missing
    for col, ctype in [("entropy_class", "TEXT"), ("entropy_score", "REAL")]:
        try:
            conn.execute(f"ALTER TABLE decisions ADD COLUMN {col} {ctype}")
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()
    conn.close()


def log_decision(data: dict):
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute(
            "INSERT INTO decisions "
            "(ts, route, strategy, confidence, decision, domain, probe_entropy, "
            "probe_ms, total_ms, prompt_len, sample_prompt, budget_retries, "
            "entropy_class, entropy_score) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(),
             data.get("route"), data.get("strategy"), data.get("confidence"),
             data.get("decision"), data.get("domain"),
             data.get("probe_entropy"), data.get("probe_ms"),
             data.get("total_ms"), data.get("prompt_len"),
             data.get("sample_prompt"), data.get("budget_retries", 0),
             data.get("entropy_class"), data.get("entropy_score")),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _log_training_pair(user_text: str, reasoning: str, content: str, retries: int):
    if not LOG_TRAINING_PAIRS:
        return
    try:
        import hashlib
        prompt_hash = hashlib.sha256(user_text.encode()).hexdigest()[:16]
        entry = {
            "ts": datetime.now().isoformat(),
            "prompt": user_text[:2000],
            "reasoning": reasoning[:8000],
            "response": content[:4000],
            "budget_retries": retries,
            "prompt_hash": prompt_hash,
        }
        TRAINING_PAIRS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TRAINING_PAIRS_PATH, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ── HTTP helpers ─────────────────────────────────────────────────────────────

def _send_to_llama(payload: dict, timeout: int = 120) -> dict:
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
    parsed = urlparse(LLAMA_BASE)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    conn.request("POST", path, body=body, headers=headers)
    resp = conn.getresponse()
    return resp, conn


# ── Handler ──────────────────────────────────────────────────────────────────

class ODOHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path == "/health":
            self._json_response(200, {
                "status": "ok",
                "engine": "odo-unified",
                "port": LISTEN_PORT,
                "backend": LLAMA_BASE,
                "force_think": FORCE_THINK,
            })
        elif self.path == "/stats":
            self._send_stats()
        elif self.path == "/routes":
            self._json_response(200, {"routes": self._list_routes()})
        elif self.path.startswith("/v1/models"):
            self._proxy_get()
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        # Non-chat endpoints: transparent proxy
        if not self.path.startswith("/v1/chat/completions"):
            self._proxy_post(body)
            return

        try:
            payload = json.loads(body)
        except Exception:
            self.send_error(400, "Bad JSON")
            return

        t0 = time.time()

        # ── 1. Sanitize messages ──
        if "messages" in payload:
            payload["messages"] = sanitize_messages(payload["messages"])

        user_text = extract_user_text(payload)
        has_img = has_image(payload)

        # ── 2. Classify intent ──
        files = []
        meta = payload.get("odo_metadata", {})
        if meta.get("files"):
            files = meta["files"]

        route_info = classify(user_text, files=files or None, has_image=has_img)
        route_id = route_info["route"]
        route_conf = route_info["confidence"]
        route_strategy = route_info["strategy"]

        # ── 3. Load pipeline ──
        pipeline = load_pipeline(route_id)
        if not pipeline:
            pipeline = load_pipeline("default")

        # ── 3b. Generation mode (fast / quality / ultra) ──
        gen_mode = payload.pop("mode", "fast")
        if gen_mode not in VALID_MODES:
            print(f"[odo] WARNING: unknown mode '{gen_mode}' → 'fast'",
                  file=sys.stderr, flush=True)
            gen_mode = "fast"
        mode_cfg = MODE_CONFIGS[gen_mode]

        # Override thinking budget & ABF threshold in pipeline
        thinking_sec = pipeline.get("thinking", {})
        if not isinstance(thinking_sec, dict):
            thinking_sec = {}
        thinking_sec["budget"] = mode_cfg["thinking_budget"]
        thinking_sec["abf_threshold"] = mode_cfg["abf_threshold"]
        pipeline["thinking"] = thinking_sec

        # Override max_tokens in pipeline params
        params_sec = pipeline.get("params", {})
        if not isinstance(params_sec, dict):
            params_sec = {}
        params_sec["max_tokens"] = mode_cfg["max_tokens"]
        pipeline["params"] = params_sec

        # Ultra: force DVTS always on
        if mode_cfg["dvts_enabled"]:
            pipeline["dvts"] = {"enabled": True, "k": mode_cfg["dvts_k"]}

        # Ultra: force pipeline execution if steps exist
        if mode_cfg["pipeline_auto"] and pipeline.get("pipeline"):
            payload["pipeline"] = True

        # Ultra: force web enrichment
        if mode_cfg["web_enrich"]:
            enrich_sec = pipeline.get("enrich", {})
            if not isinstance(enrich_sec, dict):
                enrich_sec = {}
            enrich_sec["web"] = True
            pipeline["enrich"] = enrich_sec

        # Quality/Ultra: enable dynamic engram (system prompt injection from search)
        if mode_cfg["dynamic_engram"]:
            enrich_sec = pipeline.get("enrich", {})
            if not isinstance(enrich_sec, dict):
                enrich_sec = {}
            enrich_sec["dynamic_engram"] = True
            pipeline["enrich"] = enrich_sec

        if gen_mode != "fast":
            print(f"[odo] mode={gen_mode} budget={mode_cfg['thinking_budget']} "
                  f"abf={mode_cfg['abf_threshold']} max_tok={mode_cfg['max_tokens']} "
                  f"dvts={mode_cfg['dvts_enabled']}(k={mode_cfg['dvts_k']}) "
                  f"pipeline_auto={mode_cfg['pipeline_auto']} web={mode_cfg['web_enrich']} "
                  f"engram={mode_cfg['dynamic_engram']}",
                  file=sys.stderr, flush=True)

        # ── 4. Apply pipeline params ──
        payload = apply_pipeline(payload, pipeline, route_id)

        classify_ms = int((time.time() - t0) * 1000)

        # ── 4b. ENRICH — inject context from tools (RAG, web, CSV, IoC) ──
        # IMPORTANT: enrichment BEFORE tool injection (tool injection was blocking enrichment)
        enrich_info = {"tools_used": [], "enrich_ms": 0, "context_chars": 0}
        caller_has_tools = payload.get("tools") or payload.get("functions")
        skip_enrich = (
            len(user_text) < 30
            or caller_has_tools  # Only skip if CALLER sent tools, not if we inject
            or payload.get("chat_template_kwargs", {}).get("enable_thinking") is False
        )
        if not skip_enrich:
            payload, enrich_info = enrich_context(payload, route_id, user_text, pipeline)
            if enrich_info["tools_used"]:
                print(f"[odo] enrich: {','.join(enrich_info['tools_used'])} "
                      f"{enrich_info['enrich_ms']}ms {enrich_info['context_chars']} chars",
                      file=sys.stderr, flush=True)

        # ── 4a. Inject tools AFTER enrichment (so enrichment isn't skipped) ──
        if not caller_has_tools and not payload.get("tools"):
            enrich_cfg = pipeline.get("enrich", {})
            has_web_enrich = enrich_cfg.get("web", False) if isinstance(enrich_cfg, dict) else False
            # Don't inject tools for routes that do their own web enrichment
            if not has_web_enrich:
                tools_allowed = pipeline.get("tools_allowed", [])
                if tools_allowed and _should_inject_tools(user_text, tools_allowed):
                    payload["tools"] = _build_tool_definitions(tools_allowed)
                    payload["tool_choice"] = "auto"

        # ── 4c. PIPELINE EXECUTION — sequential multi-agent steps ──
        if should_use_pipeline(pipeline, payload):
            steps = pipeline["pipeline"]
            system_prompt = pipeline.get("system_prompt", "")
            think_override = pipeline_thinking_override(pipeline)
            thinking = think_override if think_override is not None else (FORCE_THINK or True)

            print(f"[odo] PIPELINE: route={route_id} mode={gen_mode} steps={len(steps)} "
                  f"thinking={thinking} classify={classify_ms}ms backend={LLAMA_BASE}",
                  file=sys.stderr, flush=True)

            result = execute_pipeline(
                steps=steps,
                user_text=user_text,
                system_prompt=system_prompt,
                backend_url=LLAMA_BASE,
                thinking_enabled=thinking,
                timeout=FORWARD_TIMEOUT,
            )

            # Build OpenAI-compatible response
            resp_data = {
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": result["content"]},
                    "finish_reason": "stop",
                }],
                "usage": {"total_tokens": result["total_tokens"]},
                "odo": {
                    "route": route_id,
                    "pipeline": True,
                    "steps": result["steps_log"],
                    "pipeline_ms": result["total_ms"],
                    "partial": result.get("partial", False),
                },
            }
            resp_body = json.dumps(resp_data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)

            total_ms = int((time.time() - t0) * 1000)
            print(f"[odo] PIPELINE complete: route={route_id} "
                  f"{len(steps)} steps {total_ms}ms",
                  file=sys.stderr, flush=True)

            # Log decision
            odo_meta = {
                "route": route_id, "strategy": route_strategy,
                "confidence": route_conf, "decision": f"pipeline/{gen_mode}",
                "domain": "pipeline", "probe_entropy": None,
                "probe_ms": 0, "total_ms": total_ms,
                "prompt_len": len(user_text),
                "sample_prompt": user_text[:200], "budget_retries": 0,
            }
            threading.Thread(target=log_decision, daemon=True, args=(odo_meta,)).start()
            return

        # ── 4d. ENTROPY ROUTER — pre-generation entropy classification ──
        entropy_info = estimate_entropy(user_text, route_id, route_conf)
        entropy_class = entropy_info["entropy_class"]
        entropy_score = entropy_info["entropy_score"]

        # Apply entropy router recommendations to pipeline/payload:
        #   low  → hint no-think (skip expensive probe)
        #   high → inject DVTS K=2 if not already configured, tighten ABF
        if entropy_class == "low":
            payload["_entropy_hint"] = "no-think"
        elif entropy_class == "high":
            # Inject DVTS into pipeline if not already enabled
            if not pipeline.get("dvts", {}).get("enabled"):
                pipeline["dvts"] = {"enabled": True, "k": 2}
            # Tighten ABF threshold for high-entropy queries
            action = entropy_info["action"]
            if action.get("abf_threshold") is not None:
                thinking_cfg = pipeline.get("thinking", {})
                if not isinstance(thinking_cfg, dict):
                    thinking_cfg = {}
                thinking_cfg["abf_threshold"] = action["abf_threshold"]
                pipeline["thinking"] = thinking_cfg

        # ── 5. Determine thinking mode ──
        # Priority: explicit caller override > pipeline > entropy hint > FORCE_THINK > probe
        decision, domain, params, probe_entropy, probe_ms = \
            self._decide_thinking(payload, user_text, has_img, pipeline)

        enrich_tag = ""
        if enrich_info["tools_used"]:
            enrich_tag = f" enrich={','.join(enrich_info['tools_used'])}({enrich_info['enrich_ms']}ms)"
        print(f"[odo] route={route_id} mode={gen_mode} conf={route_conf:.2f} "
              f"strategy={route_strategy} decision={decision} domain={domain} "
              f"entropy={entropy_class}({entropy_score:.3f}) "
              f"classify={classify_ms}ms{enrich_tag} backend={LLAMA_BASE}",
              file=sys.stderr, flush=True)

        # ── 6. Forward with params ──
        self._forward_with_params(
            payload, params, decision, domain, route_id, route_strategy,
            route_conf, t0, user_text, pipeline,
            probe_entropy=probe_entropy, probe_ms=probe_ms,
            entropy_info=entropy_info, gen_mode=gen_mode,
        )

    def _decide_thinking(self, payload, user_text, has_img, pipeline):
        """Decide thinking mode. Returns (decision, domain, params, probe_entropy, probe_ms)."""
        code = is_code_request(user_text)
        domain = "code" if code else "general"

        # 1. Tool calls → always no-think
        if payload.get("tools") or payload.get("functions"):
            return "no-think", "tools", CODE_NO_THINK_PARAMS, None, 0

        # 2. Explicit caller override (chat_template_kwargs already set)
        if "chat_template_kwargs" in payload:
            return "caller-override", domain, {}, None, 0

        # 3. Vision → always think
        if has_img:
            params = CODE_THINK_PARAMS if code else THINK_PARAMS
            return "think", "vision", params, None, 0

        # 4. Very short → no-think
        if len(user_text) < 20:
            return "no-think", "short", NO_THINK_PARAMS, None, 0

        # 5. Greeting → no-think fast path
        if len(user_text) < 80 and GREETING_RE.match(user_text):
            return "no-think", "greeting", NO_THINK_PARAMS, None, 0

        # 6. Pipeline override
        think_override = pipeline_thinking_override(pipeline)
        if think_override is not None:
            if think_override:
                params = CODE_THINK_PARAMS if code else THINK_PARAMS
                return "think", domain, params, None, 0
            else:
                params = CODE_NO_THINK_PARAMS if code else NO_THINK_PARAMS
                return "no-think", domain, params, None, 0

        # 7. Entropy router hint (pre-generation heuristic, skips expensive probe)
        entropy_hint = payload.pop("_entropy_hint", None)
        if entropy_hint == "no-think" and not FORCE_THINK:
            params = CODE_NO_THINK_PARAMS if code else NO_THINK_PARAMS
            return "no-think", "entropy-low", params, None, 0

        # 8. FORCE_THINK env
        if FORCE_THINK:
            params = CODE_THINK_PARAMS if code else THINK_PARAMS
            return "think", domain, params, None, 0

        # 9. Entropy probe (only when not force-think)
        return self._entropy_probe(payload, user_text, code, domain)

    def _entropy_probe(self, payload, user_text, code, domain):
        """Run entropy probe to decide thinking mode."""
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
            t = time.time()
            result = _send_to_llama(probe_payload, timeout=PROBE_TIMEOUT)
            probe_ms = int((time.time() - t) * 1000)
        except Exception as e:
            print(f"[odo] probe failed ({e}), defaulting to think",
                  file=sys.stderr, flush=True)
            params = CODE_THINK_PARAMS if code else THINK_PARAMS
            return "think", "probe-fail", params, None, 0

        choice = result.get("choices", [{}])[0]
        lp_data = choice.get("logprobs", {})
        tok_list = lp_data.get("content", []) if lp_data else []
        entropy = token_entropy(tok_list) if tok_list else 999.0

        if entropy > ENTROPY_THRESHOLD:
            params = CODE_THINK_PARAMS if code else THINK_PARAMS
            decision = "think"
        else:
            params = CODE_NO_THINK_PARAMS if code else NO_THINK_PARAMS
            decision = "no-think"

        print(f"[odo] entropy={entropy:.3f} → {decision} ({domain}) probe={probe_ms}ms",
              file=sys.stderr, flush=True)
        return decision, domain, params, entropy, probe_ms

    def _forward_with_params(self, payload, params, decision, domain,
                             route_id, route_strategy, route_conf,
                             t0, user_text, pipeline,
                             probe_entropy=None, probe_ms=0,
                             entropy_info=None, gen_mode="fast"):
        """Apply sampling params and forward. Integrates ABF."""
        real_payload = dict(payload)
        # Clean internal entropy hint before forwarding to backend
        real_payload.pop("_entropy_hint", None)
        is_thinking = params.get("chat_template_kwargs", {}).get("enable_thinking", False)
        is_streaming = payload.get("stream", False)

        # Apply params (preserve caller's stream)
        for k, v in params.items():
            if k == "max_tokens":
                caller_max = payload.get("max_tokens")
                merged = min(caller_max, v) if caller_max else v
                if is_thinking:
                    merged = max(merged, THINK_MIN_TOKENS)
                real_payload[k] = merged
            else:
                real_payload[k] = v

        # Remove internal fields
        real_payload.pop("odo_metadata", None)
        real_payload.pop("odo_route", None)

        # ABF for complex thinking queries
        budget_retries = 0
        complex_q = is_complex_query(user_text)
        abf_threshold = pipeline_abf_threshold(pipeline) or ABF_THRESHOLD
        needs_abf = (ABF_ENABLED and is_thinking and complex_q and len(user_text) > 30)

        # DVTS: Diverse Verifier Tree Search (generate K candidates, score, return best)
        dvts_cfg = pipeline.get("dvts", {})
        if isinstance(dvts_cfg, dict) and dvts_cfg.get("enabled") and not is_streaming:
            try:
                from dvts import dvts_generate
                dvts_k = dvts_cfg.get("k", 4)
                dvts_result = dvts_generate(
                    messages=real_payload.get("messages", []),
                    user_text=user_text,
                    route_id=route_id,
                    k=dvts_k,
                    max_tokens=real_payload.get("max_tokens", 4096),
                )
                # Wrap in OpenAI-compatible response format
                resp_body = json.dumps({
                    "choices": [{"message": {
                        "role": "assistant",
                        "content": dvts_result["content"],
                        "reasoning_content": dvts_result.get("reasoning", ""),
                    }, "finish_reason": "stop", "index": 0}],
                    "usage": {"completion_tokens": 0},
                    "dvts": {
                        "score": dvts_result["score"],
                        "candidates": len(dvts_result.get("candidates", [])),
                        "dvts_ms": dvts_result["dvts_ms"],
                    },
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
                total_ms = int((time.time() - t0) * 1000)
                print(f"[odo] DVTS: route={route_id} k={dvts_k} "
                      f"score={dvts_result['score']:.3f} {total_ms}ms",
                      flush=True)
                return
            except Exception as e:
                print(f"[odo] DVTS failed, falling back: {e}", flush=True)

        if needs_abf and not is_streaming:
            result, budget_retries = self._abf_monitor(real_payload, user_text, abf_threshold)
            resp_body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)
        else:
            # SSE ABF monitoring (informational)
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
                self._stream_response(resp, conn, abf_monitor=abf_stream,
                                      user_text=user_text, route_id=route_id)
            else:
                self._buffer_response(resp, conn, user_text=user_text, route_id=route_id)

        total_ms = int((time.time() - t0) * 1000)

        # Log decision with ODO metadata (includes entropy classification)
        odo_meta = {
            "route": route_id,
            "strategy": route_strategy,
            "confidence": route_conf,
            "decision": f"{decision}/{gen_mode}",
            "domain": domain,
            "probe_entropy": probe_entropy,
            "probe_ms": probe_ms,
            "total_ms": total_ms,
            "prompt_len": len(user_text),
            "sample_prompt": user_text[:200],
            "budget_retries": budget_retries,
        }
        if entropy_info:
            odo_meta["entropy_class"] = entropy_info.get("entropy_class")
            odo_meta["entropy_score"] = entropy_info.get("entropy_score")
            odo_meta["entropy_components"] = entropy_info.get("components")
        threading.Thread(target=log_decision, daemon=True, args=(odo_meta,)).start()

    def _abf_monitor(self, payload, user_text, threshold=None):
        """ABF inline monitor: stream internally, compute Ct, retry if needed."""
        threshold = threshold or ABF_THRESHOLD
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
                print(f"[odo] ABF: backend error ({e})", file=sys.stderr, flush=True)
                return {"choices": [{"message": {"content": f"Error: {e}", "role": "assistant"}}]}, attempt

            reasoning_buf, content_buf, logprob_entries = [], [], []
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

            ct = compute_abf_certainty(logprob_entries) if logprob_entries else 0.0

            accepted = False
            if ct >= threshold and reasoning_len >= ABF_MIN_THINKING_CHARS:
                accepted = True
            elif reasoning_len >= ABF_FALLBACK_MIN_CHARS:
                accepted = True
            elif attempt >= ABF_MAX_RETRIES:
                accepted = True

            if accepted:
                tag = "accepted" if attempt == 0 else f"accepted after {attempt} retries"
                print(f"[odo] ABF: {tag} (Ct={ct:.3f}, {reasoning_len} chars)",
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

            print(f"[odo] ABF: attempt {attempt+1}/{ABF_MAX_RETRIES}, "
                  f"Ct={ct:.3f} < {threshold}, {reasoning_len} chars, injecting Wait",
                  file=sys.stderr, flush=True)

            messages = list(original_messages)
            prefill = f"<think>\n{reasoning_text}\nWait, let me reconsider this step by step.\n"
            messages.append({"role": "assistant", "content": prefill})
            work_payload["messages"] = messages

            if CGRS_ENABLED and ct > CGRS_DELTA:
                work_payload["logit_bias"] = CGRS_TRIGGER_IDS

        return {
            "choices": [{"message": {"role": "assistant", "content": "".join(content_buf)},
                         "finish_reason": "stop"}]
        }, ABF_MAX_RETRIES

    def _stream_response(self, resp, conn, abf_monitor=False,
                         user_text=None, route_id=None):
        """Stream SSE with optional ABF monitoring + post-stream training/quality logging."""
        self.send_response(resp.status)
        self.send_header("Content-Type", resp.getheader("Content-Type", "text/event-stream"))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        abf_window = []
        abf_halted = False
        in_thinking = False
        thinking_chars = 0

        # Accumulate content for post-stream logging and quality gate
        reasoning_parts = []
        content_parts = []

        try:
            while True:
                line = resp.readline()
                if not line:
                    break
                self.wfile.write(line)
                self.wfile.flush()

                # Parse SSE chunks to accumulate content
                stripped = line.strip()
                if stripped == b'data: [DONE]' or not stripped.startswith(b'data: '):
                    continue
                try:
                    chunk = json.loads(stripped[6:])
                except Exception:
                    continue
                choice = chunk.get("choices", [{}])[0]
                delta = choice.get("delta", {})

                rc = delta.get("reasoning_content")
                if rc:
                    in_thinking = True
                    thinking_chars += len(rc)
                    reasoning_parts.append(rc)
                ct_content = delta.get("content")
                if ct_content:
                    in_thinking = False
                    content_parts.append(ct_content)

                if abf_monitor and not abf_halted:
                    if not in_thinking or thinking_chars < ABF_MIN_THINKING_CHARS:
                        continue

                    lp_data = choice.get("logprobs")
                    if not lp_data:
                        continue
                    lp_content = lp_data.get("content", [])
                    for tok_entry in lp_content:
                        top_logprobs = tok_entry.get("top_logprobs", [])
                        if not top_logprobs:
                            continue
                        top1_logprob = tok_entry.get("logprob", top_logprobs[0].get("logprob", -10.0))

                        probs = [math.exp(lp["logprob"]) for lp in top_logprobs if "logprob" in lp]
                        if not probs:
                            continue
                        total = sum(probs)
                        if total <= 0:
                            continue
                        probs = [p / total for p in probs]

                        entropy = -sum(p * math.log(p + 1e-10) for p in probs if p > 0)
                        max_entropy = math.log(len(probs) + 1e-10)
                        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

                        ct = ABF_ALPHA * math.exp(top1_logprob) + ABF_BETA * (1.0 - norm_entropy)
                        abf_window.append(ct)
                        if len(abf_window) > ABF_STREAM_WINDOW:
                            abf_window = abf_window[-ABF_STREAM_WINDOW:]

                        if len(abf_window) == ABF_STREAM_WINDOW:
                            avg_ct = sum(abf_window) / ABF_STREAM_WINDOW
                            if avg_ct > ABF_THRESHOLD:
                                abf_halted = True
                                print(f"[odo] ABF stream: Ct={avg_ct:.3f} > {ABF_THRESHOLD} "
                                      f"(thinking_chars={thinking_chars})",
                                      file=sys.stderr, flush=True)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            conn.close()

            # Post-stream: log training pair and quality gate (async, non-blocking)
            try:
                reasoning_text = "".join(reasoning_parts)
                content_text = "".join(content_parts)

                if user_text and content_text:
                    _log_training_pair(user_text, reasoning_text or "", content_text, 0)

                if (user_text and content_text and route_id
                        and should_score(route_id, content_text, False)):
                    score_response_async(user_text, content_text, route_id,
                                         callback=on_quality_score)
            except Exception as e:
                print(f"[odo] stream post-process error: {e}",
                      file=sys.stderr, flush=True)

    def _buffer_response(self, resp, conn, user_text=None, route_id=None):
        """Buffer response, strip reasoning_content, log training pair, quality gate + reflection."""
        try:
            resp_body = resp.read()
            status = resp.status
        finally:
            conn.close()
        response_content = ""
        try:
            data = json.loads(resp_body)
            for choice in data.get("choices", []):
                msg = choice.get("message", {})
                reasoning = msg.get("reasoning_content", "")
                content = msg.get("content", "")
                response_content = content
                if reasoning and user_text:
                    _log_training_pair(user_text, reasoning, content, 0)
                if "reasoning_content" in msg:
                    del msg["reasoning_content"]
            resp_body = json.dumps(data).encode()
        except Exception:
            pass

        # Inject ODO metadata into response (confidence surfacing)
        try:
            data = json.loads(resp_body)
            # Only if it's a valid chat completion response
            if "choices" in data:
                data["odo"] = {
                    "route": route_id,
                    "enriched": bool(route_id),  # enrichment happened
                }
            resp_body = json.dumps(data).encode()
        except Exception:
            pass

        # Reflection loop: for critical routes, score BEFORE sending
        # If score <= 2, retry with self-critique and send the improved version
        REFLECT_ROUTES = {"kine", "cyber"}
        if (route_id in REFLECT_ROUTES and user_text
                and should_score(route_id, response_content, False)):
            score, reason = score_response_sync(user_text, response_content)
            if score <= 2:
                print(f"[odo] reflection: score={score}, retrying...",
                      file=sys.stderr, flush=True)
                improved = reflect_and_retry(user_text, response_content, reason)
                if improved:
                    # Replace the response
                    try:
                        data = json.loads(resp_body)
                        data["choices"][0]["message"]["content"] = improved
                        data["choices"][0]["reflection"] = {
                            "original_score": score,
                            "reason": reason,
                            "retried": True,
                        }
                        resp_body = json.dumps(data).encode()
                        response_content = improved
                        print(f"[odo] reflection: improved ({len(improved)} chars)",
                              file=sys.stderr, flush=True)
                    except Exception:
                        pass
            else:
                # Good score — async log only
                threading.Thread(target=on_quality_score, daemon=True,
                                 args=(score, reason, route_id,
                                       user_text, response_content)).start()

        try:
            self.send_response(status)
            self.send_header("Content-Type", resp.getheader("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)
        except (BrokenPipeError, ConnectionResetError):
            # Client disconnected before we finished sending — not an error
            print(f"[odo] client disconnected during buffer response (route={route_id})",
                  file=sys.stderr, flush=True)
            return

        # Quality gate for non-reflect routes (async, non-blocking)
        if (route_id and route_id not in REFLECT_ROUTES and user_text
                and should_score(route_id, response_content, False)):
            score_response_async(user_text, response_content, route_id,
                                 callback=on_quality_score)

    def _proxy_post(self, body):
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

    def _list_routes(self) -> dict:
        routes = {}
        for yaml_file in sorted(PIPELINES_DIR.glob("*.yaml")):
            rid = yaml_file.stem
            try:
                cfg = load_pipeline(rid)
                routes[rid] = {
                    "name": cfg.get("name", rid),
                    "thinking": pipeline_thinking_override(cfg),
                    "engram": bool(cfg.get("engram", {}).get("table")),
                    "lora": bool(cfg.get("lora", {}).get("path")),
                }
            except Exception:
                routes[rid] = {"configured": False}
        return routes

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
                FROM decisions WHERE ts > datetime('now', '-24 hours')
            """).fetchone()

            # Route distribution
            route_rows = conn.execute("""
                SELECT route, COUNT(*), AVG(total_ms)
                FROM decisions WHERE ts > datetime('now', '-24 hours')
                GROUP BY route ORDER BY COUNT(*) DESC
            """).fetchall()

            # Entropy router distribution
            entropy_rows = conn.execute("""
                SELECT entropy_class, COUNT(*), AVG(entropy_score)
                FROM decisions WHERE ts > datetime('now', '-24 hours')
                  AND entropy_class IS NOT NULL
                GROUP BY entropy_class
            """).fetchall()
            conn.close()

            total = rows[0] or 0
            think_count = rows[2] or 0
            budget_count = rows[5] or 0
            entropy_dist = {r[0]: {"count": r[1], "avg_score": round(r[2], 3)}
                           for r in entropy_rows}
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
                "routes": {r[0]: {"count": r[1], "avg_ms": round(r[2])} for r in route_rows},
                "entropy_router": entropy_dist,
                "config": {
                    "force_think": FORCE_THINK,
                    "abf_enabled": ABF_ENABLED,
                    "abf_threshold": ABF_THRESHOLD,
                    "abf_alpha": ABF_ALPHA,
                    "abf_beta": ABF_BETA,
                    "abf_stream_window": ABF_STREAM_WINDOW,
                    "cgrs_enabled": CGRS_ENABLED,
                    "cgrs_delta": CGRS_DELTA,
                    "entropy_threshold": ENTROPY_THRESHOLD,
                    "log_training_pairs": LOG_TRAINING_PAIRS,
                },
            })
        except Exception as e:
            self._json_response(500, {"error": str(e)})

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()


# ── Server ───────────────────────────────────────────────────────────────────

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    init_db()
    PIPELINES_DIR.mkdir(parents=True, exist_ok=True)

    server = ThreadedHTTPServer(("127.0.0.1", LISTEN_PORT), ODOHandler)
    n_pipelines = len(list(PIPELINES_DIR.glob("*.yaml")))

    print(f"[odo] listening on 127.0.0.1:{LISTEN_PORT}", flush=True)
    print(f"[odo] backend: {LLAMA_BASE}", flush=True)
    print(f"[odo] pipelines: {n_pipelines} loaded from {PIPELINES_DIR}", flush=True)
    print(f"[odo] force_think: {FORCE_THINK}", flush=True)
    print(f"[odo] ABF: enabled={ABF_ENABLED} threshold={ABF_THRESHOLD} "
          f"alpha={ABF_ALPHA} beta={ABF_BETA} retries={ABF_MAX_RETRIES} "
          f"window={ABF_STREAM_WINDOW}", flush=True)
    print(f"[odo] CGRS: enabled={CGRS_ENABLED} delta={CGRS_DELTA} "
          f"triggers={len(CGRS_TRIGGER_IDS)}", flush=True)
    print(f"[odo] entropy_router: thresholds low<={THRESHOLD_LOW} high>={THRESHOLD_HIGH} "
          f"weights cx={W_COMPLEXITY} cf={W_CONFIDENCE} hx={W_HISTORY}", flush=True)
    print(f"[odo] training logging: {LOG_TRAINING_PAIRS} → {TRAINING_PAIRS_PATH}", flush=True)
    print(f"[odo] stats: curl http://127.0.0.1:{LISTEN_PORT}/stats", flush=True)
    print(f"[odo] routes: curl http://127.0.0.1:{LISTEN_PORT}/routes", flush=True)

    # Warmup semantic few-shot FAISS index (background, non-blocking)
    try:
        import threading
        from semantic_fewshot import warmup as _sf_warmup
        threading.Thread(target=_sf_warmup, daemon=True).start()
        print("[odo] semantic_fewshot: warming up in background", flush=True)
    except Exception as e:
        print(f"[odo] semantic_fewshot: disabled ({e})", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[odo] shutting down", flush=True)
        server.shutdown()


if __name__ == "__main__":
    main()
