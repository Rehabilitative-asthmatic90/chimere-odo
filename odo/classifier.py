#!/usr/bin/env python3
"""
ODO Classifier — Intent classification with cascading strategies.

Strategy 1: Keyword/regex matching (0ms, handles ~99% of requests)
Strategy 2: File-type detection (presence of images, code files, data files)
Strategy 3: LLM fallback with GBNF constraint (~50ms, <1% of requests)

Returns: {"route": str, "confidence": float, "strategy": str}
"""

import json
import http.client
import os
import re
from urllib.parse import urlparse

# ── Route definitions (regex patterns) ───────────────────────────────────────

ROUTES = {
    "code": re.compile(
        r"(?i)(code|function|class|debug|compile|rust|python|import\b|def\s|fn\s|impl\s"
        r"|script|program|refactor|bug\b|error\b|snippet|variable|api\b|endpoint"
        r"|\.py\b|\.rs\b|\.js\b|\.ts\b|\.kt\b|\.java\b|\.go\b|\.cpp\b|\.c\b"
        r"|docker|git\b|commit|branch|merge|cmake|cargo|npm|pip\b)"
    ),
    "kine": re.compile(
        r"(?i)(patient|kin[eé]|r[eé][eé]ducation|entorse|rachis|HAS\b|bilan"
        r"|douleur|articul|muscle|mobilit[eé]|amplitude|s[eé]ance|th[eé]rapeu"
        r"|lombalgie|cervicalgie|tendin|fracture|proth[eè]se|genou|[eé]paule|hanche)"
    ),
    "agro": re.compile(
        r"(?i)(sol\b|culture\b|biocarburant|rendement|parcelle|CSTJF|agro"
        r"|semis|r[eé]colte|engrais|irrigation|ph[yf]tosanitaire|adventice"
        r"|assolement|couvert|rotation|mati[eè]re.organique|humus)"
    ),
    "cyber": re.compile(
        r"(?i)(CVE|vuln[eé]rabilit[eé]|firewall|IOC\b|Suricata|MITRE|exploit"
        r"|malware|ransomware|phishing|SIEM|YARA|sigma\b|logstash"
        r"|incident|forensic|hash\b|indicator|threat|APT\b|C2\b)"
    ),
    "data": re.compile(
        r"(?i)(csv\b|xlsx?\b|corr[eé]lation|graphique|tendance|dataset|pandas"
        r"|dataframe|statistique|r[eé]gression|histogramme|scatter|boxplot"
        r"|moyenne|m[eé]diane|[eé]cart.type|tableau\b|pivot|merge.*data)"
    ),
    "research": re.compile(
        r"(?i)(recherche|compare[rz]?\b|[eé]tat.de.l.art|rapport|analyse\b"
        r"|synth[eè]se|litt[eé]rature|source|r[eé]f[eé]rence|citation"
        r"|[eé]tude|publi|article|paper|survey|review\b|benchmark\b)"
    ),
    "tutor": re.compile(
        r"(?i)(explique|comment.fonctionne|c.est.quoi|apprends|d[eé]fini"
        r"|diff[eé]rence.entre|pourquoi\b|qu.est.ce|enseigne|cours\b"
        r"|tuto|example|illustre|simplifie|vulgarise)"
    ),
    # vision and doc_qa have no regex — detected by file presence only
}

# File extension to route mapping
EXT_ROUTES = {
    ".py": "code", ".rs": "code", ".kt": "code", ".js": "code",
    ".ts": "code", ".tsx": "code", ".jsx": "code", ".java": "code",
    ".go": "code", ".cpp": "code", ".c": "code", ".h": "code",
    ".rb": "code", ".swift": "code", ".zig": "code",
    ".csv": "data", ".xlsx": "data", ".xls": "data", ".tsv": "data",
    ".parquet": "data", ".json": "data",
    ".pdf": "doc_qa", ".docx": "doc_qa", ".doc": "doc_qa",
    ".odt": "doc_qa", ".txt": "doc_qa", ".epub": "doc_qa",
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}

# GBNF grammar for LLM fallback — constrains output to a valid route name
ROUTE_NAMES = list(ROUTES.keys()) + ["vision", "doc_qa", "general"]
GBNF_GRAMMAR = 'root ::= ' + ' | '.join(f'"{r}"' for r in ROUTE_NAMES)

# LLM classification prompt (kept minimal for speed)
CLASSIFY_SYSTEM = (
    "You are an intent classifier. Given a user message, respond with exactly one word: "
    "the intent category. Categories: " + ", ".join(ROUTE_NAMES) + ". "
    "Respond with ONLY the category name, nothing else."
)

LLM_BACKEND = os.environ.get("ODO_LLM_BACKEND", "http://127.0.0.1:8081")
LLM_TIMEOUT = int(os.environ.get("ODO_LLM_TIMEOUT", "5"))


# ── Strategy 1: Keyword/regex matching ───────────────────────────────────────

def _classify_keyword(text: str) -> tuple[str | None, float]:
    """Match against route regexes. Returns (route, confidence) or (None, 0)."""
    scores = {}
    for route, pattern in ROUTES.items():
        if pattern is None:
            continue
        matches = pattern.findall(text)
        if matches:
            scores[route] = len(matches)

    if not scores:
        return None, 0.0

    best = max(scores, key=scores.get)
    n_matches = scores[best]

    # Confidence based on match count and exclusivity
    if len(scores) == 1:
        confidence = min(0.95, 0.7 + 0.05 * n_matches)
    else:
        # Multiple routes matched — lower confidence for the winner
        second_best = sorted(scores.values(), reverse=True)[1]
        gap = n_matches - second_best
        confidence = min(0.85, 0.5 + 0.05 * gap)

    return best, confidence


# ── Strategy 2: File-type detection ──────────────────────────────────────────

def _classify_files(files: list[str] | None, has_image: bool = False) -> tuple[str | None, float]:
    """Classify based on attached file extensions or image presence."""
    if has_image:
        return "vision", 0.9

    if not files:
        return None, 0.0

    ext_votes = {}
    for f in files:
        # Extract extension (handle paths and URLs)
        name = f.rsplit("/", 1)[-1] if "/" in f else f
        dot_idx = name.rfind(".")
        if dot_idx < 0:
            continue
        ext = name[dot_idx:].lower()

        if ext in IMAGE_EXTS:
            return "vision", 0.9

        route = EXT_ROUTES.get(ext)
        if route:
            ext_votes[route] = ext_votes.get(route, 0) + 1

    if not ext_votes:
        return None, 0.0

    best = max(ext_votes, key=ext_votes.get)
    return best, 0.85


# ── Strategy 3: LLM fallback ────────────────────────────────────────────────

def _classify_llm(text: str) -> tuple[str, float]:
    """Send a short classification request to the LLM with GBNF grammar."""
    payload = {
        "model": "qwen3.5",
        "messages": [
            {"role": "system", "content": CLASSIFY_SYSTEM},
            {"role": "user", "content": text[:500]},  # truncate for speed
        ],
        "max_tokens": 20,
        "temperature": 0.1,
        "top_p": 0.5,
        "stream": False,
        "grammar": GBNF_GRAMMAR,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    try:
        parsed = urlparse(LLM_BACKEND)
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=LLM_TIMEOUT)
        body = json.dumps(payload).encode()
        conn.request("POST", "/v1/chat/completions", body=body, headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        })
        resp = conn.getresponse()
        data = json.loads(resp.read())
        conn.close()

        answer = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip().lower()
        if answer in ROUTE_NAMES:
            return answer, 0.7
        return "general", 0.3
    except Exception:
        return "general", 0.3


# ── Public API ───────────────────────────────────────────────────────────────

def classify(text: str, files: list[str] | None = None,
             has_image: bool = False) -> dict:
    """Classify intent using cascading strategies.

    Args:
        text: The user message text.
        files: Optional list of filenames or paths attached to the request.
        has_image: Whether the request contains base64/URL images.

    Returns:
        {"route": str, "confidence": float, "strategy": str}
    """
    # Strategy 1 — keyword/regex
    route, conf = _classify_keyword(text)
    if route and conf >= 0.5:
        return {"route": route, "confidence": conf, "strategy": "keyword"}

    # Strategy 2 — file type
    file_route, file_conf = _classify_files(files, has_image)
    if file_route and file_conf >= 0.5:
        # If keyword also had a match (but low confidence), boost if same route
        if route == file_route:
            file_conf = min(0.95, file_conf + 0.1)
        return {"route": file_route, "confidence": file_conf, "strategy": "filetype"}

    # If keyword had a weak match, use it over LLM
    if route and conf > 0.0:
        return {"route": route, "confidence": conf, "strategy": "keyword_weak"}

    # Strategy 3 — LLM fallback
    llm_route, llm_conf = _classify_llm(text)
    return {"route": llm_route, "confidence": llm_conf, "strategy": "llm"}


# ── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: classifier.py <message> [file1 file2 ...]")
        sys.exit(1)
    msg = sys.argv[1]
    attached = sys.argv[2:] if len(sys.argv) > 2 else None
    result = classify(msg, files=attached)
    print(json.dumps(result, indent=2))
