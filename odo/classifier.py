#!/usr/bin/env python3
"""
ODO Classifier — Intent classification with cascading strategies.

Strategy 1: Keyword/regex matching (0ms, handles ~99% of requests)
Strategy 1b: General fast-path for greetings/chitchat (0ms)
Strategy 2: File-type detection (presence of images, code files, data files)
Strategy 3: LLM fallback with GBNF constraint (~50ms, <1% of requests)
           Uses nothink proxy (port 8086) for fast single-token output.

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
        r"|docker|git\b|commit|branch|merge|cmake|cargo|npm|pip\b"
        r"|sql\b|query\b|database|bash\b|shell\b|terminal|command|syntax"
        r"|algorithm|array\b|liste?\b|dict\b|object\b|module|library|package"
        r"|test\b|unittest|pytest|lint|type.?hint|annotation|decorator"
        r"|async\b|await\b|thread|mutex|race.condition|memory.leak"
        r"|kubernetes|k8s\b|helm\b|terraform|ansible|makefile|\.sh\b)"
    ),
    "kine": re.compile(
        r"(?i)(patient|kin[eé]|r[eé][eé]ducation|entorse|rachis|HAS\b|bilan"
        r"|douleur|articul|muscle|mobilit[eé]|amplitude|s[eé]ance|th[eé]rapeu"
        r"|lombalgie|cervicalgie|tendin|fracture|proth[eè]se|genou|[eé]paule|hanche"
        r"|poignet|cheville|coude|colonne|vert[eè]bre|ligament|nerf\b"
        r"|raideur|contracture|crampe|posture|ergonomie|massage|manipulation"
        r"|reeducation|kinesith[eé]rapie|physiother|sport\b|traumatisme)"
    ),
    "agro": re.compile(
        r"(?i)(sol\b|culture\b|biocarburant|rendement|parcelle|CSTJF|agro"
        r"|semis|r[eé]colte|engrais|irrigation|ph[yf]tosanitaire|adventice"
        r"|assolement|couvert|rotation|mati[eè]re.organique|humus"
        r"|agricult|ferme\b|champ\b|tracteur|bl[eé]\b|ma[iï]s\b|colza|tournesol"
        r"|vigne|viticult|b[eé]tail|[eé]levage|v[eé]t[eé]rinaire|troupeau"
        r"|paysage|for[eê]t|haie\b|biodiversit[eé]|permaculture)"
    ),
    "cyber": re.compile(
        r"(?i)(CVE|vuln[eé]rabilit[eé]|firewall|IOC\b|Suricata|MITRE|exploit"
        r"|malware|ransomware|phishing|SIEM|YARA|sigma\b|logstash"
        r"|incident|forensic|hash\b|indicator|threat|APT\b|C2\b"
        r"|pentest|pentesting|intrusion|injection|XSS\b|CSRF\b|SQLi\b"
        r"|payload|reverse.shell|privilege.escal|nmap\b|metasploit"
        r"|s[eé]curit[eé].informatique|chiffrement|cryptographie|certificat"
        r"|authentification|brute.force|OSINT\b)"
    ),
    "data": re.compile(
        r"(?i)(csv\b|xlsx?\b|corr[eé]lation|graphique|tendance|dataset|pandas"
        r"|dataframe|statistique|r[eé]gression|histogramme|scatter|boxplot"
        r"|moyenne|m[eé]diane|[eé]cart.type|tableau\b|pivot|merge.*data"
        r"|visuali[sz]|matplotlib|seaborn|plotly|numpy|scipy|sklearn"
        r"|clustering|classification|pr[eé]diction|mod[eè]le|entra[iî]nement"
        r"|feature|label|target|accuracy|precision|recall|F1\b|AUC\b"
        r"|donn[eé]es|base.de.donn[eé]es|aggregat)"
    ),
    "research": re.compile(
        r"(?i)(recherche|compare[rz]?\b|[eé]tat.de.l.art|rapport|analyse\b"
        r"|synth[eè]se|litt[eé]rature|source|r[eé]f[eé]rence|citation"
        r"|[eé]tude|publi|article|paper|survey|review\b|benchmark\b"
        r"|actualit[eé]|news\b|cherche\b|trouve.moi|dis.moi"
        r"|histoire\b|contexte|d[eé]finition|signification|origine)"
    ),
    "tutor": re.compile(
        r"(?i)(explique|comment.fonctionne|c.est.quoi|apprends|d[eé]fini"
        r"|diff[eé]rence.entre|pourquoi\b|qu.est.ce|enseigne|cours\b"
        r"|tuto|example|illustre|simplifie|vulgarise"
        r"|comprend|comprendre|notion|concept|principe|th[eé]orie"
        r"|introduction|d[eé]butant|d[eé]marrer|guide\b)"
    ),
    "math": re.compile(
        r"(?i)(calcul[e|ez]?|combien\b|[eé]quation|formule|int[eé]grale|d[eé]riv[eé]"
        r"|matrices?\b|vecteur|probabilit[eé]|somme\b|produit\b"
        r"|solve\b|math[eé]matique|alg[eè]bre|g[eé]om[eé]trie|trigono"
        r"|logarithme|exposant|racine.carr[eé]e|factoriel|suite\b|s[eé]rie\b"
        r"|d[eé]montre|preuve\b|th[eé]or[eè]me|lemme\b|corollaire)"
    ),
    "writing": re.compile(
        r"(?i)(r[eé]dige|r[eé]diger|[eé]cris\b|[eé]crire|tradui[st]|traduire"
        r"|r[eé]sum[eé]\b|lettre\b|email\b|courriel|reformule|am[eé]liore.ce"
        r"|corrige.ce|corrig[ez]\b|orthographe|grammaire|dissertation|essai\b"
        r"|r[eé][eé]cri[st]|paraphrase|mail\b|outline\b)"
    ),
    # vision and doc_qa have no regex — detected by file presence only
}

# Fast-path: greetings and pure chitchat go straight to "general" without LLM
_GENERAL_FASTPATH = re.compile(
    r"(?i)^(bonjour|salut|hello|hi|hey|bonsoir|merci|ok|oui|non|coucou|ouais"
    r"|d.accord|parfait|super|cool|bravo|bien|oui merci|merci beaucoup|thanks"
    r"|thank you|ok merci|c.est bon|c.est tout|rien|nothing|nvm|nevermind"
    r"|pas de probl[eè]me|avec plaisir|de rien)[.!?]*$"
)

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

# Routes that have a corresponding pipeline YAML (plus special passthrough routes).
# Any route not in this set is remapped to "default" before being returned.
PIPELINE_ROUTES = {"code", "kine", "cyber", "research", "default"}
PASSTHROUGH_ROUTES = {"vision", "doc_qa"}  # handled by ODO directly, not via pipeline YAML
VALID_ROUTES = PIPELINE_ROUTES | PASSTHROUGH_ROUTES

# LLM classification prompt (kept minimal for speed)
CLASSIFY_SYSTEM = (
    "You are an intent classifier. Output exactly one word: the intent category. "
    "Categories: " + ", ".join(ROUTE_NAMES) + "."
)

# Port 8086 = nothink proxy — single-token output, does not block main inference slot
LLM_BACKEND = os.environ.get("ODO_LLM_BACKEND", "http://127.0.0.1:8086")
LLM_TIMEOUT = int(os.environ.get("ODO_LLM_TIMEOUT", "2"))


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
    """Send a short classification request to the LLM with GBNF grammar.

    Uses the nothink proxy (port 8086) with max_tokens=5 and a 2s timeout so
    classification never blocks the main inference slot.
    """
    payload = {
        "model": "qwen3.5",
        "messages": [
            {"role": "system", "content": CLASSIFY_SYSTEM},
            {"role": "user", "content": text[:200]},  # truncate hard — route names are short
        ],
        "max_tokens": 5,          # longest route name is 8 chars ≈ 3 tokens
        "temperature": 0.0,       # deterministic classification
        "top_p": 1.0,
        "top_k": 1,               # greedy
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

def _normalize_route(route: str) -> str:
    """Map any route that has no pipeline YAML to 'default'.

    Known pipeline routes: code, kine, cyber, research, default.
    Passthrough routes handled by ODO directly: vision, doc_qa.
    Everything else (tutor, agro, data, general, …) → default.
    """
    return route if route in VALID_ROUTES else "default"


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
    # Strategy 1b — fast-path for greetings / pure chitchat (no LLM needed)
    stripped = text.strip()
    if stripped and _GENERAL_FASTPATH.match(stripped):
        return {"route": "default", "confidence": 0.9, "strategy": "fastpath"}

    # Strategy 1 — keyword/regex
    route, conf = _classify_keyword(text)
    if route and conf >= 0.5:
        return {"route": _normalize_route(route), "confidence": conf, "strategy": "keyword"}

    # Strategy 2 — file type
    file_route, file_conf = _classify_files(files, has_image)
    if file_route and file_conf >= 0.5:
        # If keyword also had a match (but low confidence), boost if same route
        if route == file_route:
            file_conf = min(0.95, file_conf + 0.1)
        return {"route": _normalize_route(file_route), "confidence": file_conf, "strategy": "filetype"}

    # If keyword had a weak match, use it over LLM
    if route and conf > 0.0:
        return {"route": _normalize_route(route), "confidence": conf, "strategy": "keyword_weak"}

    # Strategy 3 — LLM fallback (nothink proxy, max_tokens=5, timeout=2s)
    llm_route, llm_conf = _classify_llm(text)
    return {"route": _normalize_route(llm_route), "confidence": llm_conf, "strategy": "llm"}


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
