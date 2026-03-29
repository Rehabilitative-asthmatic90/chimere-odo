#!/usr/bin/env python3
"""
dspy_optimize.py — Overnight DSPy MIPROv2 prompt optimization per domain.

Optimizes system_prompt in ~/.chimere/odo/pipelines/{domain}.yaml using
DSPy MIPROv2 Bayesian optimization with the local Qwen3.5 model.

Usage:
    dspy_optimize.py kine                 # optimize kine pipeline
    dspy_optimize.py code                 # optimize code pipeline
    dspy_optimize.py --all                # optimize all domains
    dspy_optimize.py kine --auto medium   # more thorough (25 trials)
    dspy_optimize.py kine --dry-run       # show eval dataset, don't optimize
    dspy_optimize.py kine --apply         # write optimized prompt to YAML

Overnight workflow:
    01:00 — dspy_optimize.py --all --auto light
    → writes optimized prompts to /tmp/dspy_optimized_{domain}.yaml
    → morning review before applying
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Use nothink proxy (8086) to avoid wasting think tokens during optimization
LLM_URL = os.environ.get("DSPY_LLM_URL", "http://127.0.0.1:8086/v1")
PIPELINES_DIR = Path.home() / ".chimere/odo/pipelines"
OUTPUT_DIR = Path("/tmp/dspy_optimized")

# ── Eval datasets per domain ────────────────────────────────────────────────

DATASETS = {
    "kine": {
        "signature_doc": "Answer a physiotherapy clinical question using evidence-based practice and HAS guidelines. Structure with SOAP when appropriate.",
        "train": [
            ("Patient with acute low back pain, no red flags. First-line treatment per HAS?",
             "Active management: education, reassurance, progressive return to activity. Avoid bed rest. NSAIDs if needed. Grade A per HAS 2019."),
            ("How to assess shoulder impingement syndrome?",
             "Neer test, Hawkins-Kennedy, painful arc 60-120°. Assess rotator cuff (Jobe, external rotation). Rule out cervical radiculopathy."),
            ("Post-ACL reconstruction rehab protocol week 1-6?",
             "Phase 1: ROM 0-90°, WB as tolerated, quad sets, SLR. Phase 2 (wk 3-6): closed chain, stationary bike, progress ROM. No open chain quad until wk 12."),
            ("Red flags for cauda equina in back pain?",
             "Saddle anesthesia, bilateral leg weakness, urinary retention/incontinence, fecal incontinence. EMERGENCY: immediate MRI + surgical consult within 24h."),
            ("TENS parameters for chronic knee OA?",
             "Conventional: 80-100 Hz, 50-80 µs, 30-60 min. Electrodes: medial + lateral joint line. Evidence B per HAS. Complement with exercise."),
            ("Exercices tendinopathie achilléenne chronique?",
             "Isométrique (S1-2): 45s x 5. HSR (S2-6): heel raise 3x15 tempo 3-0-3. Excentrique Alfredson (S4-12): 3x15 2x/jour. Pliométrie (S8+). VISA-A >80."),
            ("Bilan kiné lombalgie chronique?",
             "SOAP: Schöber, distance doigts-sol, Sorensen, FABQ, Tampa. Classification spécifique/non-spécifique. Profil biopsychosocial. Exercice actif = grade A HAS 2019."),
        ],
        "dev": [
            ("Grade 2 lateral ankle sprain rehab timeline?",
             "Acute (0-72h): PEACE. Subacute (3-14d): ROM, proprioception. Functional (2-6 wk): strength, sport-specific. Return: pain-free ROM, ≥90% strength symmetry."),
            ("Contraindications for cervical manipulation?",
             "Absolute: vertebral artery insufficiency, upper cervical instability, fracture, inflammatory arthritis, malignancy. Relative: anticoagulation, osteoporosis. VBI screening."),
            ("McKenzie classification lumbar disc?",
             "Derangement: directional preference, centralization. Dysfunction: end-range pain, no centralization. Postural: sustained posture pain. Treatment follows classification."),
            ("Exercise prescription stage 2 COPD pulmonary rehab?",
             "Endurance: walking/cycling 60-80% peak, 20-40 min, 3-5x/wk. Strength: 60-70% 1RM, 2-3x8-12. Breathing exercises. 6MWT baseline."),
            ("Protocole rééducation post-arthroscopie genou ménisque?",
             "S1-2: glaçage, contractions iso quadriceps, mobilisation passive. S3-6: vélo, proprioception, renforcement chaîne fermée. S6-12: course progressive, tests fonctionnels."),
        ],
    },
    "code": {
        "signature_doc": "Provide clean, efficient, well-tested code solutions. Include type hints. Prefer simple solutions over clever ones.",
        "train": [
            ("Write a function to check if a number is prime",
             "def is_prime(n: int) -> bool:\n    if n <= 1: return False\n    if n <= 3: return True\n    if n % 2 == 0 or n % 3 == 0: return False\n    i = 5\n    while i * i <= n:\n        if n % i == 0 or n % (i+2) == 0: return False\n        i += 6\n    return True"),
            ("Read a CSV and compute column statistics in Python",
             "import csv\nfrom statistics import mean, median, stdev\n\ndef csv_stats(path: str, col: str) -> dict:\n    vals = []\n    with open(path) as f:\n        for row in csv.DictReader(f):\n            try: vals.append(float(row[col]))\n            except: continue\n    return {'count': len(vals), 'mean': mean(vals), 'median': median(vals), 'stdev': stdev(vals)} if vals else {}"),
            ("Create a Python decorator that retries a function N times on exception",
             "import functools, time\n\ndef retry(n: int = 3, delay: float = 1.0):\n    def decorator(fn):\n        @functools.wraps(fn)\n        def wrapper(*a, **kw):\n            for attempt in range(n):\n                try: return fn(*a, **kw)\n                except Exception:\n                    if attempt == n - 1: raise\n                    time.sleep(delay)\n        return wrapper\n    return decorator"),
            ("Write a bash script to monitor disk usage and alert if above threshold",
             "#!/bin/bash\nTHRESHOLD=${1:-90}\ndf -h --output=pcent,target | tail -n+2 | while read pct mount; do\n    usage=${pct%\\%}\n    [ \"$usage\" -ge \"$THRESHOLD\" ] && echo \"ALERT: $mount at $pct\"\ndone"),
        ],
        "dev": [
            ("Implement a simple LRU cache in Python without functools",
             "class LRUCache:\n    def __init__(self, cap: int):\n        self.cap = cap; self.cache = {}\n        from collections import OrderedDict; self.cache = OrderedDict()\n    def get(self, k): ...\n    def put(self, k, v): ..."),
            ("Write a Python async HTTP client that fetches N URLs concurrently",
             "import asyncio, aiohttp\nasync def fetch_all(urls: list[str]) -> list[str]:\n    async with aiohttp.ClientSession() as s:\n        tasks = [s.get(u) for u in urls]\n        responses = await asyncio.gather(*tasks)\n        return [await r.text() for r in responses]"),
            ("Create a git pre-commit hook that checks for TODO comments",
             "#!/bin/bash\nif git diff --cached | grep -qE '^\\+.*TODO'; then\n    echo 'ERROR: TODO found in staged changes'\n    git diff --cached | grep -n '^+.*TODO'\n    exit 1\nfi"),
        ],
    },
    "cyber": {
        "signature_doc": "Analyze cybersecurity threats using MITRE ATT&CK framework. Classify IOCs, assess severity, propose countermeasures.",
        "train": [
            ("Analyze IP 185.220.101.42",
             "Tor exit node (DE). AbuseIPDB 98% confidence. MITRE T1090.003 (Multi-hop Proxy). SSH bruteforce + web scanning. Block inbound, monitor logs, correlate 185.220.101.0/24."),
            ("CVE-2024-3094 XZ Utils backdoor — impact assessment",
             "CRITICAL. Supply chain attack (T1195.002). Affected: xz 5.6.0-5.6.1, sshd via systemd. Impact: RCE as root. Remediation: downgrade to 5.4.x, verify checksums, audit dependencies."),
            ("Suspicious PowerShell encoded command detected in logs",
             "MITRE T1059.001 (PowerShell) + T1027 (Obfuscation). Decode base64 payload. Check for: download cradle, AMSI bypass, lateral movement (T1021). Severity: HIGH. Block execution, isolate host."),
        ],
        "dev": [
            ("SHA256 hash: a1b2c3d4... found in endpoint detection",
             "Query VT, MalwareBazaar, OTX. Map to MITRE kill chain. Assess: first seen, detection ratio, family. If malicious: quarantine, check lateral movement, update YARA rules."),
            ("Suricata alert: ET TROJAN Win32/AgentTesla Exfiltration",
             "AgentTesla = keylogger/stealer (T1056.001). Exfil via SMTP/FTP/HTTP (T1041). Severity: HIGH. Contain: isolate host, block C2 domains, credential reset for affected user, scan other endpoints."),
        ],
    },
}


EXTERNAL_DATASETS_DIR = Path.home() / ".chimere/data/dspy_datasets"


def _load_external_pairs(domain: str, max_pairs: int = 50) -> list[tuple[str, str]]:
    """Load (question, answer) pairs from external JSONL files.

    Searches for: {domain}_opus_gold.jsonl (priority), {domain}_chimere.jsonl (fallback).
    """
    pairs = []
    # Priority: Opus gold > chimere distill
    for suffix in ["_opus_gold.jsonl", "_chimere.jsonl"]:
        ext_file = EXTERNAL_DATASETS_DIR / f"{domain}{suffix}"
        if not ext_file.exists():
            continue
        with open(ext_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    q = d.get("question", "")
                    a = d.get("answer", "")
                    if q and a and len(a) > 50:
                        pairs.append((q[:500], a[:2000]))
                except json.JSONDecodeError:
                    continue
    # Diverse sampling: take evenly spaced samples if too many
    if len(pairs) > max_pairs:
        import random
        random.seed(42)
        pairs = random.sample(pairs, max_pairs)
    return pairs


def run_optimization(domain: str, auto: str = "light", dry_run: bool = False):
    """Run MIPROv2 optimization for a domain."""
    import dspy
    from dspy.teleprompt import MIPROv2

    if domain not in DATASETS:
        print(f"ERROR: Unknown domain '{domain}'. Available: {list(DATASETS.keys())}")
        return None

    ds = DATASETS[domain]
    print(f"\n{'='*60}")
    print(f"DSPy MIPROv2 — {domain} (auto={auto})")
    print(f"{'='*60}")

    # Configure LM
    lm = dspy.LM(
        "openai/qwen3.5-35b",
        api_base=LLM_URL,
        api_key="none",
        model_type="chat",
        cache=False,
        max_tokens=2048,
    )
    dspy.configure(lm=lm)

    # Build datasets: inline + external chimere pairs
    external = _load_external_pairs(domain, max_pairs=50)
    all_train = list(ds["train"]) + external

    trainset = [
        dspy.Example(question=q, answer=a).with_inputs("question")
        for q, a in all_train
    ]
    devset = [
        dspy.Example(question=q, answer=a).with_inputs("question")
        for q, a in ds["dev"]
    ]

    print(f"  Train: {len(trainset)} examples (inline: {len(ds['train'])}, external: {len(external)})")
    print(f"  Dev:   {len(devset)} examples")

    if dry_run:
        print("\n[DRY RUN] Dataset preview:")
        for i, ex in enumerate(trainset[:3]):
            print(f"  [{i+1}] Q: {ex.question[:80]}...")
            print(f"       A: {ex.answer[:80]}...")
        return None

    # Define signature dynamically
    QASignature = dspy.Signature(
        "question -> answer",
        instructions=ds["signature_doc"],
    )

    # Define program
    class DomainQA(dspy.Module):
        def __init__(self):
            self.qa = dspy.ChainOfThought(QASignature)
        def forward(self, question: str):
            return self.qa(question=question)

    # Metric: keyword overlap + length check (avoids LLM-as-judge self-bias)
    def quality_metric(example, pred, trace=None):
        ref_words = set(example.answer.lower().split())
        pred_words = set(pred.answer.lower().split()) if hasattr(pred, 'answer') else set()
        if not pred_words:
            return 0.0
        overlap = len(ref_words & pred_words) / max(len(ref_words), 1)
        length_ok = 0.3 < len(pred.answer) / max(len(example.answer), 1) < 3.0
        return overlap * (1.0 if length_ok else 0.5)

    # Optimize
    program = DomainQA()
    t0 = time.time()

    optimizer = MIPROv2(
        metric=quality_metric,
        auto=auto,
        num_threads=1,
        log_dir=str(OUTPUT_DIR / f"{domain}_logs"),
    )

    try:
        optimized = optimizer.compile(
            program,
            trainset=trainset,
        )
    except Exception as e:
        print(f"ERROR: Optimization failed: {e}")
        return None

    elapsed = time.time() - t0
    print(f"\n  Optimization completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Extract optimized instruction
    optimized_instruction = None
    for predictor in optimized.predictors():
        sig = predictor.signature
        optimized_instruction = sig.instructions
        print(f"\n  Optimized instruction ({len(optimized_instruction)} chars):")
        print(f"  {optimized_instruction[:300]}...")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "domain": domain,
        "auto": auto,
        "elapsed_s": round(elapsed),
        "train_size": len(trainset),
        "dev_size": len(devset),
        "optimized_instruction": optimized_instruction,
    }
    out_path = OUTPUT_DIR / f"{domain}_result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Result saved to {out_path}")

    # Save optimized program
    try:
        optimized.save(str(OUTPUT_DIR / f"{domain}_program.json"))
    except Exception:
        pass

    return optimized_instruction


def apply_to_yaml(domain: str, instruction: str):
    """Write optimized instruction to pipeline YAML."""
    yaml_path = PIPELINES_DIR / f"{domain}.yaml"
    if not yaml_path.exists():
        print(f"ERROR: {yaml_path} not found")
        return

    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML not installed")
        return

    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    old = config.get("system_prompt", "")
    config["system_prompt"] = instruction

    # Backup
    backup = yaml_path.with_suffix(".yaml.bak")
    with open(backup, "w") as f:
        import shutil
        shutil.copy2(yaml_path, backup)

    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"  Applied to {yaml_path}")
    print(f"  Backup at {backup}")
    print(f"  Old prompt: {len(old)} chars → New: {len(instruction)} chars")


def main():
    parser = argparse.ArgumentParser(description="DSPy MIPROv2 prompt optimization")
    parser.add_argument("domain", nargs="?", help="Domain to optimize (kine, code, cyber)")
    parser.add_argument("--all", action="store_true", help="Optimize all domains")
    parser.add_argument("--auto", default="light", choices=["light", "medium", "heavy"],
                        help="Optimization intensity (default: light)")
    parser.add_argument("--dry-run", action="store_true", help="Show dataset, don't optimize")
    parser.add_argument("--apply", action="store_true", help="Apply optimized prompt to YAML")
    args = parser.parse_args()

    domains = list(DATASETS.keys()) if args.all else [args.domain] if args.domain else []
    if not domains:
        print("Usage: dspy_optimize.py <domain> or --all")
        print(f"Available: {list(DATASETS.keys())}")
        return

    for domain in domains:
        instruction = run_optimization(domain, args.auto, args.dry_run)
        if instruction and args.apply:
            apply_to_yaml(domain, instruction)


if __name__ == "__main__":
    main()
