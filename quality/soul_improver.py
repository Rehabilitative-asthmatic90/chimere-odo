#!/usr/bin/env python3
"""
Soul Improver — Autonomous SOUL.md Self-Improvement System
============================================================
6-phase cycle: COLLECT → JUDGE → DETECT → GENERATE → APPLY+TEST → VALIDATE/ROLLBACK

Runs every 30 minutes via systemd timer. Detects recurring failure patterns in
GLM agent interactions, generates SOUL.md patches using Claude Opus, applies them
with git versioning, and validates via targeted re-testing. Reverts on degradation.

Usage:
    soul_improver.py run [--dry-run] [--agent AGENT] [--force]
    soul_improver.py detect [--agent AGENT]
    soul_improver.py status
    soul_improver.py history [--last N]
    soul_improver.py rules [--agent AGENT]
    soul_improver.py git-init
    soul_improver.py revert --run-id N
"""
import argparse
import hashlib
import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Import shared library
sys.path.insert(0, str(Path(__file__).parent))
from judge_lib import (
    CHIMERE_HOME, AGENTS_DIR, JUDGE_DIR, DB_PATH, PATCHES_DIR,
    AGENTS_TO_JUDGE, CLAUDE_TIMEOUT,
    _load_env, init_db, send_telegram_alert, judge_interaction, store_judgment,
)

# ── Config ──
CONFIG_PATH = Path(__file__).parent / "soul_improver_config.json"
LOCK_FILE = CHIMERE_HOME / ".soul_improver_lock"
AGENTS_GIT_DIR = AGENTS_DIR  # git repo root
GLOBAL_TIMEOUT = 2700  # 45 min hard kill


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


# ── Lock ──

class LockError(Exception):
    pass


def acquire_lock():
    """Acquire lock file. Raises LockError if already locked."""
    if LOCK_FILE.exists():
        # Check if PID is still alive
        try:
            pid = int(LOCK_FILE.read_text().strip())
            os.kill(pid, 0)  # Check if process exists
            raise LockError(f"Another instance running (PID {pid})")
        except (ValueError, ProcessLookupError, PermissionError):
            # Stale lock
            LOCK_FILE.unlink(missing_ok=True)
    LOCK_FILE.write_text(str(os.getpid()))


def release_lock():
    LOCK_FILE.unlink(missing_ok=True)


# ── Git helpers ──

def git_run(*args, check=True) -> subprocess.CompletedProcess:
    """Run git command in agents directory."""
    return subprocess.run(
        ["git"] + list(args),
        cwd=str(AGENTS_GIT_DIR),
        capture_output=True, text=True,
        check=check, timeout=30,
    )


def git_current_commit() -> str:
    """Get current HEAD commit hash."""
    r = git_run("rev-parse", "HEAD", check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def git_commit_soul(message: str) -> str:
    """Stage all SOUL.md files and commit. Returns commit hash."""
    for agent in AGENTS_TO_JUDGE:
        soul_path = AGENTS_GIT_DIR / agent / "SOUL.md"
        if soul_path.exists():
            git_run("add", str(soul_path))
    r = git_run("commit", "-m", message, check=False)
    if r.returncode == 0:
        return git_current_commit()
    return ""


def git_revert_to(commit_hash: str) -> bool:
    """Revert to a specific commit (hard reset)."""
    r = git_run("reset", "--hard", commit_hash, check=False)
    return r.returncode == 0


def git_diff_from(commit_hash: str) -> str:
    """Get diff from a commit to current HEAD."""
    r = git_run("diff", commit_hash, "HEAD", "--", "*/SOUL.md", check=False)
    return r.stdout if r.returncode == 0 else ""


# ── Phase 1: COLLECT ──

def phase_collect(conn: sqlite3.Connection, config: dict) -> dict:
    """Collect recent judgments and run rotated test subset."""
    print("\n[Phase 1: COLLECT]")
    window_days = config["patterns"]["rolling_window_days"]
    cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()

    # Get recent judgments
    rows = conn.execute("""
        SELECT id, judged_at, agent, user_message, overall, verdict,
               factual_accuracy, routing, tool_calling, output_format,
               user_utility, anti_hallucination, weaknesses, suggestions
        FROM judgments
        WHERE judged_at >= ? AND overall IS NOT NULL
        ORDER BY judged_at DESC
    """, (cutoff,)).fetchall()

    print(f"  {len(rows)} judgments in {window_days}-day window")

    # Run rotated test battery
    test_cfg = config["test_battery"]
    tests_per_cycle = test_cfg["tests_per_cycle"]

    print(f"  Running test battery (rotate, {tests_per_cycle} tests)...")
    test_results = _run_test_battery_rotate()
    print(f"  Tests completed: {len(test_results)} results")

    return {
        "judgments": rows,
        "test_results": test_results,
        "window_days": window_days,
    }


def _run_test_battery_rotate() -> list[dict]:
    """Run the next group of tests in round-robin rotation."""
    test_battery_path = JUDGE_DIR / "test_battery.py"
    python = str(CHIMERE_HOME / "venvs" / "pipeline" / "bin" / "python3")

    try:
        result = subprocess.run(
            [python, str(test_battery_path), "--rotate"],
            capture_output=True, text=True, timeout=600,
        )
        # Parse results from JSONL output file
        results_path = JUDGE_DIR / "test_battery_results.jsonl"
        if results_path.exists():
            results = []
            with open(results_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            results.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            return results
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"  Test battery error: {e}")

    return []


# ── Phase 2: JUDGE ──

def phase_judge(conn: sqlite3.Connection, test_results: list[dict]) -> list[dict]:
    """Judge test battery results via Claude Opus."""
    print("\n[Phase 2: JUDGE]")

    new_judgments = []
    for i, result in enumerate(test_results):
        if result.get("error"):
            print(f"  [{i+1}] {result['test_id']}: SKIP (error: {result['error']})")
            continue

        # Convert test result to interaction format for judge
        interaction = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_message": result.get("message", ""),
            "assistant_text": result.get("response", ""),
            "tool_calls": [],
            "tool_results": [],
            "model": result.get("model"),
            "provider": "llama.cpp",
            "usage": result.get("usage"),
        }
        agent = result.get("agent", "main")

        print(f"  [{i+1}] Judging {result['test_id']} [{agent}]...", end=" ", flush=True)
        t0 = time.time()
        judgment = judge_interaction(interaction, agent)
        elapsed = time.time() - t0

        if judgment.get("error"):
            print(f"ERROR ({elapsed:.0f}s)")
        else:
            verdict = judgment.get("verdict", "?")
            overall = judgment.get("overall", "?")
            print(f"{verdict} ({overall}/5) ({elapsed:.0f}s)")

            # Store in DB
            store_judgment(conn, agent, f"test_battery:{result['test_id']}",
                          interaction, judgment)
            judgment["test_id"] = result["test_id"]
            judgment["agent"] = agent
            new_judgments.append(judgment)

    print(f"  Judged: {len(new_judgments)} test results")
    return new_judgments


# ── Phase 3: DETECT ──

def phase_detect(conn: sqlite3.Connection, config: dict,
                 target_agent: str | None = None) -> list[dict]:
    """Detect recurring failure patterns from judgments. No LLM calls."""
    print("\n[Phase 3: DETECT]")

    window_days = config["patterns"]["rolling_window_days"]
    min_freq = config["patterns"]["min_frequency"]
    cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()

    agent_filter = ""
    params = [cutoff]
    if target_agent:
        agent_filter = "AND agent = ?"
        params.append(target_agent)

    # Get FAIL/WARN judgments
    rows = conn.execute(f"""
        SELECT agent, user_message, overall, verdict,
               factual_accuracy, routing, tool_calling, output_format,
               user_utility, anti_hallucination, weaknesses, suggestions
        FROM judgments
        WHERE judged_at >= ? AND verdict IN ('FAIL', 'WARN') {agent_filter}
        ORDER BY judged_at DESC
    """, params).fetchall()

    if not rows:
        print("  No failures/warnings in window")
        return []

    print(f"  Analyzing {len(rows)} FAIL/WARN judgments...")

    # Cluster by weakness keywords
    weakness_counter = Counter()
    criterion_scores = {}
    agent_issues = {}
    example_queries = {}

    criteria = ["factual_accuracy", "routing", "tool_calling",
                "output_format", "user_utility", "anti_hallucination"]

    for row in rows:
        (agent, user_msg, overall, verdict,
         fa, rt, tc, of_, uu, ah, weaknesses_json, suggestions_json) = row

        scores = {"factual_accuracy": fa, "routing": rt, "tool_calling": tc,
                  "output_format": of_, "user_utility": uu, "anti_hallucination": ah}

        # Track per-criterion scores
        for crit, val in scores.items():
            if val is not None:
                criterion_scores.setdefault(agent, {}).setdefault(crit, []).append(val)

        # Track weakness keywords
        weaknesses = json.loads(weaknesses_json) if weaknesses_json else []
        for w in weaknesses:
            # Normalize: lowercase, strip punctuation, take first 5 words
            normalized = re.sub(r'[^\w\s]', '', w.lower()).strip()
            key_words = " ".join(normalized.split()[:5])
            if key_words:
                weakness_counter[(agent, key_words)] += 1
                example_queries.setdefault((agent, key_words), []).append(
                    user_msg[:60] if user_msg else "")

        agent_issues.setdefault(agent, []).append({
            "verdict": verdict, "overall": overall,
            "scores": scores, "weaknesses": weaknesses,
        })

    # Build patterns from recurring weaknesses
    patterns = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for (agent, keywords), freq in weakness_counter.most_common(20):
        if freq < min_freq:
            continue

        # Find the weakest criterion for this agent
        agent_scores = criterion_scores.get(agent, {})
        weakest_crit = None
        weakest_avg = 5.0
        for crit, vals in agent_scores.items():
            avg = sum(vals) / len(vals)
            if avg < weakest_avg:
                weakest_avg = avg
                weakest_crit = crit

        examples = example_queries.get((agent, keywords), [])[:3]

        pattern = {
            "agent": agent,
            "criterion": weakest_crit,
            "pattern_type": "recurring_weakness",
            "description": keywords,
            "frequency": freq,
            "avg_score": weakest_avg,
            "example_queries": examples,
        }
        patterns.append(pattern)

        # Store in DB
        conn.execute("""
            INSERT INTO detected_patterns
                (detected_at, agent, criterion, pattern_type, description,
                 frequency, avg_score, example_queries)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (now_iso, agent, weakest_crit, "recurring_weakness",
              keywords, freq, weakest_avg,
              json.dumps(examples, ensure_ascii=False)))

    conn.commit()

    # Also check for score drops per criterion
    for agent, crit_scores in criterion_scores.items():
        for crit, vals in crit_scores.items():
            avg = sum(vals) / len(vals)
            if avg < 2.5 and len(vals) >= min_freq:
                pattern = {
                    "agent": agent,
                    "criterion": crit,
                    "pattern_type": "low_criterion",
                    "description": f"{crit} consistently low ({avg:.1f}/5)",
                    "frequency": len(vals),
                    "avg_score": avg,
                    "example_queries": [],
                }
                # Avoid duplicates
                if not any(p["criterion"] == crit and p["agent"] == agent
                          and p["pattern_type"] == "low_criterion" for p in patterns):
                    patterns.append(pattern)

    for p in patterns:
        print(f"  [{p['agent']}] {p['pattern_type']}: {p['description']} "
              f"(freq={p['frequency']}, avg={p['avg_score']:.1f})")

    if not patterns:
        print("  No recurring patterns detected")

    return patterns


# ── Phase 4: GENERATE ──

PATCH_SYSTEM_PROMPT = """\
Tu es un expert en prompt engineering pour LLMs locaux (Qwen3.5-35B-A3B, MoE).
Tu optimises des fichiers SOUL.md qui guident le comportement d'agents IA.

CONTRAINTES STRICTES :
- Maximum 2 ajouts de règle et 1 suppression par patch
- Les règles doivent être courtes (1-2 lignes max)
- Utilise le français simple, impératif, concret
- Les exemples few-shot fonctionnent bien (ex: "17 moutons - 9 = 8")
- Les budgets d'outils fonctionnent (ex: "Maximum 2 tool calls")
- Ne touche JAMAIS aux règles protégées (listées dans le contexte)

FORMAT DE RÉPONSE — JSON strict :
{
  "reasoning": "Explication courte de pourquoi ce patch",
  "changes": [
    {"action": "add", "after_line_containing": "## RÈGLES", "content": "nouvelle règle ici"},
    {"action": "remove", "line_containing": "texte exact de la ligne à supprimer"},
    {"action": "replace", "old": "texte ancien", "new": "texte nouveau"}
  ],
  "expected_improvement": "Ce que ce patch devrait améliorer",
  "target_criterion": "nom du critère visé"
}"""


def phase_generate(conn: sqlite3.Connection, config: dict,
                   patterns: list[dict], target_agent: str | None = None,
                   dry_run: bool = False) -> dict | None:
    """Generate a SOUL.md patch proposal using Claude Opus."""
    print("\n[Phase 4: GENERATE]")

    # ── Gate checks ──
    patch_cfg = config["patch"]
    sched_cfg = config["schedule"]

    if not patterns:
        print("  SKIP: No patterns to address")
        return None

    # Pick target agent (most frequent issues)
    if target_agent:
        agent = target_agent
    else:
        agent_freqs = Counter()
        for p in patterns:
            agent_freqs[p["agent"]] += p["frequency"]
        agent = agent_freqs.most_common(1)[0][0]

    # Check cooldown
    last_patch = conn.execute("""
        SELECT finished_at FROM improvement_runs
        WHERE patch_generated = 1 AND target_agent = ?
        ORDER BY id DESC LIMIT 1
    """, (agent,)).fetchone()

    if last_patch and last_patch[0]:
        last_time = datetime.fromisoformat(last_patch[0])
        cooldown = timedelta(minutes=sched_cfg["cooldown_min"])
        if datetime.now(timezone.utc) - last_time < cooldown:
            remaining = cooldown - (datetime.now(timezone.utc) - last_time)
            print(f"  SKIP: Cooldown active ({remaining.seconds // 60}min remaining)")
            return None

    # Check daily limit
    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0).isoformat()
    patches_today = conn.execute("""
        SELECT COUNT(*) FROM improvement_runs
        WHERE patch_generated = 1 AND started_at >= ?
    """, (today_start,)).fetchone()[0]

    if patches_today >= sched_cfg["max_patches_per_day"]:
        print(f"  SKIP: Daily limit reached ({patches_today}/{sched_cfg['max_patches_per_day']})")
        return None

    # Check minimum judgments
    min_judgments = config["patterns"]["min_judgments_for_patch"]
    total_recent = conn.execute("""
        SELECT COUNT(*) FROM judgments
        WHERE judged_at >= ? AND agent = ?
    """, ((datetime.now(timezone.utc) - timedelta(days=7)).isoformat(), agent)).fetchone()[0]

    if total_recent < min_judgments:
        print(f"  SKIP: Not enough judgments ({total_recent}/{min_judgments})")
        return None

    # Check "do not retry" blocklist
    max_reverts = patch_cfg.get("do_not_retry_after_reverts", 2)
    agent_patterns = [p for p in patterns if p["agent"] == agent]

    for p in agent_patterns:
        revert_count = conn.execute("""
            SELECT COUNT(*) FROM improvement_runs
            WHERE target_agent = ? AND outcome = 'reverted'
            AND pattern_summary LIKE ?
        """, (agent, f"%{p['description'][:30]}%")).fetchone()[0]
        if revert_count >= max_reverts:
            print(f"  SKIP pattern '{p['description']}': reverted {revert_count}x (blocklist)")
            agent_patterns = [x for x in agent_patterns if x != p]

    if not agent_patterns:
        print("  SKIP: All patterns blocklisted")
        return None

    # ── Read current SOUL.md ──
    soul_path = AGENTS_DIR / agent / "SOUL.md"
    if not soul_path.exists():
        print(f"  ERROR: {soul_path} not found")
        return None

    current_soul = soul_path.read_text()
    current_lines = len(current_soul.splitlines())
    max_lines = patch_cfg["max_lines_per_agent"].get(agent, 100)

    # ── Build prompt ──
    patterns_text = "\n".join(
        f"- [{p['criterion']}] {p['description']} (freq={p['frequency']}, avg={p['avg_score']:.1f})"
        for p in agent_patterns[:5]
    )
    protected = ", ".join(patch_cfg["protected_rules"])

    prompt = f"""Analyse ces patterns de défaillance de l'agent '{agent}' et propose un patch SOUL.md.

PATTERNS DÉTECTÉS :
{patterns_text}

SOUL.MD ACTUEL ({current_lines} lignes, max {max_lines}) :
```
{current_soul}
```

RÈGLES PROTÉGÉES (NE PAS TOUCHER) : {protected}

CONTRAINTES :
- Max {patch_cfg['max_adds_per_patch']} ajouts, {patch_cfg['max_removes_per_patch']} suppression
- Ne pas dépasser {max_lines} lignes total
- Actuellement {current_lines} lignes

Propose un patch JSON."""

    if len(prompt) > 12000:
        prompt = prompt[:12000] + "\n[...tronqué]"

    if dry_run:
        print(f"  [DRY-RUN] Would generate patch for agent '{agent}'")
        print(f"  Patterns: {len(agent_patterns)}, Current lines: {current_lines}/{max_lines}")
        return {"agent": agent, "patterns": agent_patterns, "dry_run": True}

    # ── Call Claude ──
    print(f"  Generating patch for '{agent}' ({len(agent_patterns)} patterns)...")

    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)

    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--system-prompt", PATCH_SYSTEM_PROMPT,
             "--output-format", "text", "--max-turns", "1",
             "--allowedTools", "", "--no-session-persistence"],
            capture_output=True, text=True, timeout=CLAUDE_TIMEOUT, env=env,
        )
        raw_output = result.stdout.strip()
    except subprocess.TimeoutExpired:
        print("  ERROR: Claude timeout")
        return None
    except FileNotFoundError:
        print("  ERROR: claude CLI not found")
        return None

    # Parse JSON from response
    json_str = raw_output
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0].strip()
    elif "```" in json_str:
        json_str = json_str.split("```")[1].split("```")[0].strip()

    start = json_str.find("{")
    end = json_str.rfind("}")
    if start == -1 or end == -1:
        print("  ERROR: No JSON in Claude response")
        return None

    json_str = json_str[start:end + 1]

    try:
        patch = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"  ERROR: JSON parse failed: {e}")
        return None

    # ── Validate patch ──
    changes = patch.get("changes", [])
    if not changes:
        print("  SKIP: Empty patch")
        return None

    adds = sum(1 for c in changes if c.get("action") == "add")
    removes = sum(1 for c in changes if c.get("action") == "remove")

    if adds > patch_cfg["max_adds_per_patch"]:
        print(f"  REJECT: Too many adds ({adds} > {patch_cfg['max_adds_per_patch']})")
        return None
    if removes > patch_cfg["max_removes_per_patch"]:
        print(f"  REJECT: Too many removes ({removes} > {patch_cfg['max_removes_per_patch']})")
        return None

    # Check protected rules
    for change in changes:
        if change.get("action") in ("remove", "replace"):
            target_text = change.get("line_containing", "") or change.get("old", "")
            for protected_rule in patch_cfg["protected_rules"]:
                if protected_rule.lower() in target_text.lower():
                    print(f"  REJECT: Attempts to modify protected rule: {protected_rule}")
                    return None

    # Estimate new line count
    new_lines_estimate = current_lines + adds - removes
    if new_lines_estimate > max_lines:
        print(f"  REJECT: Would exceed line limit ({new_lines_estimate} > {max_lines})")
        return None

    patch["agent"] = agent
    patch["patterns"] = [p["description"] for p in agent_patterns[:5]]
    print(f"  Patch generated: {adds} adds, {removes} removes")
    print(f"  Reasoning: {patch.get('reasoning', 'N/A')[:100]}")

    return patch


# ── Phase 5: APPLY & TEST ──

def phase_apply_and_test(conn: sqlite3.Connection, config: dict,
                         patch: dict, run_id: int) -> dict:
    """Apply SOUL.md patch, commit, run targeted tests."""
    print("\n[Phase 5: APPLY & TEST]")

    agent = patch["agent"]
    soul_path = AGENTS_DIR / agent / "SOUL.md"
    current_soul = soul_path.read_text()

    # Git commit pre-patch state
    commit_before = git_current_commit()
    pre_commit = git_commit_soul(f"soul_improver: pre-patch snapshot (run #{run_id})")
    if pre_commit:
        commit_before = pre_commit
    print(f"  Pre-patch commit: {commit_before[:8]}")

    # Calculate pre-patch score from recent judgments
    pre_scores = conn.execute("""
        SELECT AVG(overall) FROM judgments
        WHERE agent = ? AND overall IS NOT NULL
        AND judged_at >= ?
        ORDER BY judged_at DESC LIMIT 10
    """, (agent, (datetime.now(timezone.utc) - timedelta(days=7)).isoformat())).fetchone()
    pre_patch_score = pre_scores[0] if pre_scores and pre_scores[0] else 0.0

    # Apply changes
    new_soul = current_soul
    changes = patch.get("changes", [])

    for change in changes:
        action = change.get("action")

        if action == "add":
            anchor = change.get("after_line_containing", "")
            content = change.get("content", "")
            if anchor and anchor in new_soul:
                # Insert after the anchor line
                lines = new_soul.split("\n")
                for i, line in enumerate(lines):
                    if anchor in line:
                        lines.insert(i + 1, content)
                        break
                new_soul = "\n".join(lines)
            else:
                # Append to end
                new_soul = new_soul.rstrip() + "\n" + content + "\n"

        elif action == "remove":
            target = change.get("line_containing", "")
            if target:
                lines = new_soul.split("\n")
                lines = [l for l in lines if target not in l]
                new_soul = "\n".join(lines)

        elif action == "replace":
            old = change.get("old", "")
            new = change.get("new", "")
            if old and old in new_soul:
                new_soul = new_soul.replace(old, new, 1)

    # Validate line count
    max_lines = config["patch"]["max_lines_per_agent"].get(agent, 100)
    new_line_count = len(new_soul.splitlines())
    if new_line_count > max_lines:
        print(f"  ABORT: New SOUL.md exceeds {max_lines} lines ({new_line_count})")
        return {"success": False, "reason": "line_limit_exceeded",
                "commit_before": commit_before}

    # Write new SOUL.md
    soul_path.write_text(new_soul)
    print(f"  Applied {len(changes)} changes to {agent}/SOUL.md ({new_line_count} lines)")

    # Git commit post-patch
    patch_summary = patch.get("reasoning", "auto-patch")[:80]
    commit_after = git_commit_soul(
        f"soul_improver: patch {agent} (run #{run_id}) — {patch_summary}")
    print(f"  Post-patch commit: {commit_after[:8] if commit_after else 'FAILED'}")

    # Run targeted post-patch tests
    print("  Running post-patch tests...")
    test_results = _run_targeted_tests(agent)

    # Judge post-patch results
    post_judgments = []
    for result in test_results:
        if result.get("error"):
            continue
        interaction = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_message": result.get("message", ""),
            "assistant_text": result.get("response", ""),
            "tool_calls": [], "tool_results": [],
            "model": result.get("model"),
            "provider": "llama.cpp",
            "usage": result.get("usage"),
        }
        judgment = judge_interaction(interaction, result.get("agent", agent))
        if not judgment.get("error"):
            post_judgments.append(judgment)
            store_judgment(conn, result.get("agent", agent),
                          f"test_battery:post_patch:{result['test_id']}",
                          interaction, judgment)

    # Calculate post-patch score (only tests matching patched agent)
    agent_scores = [j["overall"] for j in post_judgments
                    if j.get("overall") and j.get("agent") == agent]
    # Fallback to all scores if no agent-specific ones
    post_scores = agent_scores if agent_scores else [
        j["overall"] for j in post_judgments if j.get("overall")]
    post_patch_score = sum(post_scores) / len(post_scores) if post_scores else 0.0

    print(f"  Post-patch score: {post_patch_score:.1f}/5 "
          f"(pre: {pre_patch_score:.1f}, delta: {post_patch_score - pre_patch_score:+.1f})")

    return {
        "success": True,
        "commit_before": commit_before,
        "commit_after": commit_after,
        "pre_patch_score": pre_patch_score,
        "post_patch_score": post_patch_score,
        "delta": post_patch_score - pre_patch_score,
        "tests_run": len(test_results),
        "tests_judged": len(post_judgments),
        "patch_diff": git_diff_from(commit_before) if commit_after else "",
    }


def _run_targeted_tests(agent: str) -> list[dict]:
    """Run tests relevant to the patched agent (2 groups for better signal)."""
    test_battery_path = JUDGE_DIR / "test_battery.py"
    python = str(CHIMERE_HOME / "venvs" / "pipeline" / "bin" / "python3")

    # Run 2 groups for better signal coverage
    group_map = {
        "main": ["A", "B"],
        "cyber": ["C", "A"],
        "datascience": ["C", "A"],
        "projectops": ["A", "B"],
    }
    groups = group_map.get(agent, ["A"])
    all_results = []

    for group in groups:
        try:
            result = subprocess.run(
                [python, str(test_battery_path), "--group", group],
                capture_output=True, text=True, timeout=600,
            )
            results_path = JUDGE_DIR / "test_battery_results.jsonl"
            if results_path.exists():
                with open(results_path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                all_results.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
        except (subprocess.TimeoutExpired, Exception) as e:
            print(f"  Test error (group {group}): {e}")

    return all_results


# ── Phase 6: VALIDATE / ROLLBACK ──

def phase_validate(conn: sqlite3.Connection, config: dict,
                   apply_result: dict, patch: dict, run_id: int) -> str:
    """Validate patch results, keep or revert."""
    print("\n[Phase 6: VALIDATE / ROLLBACK]")

    if not apply_result.get("success"):
        reason = apply_result.get("reason", "unknown")
        print(f"  SKIP: Apply failed ({reason})")
        return "apply_failed"

    delta = apply_result["delta"]
    revert_threshold = config["validation"]["revert_if_delta_below"]
    min_delta = config["validation"]["min_improvement_delta"]

    agent = patch["agent"]
    commit_before = apply_result["commit_before"]
    commit_after = apply_result.get("commit_after", "")
    pre_score = apply_result["pre_patch_score"]
    post_score = apply_result["post_patch_score"]

    # Decision
    if delta < revert_threshold:
        # REVERT
        print(f"  REVERT: delta {delta:+.1f} below threshold {revert_threshold}")
        reverted = git_revert_to(commit_before)
        revert_commit = git_current_commit() if reverted else ""

        outcome = "reverted"
        _notify_telegram(
            f"*Soul Improver — REVERT*\n\n"
            f"Agent: {agent}\n"
            f"Run: #{run_id}\n"
            f"Pre: {pre_score:.1f}/5 → Post: {post_score:.1f}/5 (delta: {delta:+.1f})\n"
            f"Reverted to: {commit_before[:8]}\n"
            f"Reason: {patch.get('reasoning', 'N/A')[:100]}"
        )

        # Update DB
        conn.execute("""
            UPDATE improvement_runs SET
                outcome = 'reverted',
                git_commit_revert = ?,
                telegram_notified = 1
            WHERE id = ?
        """, (revert_commit, run_id))
        conn.commit()

        return "reverted"

    elif delta >= min_delta:
        # KEEP
        print(f"  KEEP: delta {delta:+.1f} meets threshold {min_delta}")
        outcome = "kept"
        _notify_telegram(
            f"*Soul Improver — KEPT*\n\n"
            f"Agent: {agent}\n"
            f"Run: #{run_id}\n"
            f"Pre: {pre_score:.1f}/5 → Post: {post_score:.1f}/5 (delta: {delta:+.1f})\n"
            f"Commit: {commit_after[:8]}\n"
            f"Patch: {patch.get('reasoning', 'N/A')[:100]}"
        )

        conn.execute("""
            UPDATE improvement_runs SET
                outcome = 'kept',
                telegram_notified = 1
            WHERE id = ?
        """, (run_id,))
        conn.commit()

        return "kept"

    else:
        # Neutral — keep but flag
        print(f"  NEUTRAL: delta {delta:+.1f}, keeping cautiously")
        conn.execute("""
            UPDATE improvement_runs SET outcome = 'neutral', telegram_notified = 0
            WHERE id = ?
        """, (run_id,))
        conn.commit()
        return "neutral"


def _notify_telegram(message: str):
    """Send Telegram notification (best-effort)."""
    try:
        send_telegram_alert(message)
    except Exception as e:
        print(f"  Telegram notification failed: {e}")


# ── Archive ──

def archive_patch(run_id: int, patch: dict, apply_result: dict, outcome: str):
    """Save patch details to JSON archive."""
    PATCHES_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = {
        "run_id": run_id,
        "timestamp": ts,
        "agent": patch.get("agent"),
        "reasoning": patch.get("reasoning"),
        "changes": patch.get("changes"),
        "expected_improvement": patch.get("expected_improvement"),
        "target_criterion": patch.get("target_criterion"),
        "pre_patch_score": apply_result.get("pre_patch_score"),
        "post_patch_score": apply_result.get("post_patch_score"),
        "delta": apply_result.get("delta"),
        "outcome": outcome,
        "commit_before": apply_result.get("commit_before"),
        "commit_after": apply_result.get("commit_after"),
        "patch_diff": apply_result.get("patch_diff", "")[:5000],
    }
    path = PATCHES_DIR / f"patch_{ts}_run{run_id}.json"
    with open(path, "w") as f:
        json.dump(archive, f, indent=2, ensure_ascii=False)
    print(f"  Archived: {path}")


# ── Rule Effectiveness ──

def update_rule_effectiveness(conn: sqlite3.Connection, agent: str):
    """Scan SOUL.md rules and update effectiveness tracking."""
    soul_path = AGENTS_DIR / agent / "SOUL.md"
    if not soul_path.exists():
        return

    soul_text = soul_path.read_text()
    now_iso = datetime.now(timezone.utc).isoformat()

    # Extract rules (lines starting with number or bold)
    rules = []
    for line in soul_text.splitlines():
        line = line.strip()
        if (re.match(r'^\d+\.?\s+\*\*', line) or
                re.match(r'^\*\*[A-Z]', line) or
                re.match(r'^##\s+', line)):
            if len(line) > 10:
                rules.append(line[:200])

    for rule_text in rules:
        rule_hash = hashlib.sha256(rule_text.encode()).hexdigest()[:16]

        existing = conn.execute(
            "SELECT id FROM rule_effectiveness WHERE rule_hash = ? AND agent = ?",
            (rule_hash, agent)
        ).fetchone()

        if existing:
            conn.execute("""
                UPDATE rule_effectiveness SET last_seen = ? WHERE id = ?
            """, (now_iso, existing[0]))
        else:
            conn.execute("""
                INSERT INTO rule_effectiveness
                    (agent, rule_text, rule_hash, first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?)
            """, (agent, rule_text, rule_hash, now_iso, now_iso))

    conn.commit()


# ── CLI Commands ──

def cmd_run(args):
    """Full 6-phase improvement cycle."""
    config = load_config()

    # Active hours check
    hour = datetime.now().hour
    active_start, active_end = config["schedule"]["active_hours"]
    if not (active_start <= hour < active_end) and not args.force:
        print(f"Outside active hours ({active_start}-{active_end}), current: {hour}. Use --force to override.")
        return

    # Global timeout
    signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(SystemExit("Global timeout")))
    signal.alarm(GLOBAL_TIMEOUT)

    try:
        acquire_lock()
    except LockError as e:
        print(f"LOCKED: {e}")
        return

    conn = init_db()
    now_iso = datetime.now(timezone.utc).isoformat()
    run_id = None

    try:
        # Create run record
        conn.execute("""
            INSERT INTO improvement_runs (started_at, status) VALUES (?, 'running')
        """, (now_iso,))
        conn.commit()
        run_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        print(f"[Soul Improver] Run #{run_id} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Phase 1: COLLECT
        collect_data = phase_collect(conn, config)
        conn.execute("""
            UPDATE improvement_runs SET
                phase_reached = 'collect',
                interactions_scanned = ?
            WHERE id = ?
        """, (len(collect_data["judgments"]), run_id))
        conn.commit()

        # Phase 2: JUDGE
        new_judgments = phase_judge(conn, collect_data["test_results"])
        conn.execute("""
            UPDATE improvement_runs SET
                phase_reached = 'judge',
                tests_run = ?,
                tests_passed = ?,
                new_judgments = ?
            WHERE id = ?
        """, (
            len(collect_data["test_results"]),
            sum(1 for r in collect_data["test_results"] if not r.get("error")),
            len(new_judgments),
            run_id
        ))
        conn.commit()

        # Phase 3: DETECT
        patterns = phase_detect(conn, config, target_agent=args.agent)
        conn.execute("""
            UPDATE improvement_runs SET
                phase_reached = 'detect',
                patterns_detected = ?,
                pattern_summary = ?
            WHERE id = ?
        """, (
            len(patterns),
            json.dumps([p["description"] for p in patterns[:5]], ensure_ascii=False),
            run_id
        ))
        conn.commit()

        if not patterns:
            print("\n[DONE] No patterns to address. Cycle complete.")
            conn.execute("""
                UPDATE improvement_runs SET
                    finished_at = ?, status = 'completed', outcome = 'no_patterns'
                WHERE id = ?
            """, (datetime.now(timezone.utc).isoformat(), run_id))
            conn.commit()
            return

        # Phase 4: GENERATE
        patch = phase_generate(conn, config, patterns,
                               target_agent=args.agent, dry_run=args.dry_run)

        if not patch:
            print("\n[DONE] No patch generated (gated). Cycle complete.")
            conn.execute("""
                UPDATE improvement_runs SET
                    finished_at = ?, status = 'completed', outcome = 'gated'
                WHERE id = ?
            """, (datetime.now(timezone.utc).isoformat(), run_id))
            conn.commit()
            return

        if args.dry_run:
            print("\n[DRY-RUN] Patch proposal generated but not applied.")
            print(f"  Agent: {patch.get('agent')}")
            conn.execute("""
                UPDATE improvement_runs SET
                    finished_at = ?, status = 'completed', outcome = 'dry_run',
                    target_agent = ?, patch_generated = 1
                WHERE id = ?
            """, (datetime.now(timezone.utc).isoformat(), patch.get("agent"), run_id))
            conn.commit()
            return

        conn.execute("""
            UPDATE improvement_runs SET
                phase_reached = 'generate',
                patch_generated = 1,
                target_agent = ?
            WHERE id = ?
        """, (patch["agent"], run_id))
        conn.commit()

        # Update rule effectiveness before patching
        update_rule_effectiveness(conn, patch["agent"])

        # Phase 5: APPLY & TEST
        apply_result = phase_apply_and_test(conn, config, patch, run_id)

        conn.execute("""
            UPDATE improvement_runs SET
                phase_reached = 'apply_test',
                git_commit_before = ?,
                git_commit_after = ?,
                pre_patch_score = ?,
                post_patch_score = ?,
                patch_diff = ?
            WHERE id = ?
        """, (
            apply_result.get("commit_before"),
            apply_result.get("commit_after"),
            apply_result.get("pre_patch_score"),
            apply_result.get("post_patch_score"),
            apply_result.get("patch_diff", "")[:5000],
            run_id
        ))
        conn.commit()

        # Phase 6: VALIDATE
        outcome = phase_validate(conn, config, apply_result, patch, run_id)

        # Archive
        archive_patch(run_id, patch, apply_result, outcome)

        # Finalize
        conn.execute("""
            UPDATE improvement_runs SET
                finished_at = ?,
                status = 'completed',
                phase_reached = 'validate'
            WHERE id = ?
        """, (datetime.now(timezone.utc).isoformat(), run_id))
        conn.commit()

        print(f"\n[DONE] Run #{run_id} completed. Outcome: {outcome}")

    except SystemExit as e:
        print(f"\n[ABORT] {e}")
        if run_id:
            conn.execute("""
                UPDATE improvement_runs SET
                    finished_at = ?, status = 'aborted', abort_reason = ?
                WHERE id = ?
            """, (datetime.now(timezone.utc).isoformat(), str(e), run_id))
            conn.commit()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        if run_id:
            conn.execute("""
                UPDATE improvement_runs SET
                    finished_at = ?, status = 'error', abort_reason = ?
                WHERE id = ?
            """, (datetime.now(timezone.utc).isoformat(), str(e)[:500], run_id))
            conn.commit()
    finally:
        signal.alarm(0)
        release_lock()
        conn.close()


def cmd_detect(args):
    """Pattern detection only (phase 3)."""
    config = load_config()
    conn = init_db()
    patterns = phase_detect(conn, config, target_agent=args.agent)
    conn.close()

    if patterns:
        print(f"\nTotal: {len(patterns)} patterns detected")
    else:
        print("\nNo patterns found.")


def cmd_status(args):
    """Show improvement system status."""
    conn = init_db()

    # Last run
    last = conn.execute("""
        SELECT id, started_at, status, phase_reached, outcome,
               tests_run, patterns_detected, patch_generated,
               pre_patch_score, post_patch_score, target_agent
        FROM improvement_runs ORDER BY id DESC LIMIT 1
    """).fetchone()

    total_runs = conn.execute("SELECT COUNT(*) FROM improvement_runs").fetchone()[0]
    total_patches = conn.execute(
        "SELECT COUNT(*) FROM improvement_runs WHERE patch_generated = 1").fetchone()[0]
    total_kept = conn.execute(
        "SELECT COUNT(*) FROM improvement_runs WHERE outcome = 'kept'").fetchone()[0]
    total_reverted = conn.execute(
        "SELECT COUNT(*) FROM improvement_runs WHERE outcome = 'reverted'").fetchone()[0]
    total_patterns = conn.execute("SELECT COUNT(*) FROM detected_patterns").fetchone()[0]

    print(f"Soul Improver Status")
    print(f"{'='*40}")
    print(f"Total runs:     {total_runs}")
    print(f"Patches:        {total_patches} generated, {total_kept} kept, {total_reverted} reverted")
    print(f"Patterns:       {total_patterns} detected")

    if last:
        print(f"\nLast run: #{last[0]}")
        print(f"  Started:   {last[1][:19]}")
        print(f"  Status:    {last[2]}")
        print(f"  Phase:     {last[3]}")
        print(f"  Outcome:   {last[4]}")
        if last[7]:
            print(f"  Agent:     {last[10]}")
            pre = last[8] or 0
            post = last[9] or 0
            print(f"  Score:     {pre:.1f} → {post:.1f} (delta: {post-pre:+.1f})")

    # Today's activity
    today = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0).isoformat()
    today_patches = conn.execute("""
        SELECT COUNT(*) FROM improvement_runs
        WHERE patch_generated = 1 AND started_at >= ?
    """, (today,)).fetchone()[0]
    config = load_config()
    max_daily = config["schedule"]["max_patches_per_day"]
    print(f"\nToday: {today_patches}/{max_daily} patches used")

    conn.close()


def cmd_history(args):
    """Show patch history."""
    conn = init_db()
    limit = args.last if hasattr(args, 'last') and args.last else 10

    rows = conn.execute("""
        SELECT id, started_at, status, target_agent, outcome,
               pre_patch_score, post_patch_score, pattern_summary
        FROM improvement_runs
        WHERE patch_generated = 1
        ORDER BY id DESC LIMIT ?
    """, (limit,)).fetchall()

    if not rows:
        print("No patches in history.")
        conn.close()
        return

    print(f"Patch History (last {limit}):")
    print(f"{'ID':>4} | {'Date':>16} | {'Agent':>12} | {'Outcome':>10} | {'Pre':>5} | {'Post':>5} | Patterns")
    print("-" * 90)

    for row in rows:
        rid, started, status, agent, outcome, pre, post, patterns = row
        date = started[:16].replace("T", " ") if started else "?"
        pre = f"{pre:.1f}" if pre else "?"
        post = f"{post:.1f}" if post else "?"
        patterns_short = (patterns[:40] + "...") if patterns and len(patterns) > 40 else (patterns or "")
        print(f"{rid:>4} | {date:>16} | {agent or '?':>12} | {outcome or '?':>10} | {pre:>5} | {post:>5} | {patterns_short}")

    conn.close()


def cmd_rules(args):
    """Show rule effectiveness analysis."""
    conn = init_db()
    agent = args.agent or "main"

    rows = conn.execute("""
        SELECT rule_text, first_seen, last_seen,
               total_relevant_interactions, pass_count, fail_count,
               avg_score_when_relevant, effectiveness, confidence,
               added_by, removed_at
        FROM rule_effectiveness
        WHERE agent = ?
        ORDER BY effectiveness ASC NULLS LAST
    """, (agent,)).fetchall()

    if not rows:
        # Try to populate first
        update_rule_effectiveness(conn, agent)
        rows = conn.execute("""
            SELECT rule_text, first_seen, last_seen,
                   total_relevant_interactions, pass_count, fail_count,
                   avg_score_when_relevant, effectiveness, confidence,
                   added_by, removed_at
            FROM rule_effectiveness WHERE agent = ?
            ORDER BY effectiveness ASC NULLS LAST
        """, (agent,)).fetchall()

    if not rows:
        print(f"No rules tracked for agent '{agent}'")
        conn.close()
        return

    print(f"Rule Effectiveness — {agent} ({len(rows)} rules)")
    print(f"{'='*60}")
    for row in rows:
        (text, first, last, total, passes, fails,
         avg_score, eff, conf, added_by, removed) = row
        text_short = text[:60] + "..." if len(text) > 60 else text
        status = " [REMOVED]" if removed else ""
        eff_str = f"{eff:.0%}" if eff is not None else "N/A"
        print(f"  {text_short}{status}")
        print(f"    Effectiveness: {eff_str} | Total: {total or 0} | P/F: {passes or 0}/{fails or 0}")

    conn.close()


def cmd_git_init(args):
    """Initialize git repo in agents directory."""
    if (AGENTS_GIT_DIR / ".git").exists():
        print("Git repo already exists.")
        r = git_run("log", "--oneline", "-5", check=False)
        if r.returncode == 0:
            print(r.stdout)
        return

    git_run("init")

    gitignore_path = AGENTS_GIT_DIR / ".gitignore"
    gitignore_path.write_text(
        "sessions/\n*/sessions/\n__pycache__/\n*/__pycache__/\n"
        "*/tools/__pycache__/\nagent/\nmemory/\n*/memory/\n*.pyc\n"
    )

    for agent in AGENTS_TO_JUDGE:
        soul_path = AGENTS_GIT_DIR / agent / "SOUL.md"
        if soul_path.exists():
            git_run("add", str(soul_path))
    git_run("add", str(gitignore_path))

    git_run("config", "user.email", "soul-improver@chimere.local")
    git_run("config", "user.name", "Soul Improver")

    r = git_run("commit", "-m", "soul_improver: initial SOUL.md snapshot", check=False)
    if r.returncode == 0:
        print(f"Git repo initialized. Commit: {git_current_commit()[:8]}")
    else:
        print(f"Git init done but commit failed: {r.stderr[:200]}")


def cmd_revert(args):
    """Revert a specific improvement run."""
    if not args.run_id:
        print("--run-id required")
        return

    conn = init_db()
    row = conn.execute("""
        SELECT git_commit_before, git_commit_after, target_agent, outcome
        FROM improvement_runs WHERE id = ?
    """, (args.run_id,)).fetchone()

    if not row:
        print(f"Run #{args.run_id} not found")
        conn.close()
        return

    commit_before, commit_after, agent, outcome = row

    if outcome == "reverted":
        print(f"Run #{args.run_id} was already reverted")
        conn.close()
        return

    if not commit_before:
        print(f"Run #{args.run_id} has no pre-patch commit to revert to")
        conn.close()
        return

    print(f"Reverting run #{args.run_id} (agent: {agent})")
    print(f"  Rolling back to commit: {commit_before[:8]}")

    if git_revert_to(commit_before):
        revert_commit = git_current_commit()
        conn.execute("""
            UPDATE improvement_runs SET
                outcome = 'reverted',
                git_commit_revert = ?
            WHERE id = ?
        """, (revert_commit, args.run_id))
        conn.commit()
        print(f"  Reverted successfully. Current: {revert_commit[:8]}")
    else:
        print("  Revert FAILED")

    conn.close()


# ── Main ──

def main():
    _load_env()

    parser = argparse.ArgumentParser(
        description="Soul Improver — Autonomous SOUL.md Self-Improvement System")
    subparsers = parser.add_subparsers(dest="command")

    # run
    p_run = subparsers.add_parser("run", help="Full 6-phase improvement cycle")
    p_run.add_argument("--dry-run", action="store_true",
                       help="Generate patch proposal without applying")
    p_run.add_argument("--agent", choices=AGENTS_TO_JUDGE,
                       help="Target specific agent")
    p_run.add_argument("--force", action="store_true",
                       help="Override active hours and cooldown checks")

    # detect
    p_detect = subparsers.add_parser("detect", help="Pattern detection only")
    p_detect.add_argument("--agent", choices=AGENTS_TO_JUDGE)

    # status
    subparsers.add_parser("status", help="Show system status")

    # history
    p_hist = subparsers.add_parser("history", help="Show patch history")
    p_hist.add_argument("--last", type=int, default=10)

    # rules
    p_rules = subparsers.add_parser("rules", help="Rule effectiveness analysis")
    p_rules.add_argument("--agent", choices=AGENTS_TO_JUDGE)

    # git-init
    subparsers.add_parser("git-init", help="Initialize git repo")

    # revert
    p_revert = subparsers.add_parser("revert", help="Revert a specific run")
    p_revert.add_argument("--run-id", type=int, required=True)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    commands = {
        "run": cmd_run,
        "detect": cmd_detect,
        "status": cmd_status,
        "history": cmd_history,
        "rules": cmd_rules,
        "git-init": cmd_git_init,
        "revert": cmd_revert,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
