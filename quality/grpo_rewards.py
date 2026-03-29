#!/usr/bin/env python3
"""GRPO verifiable reward functions for reinforcement learning training.

Provides domain-specific reward functions that return float scores in [0, 1]:
  - code_exec_reward:   execute Python code blocks in a sandboxed subprocess
  - json_schema_reward: validate JSON tool-call structure
  - thinkprm_reward:    delegate scoring to ThinkPRM service
  - combined_reward:    route to the appropriate function by domain
"""

import json
import logging
import re
import subprocess
import sys
import tempfile
import textwrap
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("grpo_rewards")
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CODE_EXEC_TIMEOUT_S = 10
CODE_EXEC_MEM_LIMIT_MB = 256
THINKPRM_URL = "http://127.0.0.1:8085/score"
THINKPRM_TIMEOUT_S = 15
NEUTRAL_SCORE = 0.5

# Regex: fenced Python code blocks (```python ... ``` or ```py ... ```)
_CODE_BLOCK_RE = re.compile(
    r"```(?:python|py)\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

# Regex: JSON object (greedy outermost braces)
_JSON_OBJECT_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


# ---------------------------------------------------------------------------
# 1. Code execution reward
# ---------------------------------------------------------------------------
def code_exec_reward(prompt: str, response: str) -> float:
    """Extract Python code blocks from *response* and execute in a sandbox.

    Returns:
        1.0  if execution exits with code 0
        0.0  if execution fails (non-zero exit, timeout, error)
        0.5  if no code block is found (neutral)
    """
    blocks = _CODE_BLOCK_RE.findall(response)
    if not blocks:
        logger.debug("code_exec_reward: no Python code block found")
        return NEUTRAL_SCORE

    # Concatenate all code blocks (some responses split across multiple)
    code = "\n\n".join(blocks)

    # Write to a temp file so the subprocess has a real script path
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="grpo_exec_", delete=False
    )
    try:
        tmp.write(code)
        tmp.flush()
        tmp.close()

        # Build sandboxed command:
        #   - unshare --net: no network namespace (blocks all network access)
        #   - timeout: hard wall-clock limit
        #   - ulimit -v: virtual memory cap
        mem_kb = CODE_EXEC_MEM_LIMIT_MB * 1024
        shell_cmd = (
            f"ulimit -v {mem_kb} 2>/dev/null; "
            f"exec python3 {tmp.name}"
        )
        result = subprocess.run(
            ["unshare", "--net", "--map-root-user",
             "timeout", str(CODE_EXEC_TIMEOUT_S),
             "bash", "-c", shell_cmd],
            capture_output=True,
            timeout=CODE_EXEC_TIMEOUT_S + 5,  # outer safety margin
        )

        if result.returncode == 0:
            logger.debug("code_exec_reward: success (exit 0)")
            return 1.0
        else:
            stderr_snippet = result.stderr.decode(errors="replace")[:300]
            logger.info(
                "code_exec_reward: failure (exit %d): %s",
                result.returncode, stderr_snippet,
            )
            return 0.0

    except subprocess.TimeoutExpired:
        logger.warning("code_exec_reward: outer timeout (%ds)", CODE_EXEC_TIMEOUT_S + 5)
        return 0.0
    except OSError as exc:
        logger.error("code_exec_reward: OS error: %s", exc)
        return 0.0
    finally:
        Path(tmp.name).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 2. JSON schema reward (tool-call validation)
# ---------------------------------------------------------------------------
def json_schema_reward(prompt: str, response: str) -> float:
    """Check whether *response* contains a valid JSON tool call.

    A valid tool call must be a JSON object with at minimum:
        - "name"      (str)  — the tool/function name
        - "arguments" (dict) — the arguments mapping

    Returns:
        1.0  if valid tool-call JSON found
        0.0  if JSON is present but malformed / missing required keys
        0.5  if no JSON object found at all (neutral)
    """
    candidates = _JSON_OBJECT_RE.findall(response)
    if not candidates:
        logger.debug("json_schema_reward: no JSON object found")
        return NEUTRAL_SCORE

    for raw in candidates:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue

        if not isinstance(obj, dict):
            continue

        # Check required keys
        name = obj.get("name")
        arguments = obj.get("arguments")

        if isinstance(name, str) and name and isinstance(arguments, dict):
            logger.debug("json_schema_reward: valid tool call '%s'", name)
            return 1.0

    # JSON was found but none matched the schema
    logger.info("json_schema_reward: JSON found but no valid tool-call structure")
    return 0.0


# ---------------------------------------------------------------------------
# 3. ThinkPRM reward (external scorer)
# ---------------------------------------------------------------------------
def thinkprm_reward(prompt: str, response: str) -> float:
    """Call the ThinkPRM service and return its normalized score.

    ThinkPRM returns a score in [1, 5]. We normalize to [0, 1]:
        normalized = (score - 1) / 4

    Returns:
        Normalized score on success
        0.5  if the service is unavailable (neutral fallback)
    """
    payload = json.dumps({"prompt": prompt, "response": response}).encode()
    req = urllib.request.Request(
        THINKPRM_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=THINKPRM_TIMEOUT_S) as resp:
            body = json.loads(resp.read().decode())
            raw_score = float(body["score"])
            # Clamp to [1, 5] then normalize to [0, 1]
            clamped = max(1.0, min(5.0, raw_score))
            normalized = (clamped - 1.0) / 4.0
            logger.debug("thinkprm_reward: raw=%.2f normalized=%.3f", raw_score, normalized)
            return normalized
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        logger.warning("thinkprm_reward: service unavailable: %s", exc)
        return NEUTRAL_SCORE
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        logger.warning("thinkprm_reward: bad response: %s", exc)
        return NEUTRAL_SCORE
    except OSError as exc:
        logger.warning("thinkprm_reward: connection error: %s", exc)
        return NEUTRAL_SCORE


# ---------------------------------------------------------------------------
# 4. Combined reward (domain router)
# ---------------------------------------------------------------------------
def combined_reward(prompt: str, response: str, domain: Optional[str] = None) -> float:
    """Route to the appropriate reward function based on *domain*.

    Domains:
        "code"  → code_exec_reward
        "tools" → json_schema_reward
        *       → thinkprm_reward  (default)
    """
    if domain == "code":
        return code_exec_reward(prompt, response)
    elif domain == "tools":
        return json_schema_reward(prompt, response)
    else:
        return thinkprm_reward(prompt, response)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
    logger.setLevel(logging.DEBUG)

    passed = 0
    failed = 0

    def check(name: str, got: float, expected: float, tol: float = 0.01):
        global passed, failed
        ok = abs(got - expected) <= tol
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: got={got:.3f} expected={expected:.3f}")
        if ok:
            passed += 1
        else:
            failed += 1

    # -- code_exec_reward tests --
    print("\n=== code_exec_reward ===")

    check(
        "valid code (print hello)",
        code_exec_reward("", "Here:\n```python\nprint('hello')\n```"),
        1.0,
    )
    check(
        "failing code (syntax error)",
        code_exec_reward("", "```python\nif if if\n```"),
        0.0,
    )
    check(
        "no code block",
        code_exec_reward("", "Just a plain text response."),
        0.5,
    )
    check(
        "code with non-zero exit",
        code_exec_reward("", "```python\nimport sys; sys.exit(1)\n```"),
        0.0,
    )

    # -- json_schema_reward tests --
    print("\n=== json_schema_reward ===")

    valid_tool = textwrap.dedent("""\
        I'll use the search tool:
        {"name": "web_search", "arguments": {"query": "GRPO training"}}
    """)
    check("valid tool call", json_schema_reward("", valid_tool), 1.0)

    bad_tool = textwrap.dedent("""\
        {"name": "search"}
    """)
    check("missing arguments key", json_schema_reward("", bad_tool), 0.0)

    check(
        "no JSON at all",
        json_schema_reward("", "No JSON here, just text."),
        0.5,
    )

    malformed = '{"name": "test", "arguments": "not_a_dict"}'
    check("arguments not a dict", json_schema_reward("", malformed), 0.0)

    # -- thinkprm_reward tests --
    print("\n=== thinkprm_reward ===")

    # ThinkPRM is likely not running, so we expect fallback
    score = thinkprm_reward("What is 2+2?", "4")
    check("fallback when service down", score, 0.5)

    # -- combined_reward tests --
    print("\n=== combined_reward ===")

    check(
        "domain=code routes to code_exec",
        combined_reward("", "```python\nprint(42)\n```", domain="code"),
        1.0,
    )
    check(
        "domain=tools routes to json_schema",
        combined_reward("", valid_tool, domain="tools"),
        1.0,
    )
    check(
        "domain=None routes to thinkprm (fallback)",
        combined_reward("Q", "A", domain=None),
        0.5,
    )

    # -- Summary --
    total = passed + failed
    print(f"\n{'='*40}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed:
        sys.exit(1)
    else:
        print("All tests passed.")
