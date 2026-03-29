#!/usr/bin/env python3
"""
Pipeline Executor -- sequential multi-agent pipeline execution for ODO.

When a pipeline YAML defines steps (e.g., kine.yaml has evidence_search -> diagnostic
-> protocol -> dosage), this module executes them sequentially via llama-server (8081).

Each step receives:
  - The route's system_prompt + the step's system_suffix
  - The user's original question
  - Accumulated context from all previous steps

The final step's output is returned as the response.
"""

import json
import http.client
import sys
import time
from urllib.parse import urlparse


def execute_pipeline(steps: list, user_text: str, system_prompt: str,
                     backend_url: str, thinking_enabled: bool = True,
                     timeout: int = 300) -> dict:
    """Execute a multi-agent pipeline sequentially.

    Args:
        steps: List of pipeline step dicts from YAML (agent, params, system_suffix, ...).
        user_text: Original user message text.
        system_prompt: Route-level system prompt (from pipeline YAML top-level).
        backend_url: llama-server URL (e.g., http://127.0.0.1:8081).
        thinking_enabled: Whether to enable thinking mode for steps.
        timeout: HTTP timeout per step in seconds.

    Returns:
        dict with keys: content, steps_log, total_ms, total_tokens
    """
    accumulated_context = []
    steps_log = []
    total_tokens = 0
    t_pipeline = time.time()

    for i, step in enumerate(steps):
        agent_name = step.get("agent", f"step_{i}")
        step_params = step.get("params", {})
        system_suffix = step.get("system_suffix", "")
        step_temp = step_params.get("temperature", 0.6)
        step_max_tokens = step_params.get("max_tokens", 4096)

        # Build system message: route system_prompt + step suffix + accumulated context
        system_parts = []
        if system_prompt:
            system_parts.append(system_prompt.strip())
        if system_suffix:
            system_parts.append(f"[Step {i+1}/{len(steps)}: {agent_name}]\n{system_suffix.strip()}")
        if accumulated_context:
            ctx_block = "\n\n---\n\n".join(accumulated_context)
            system_parts.append(f"[Context from previous steps]\n{ctx_block}")

        system_content = "\n\n".join(system_parts)

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_text},
        ]

        payload = {
            "messages": messages,
            "temperature": step_temp,
            "top_p": step_params.get("top_p", 0.95),
            "top_k": step_params.get("top_k", 20),
            "presence_penalty": step_params.get("presence_penalty", 0.0),
            "max_tokens": step_max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": thinking_enabled},
        }

        # Ensure thinking gets enough budget
        if thinking_enabled and step_max_tokens < 4096:
            payload["max_tokens"] = 4096

        t_step = time.time()
        try:
            result = _send_request(backend_url, payload, timeout)
        except Exception as e:
            error_msg = f"Step {i+1} ({agent_name}) failed: {e}"
            print(f"[pipeline] {error_msg}", file=sys.stderr, flush=True)
            steps_log.append({
                "agent": agent_name, "step": i + 1,
                "error": str(e), "ms": int((time.time() - t_step) * 1000),
            })
            # On error, return what we have so far
            if accumulated_context:
                return {
                    "content": accumulated_context[-1],
                    "steps_log": steps_log,
                    "total_ms": int((time.time() - t_pipeline) * 1000),
                    "total_tokens": total_tokens,
                    "partial": True,
                }
            return {
                "content": f"Pipeline failed at step {i+1} ({agent_name}): {e}",
                "steps_log": steps_log,
                "total_ms": int((time.time() - t_pipeline) * 1000),
                "total_tokens": total_tokens,
                "partial": True,
            }

        step_ms = int((time.time() - t_step) * 1000)

        # Extract response content
        choice = result.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content", "")
        reasoning = msg.get("reasoning_content", "")

        # Token accounting
        usage = result.get("usage", {})
        step_tokens = usage.get("completion_tokens", 0) + usage.get("prompt_tokens", 0)
        total_tokens += step_tokens

        # Accumulate context for next step
        accumulated_context.append(f"[{agent_name}]\n{content}")

        step_info = {
            "agent": agent_name,
            "step": i + 1,
            "ms": step_ms,
            "tokens": step_tokens,
            "output_chars": len(content),
        }
        steps_log.append(step_info)

        print(f"[pipeline] step {i+1}/{len(steps)} agent={agent_name} "
              f"{step_ms}ms {step_tokens}tok {len(content)}chars",
              file=sys.stderr, flush=True)

    total_ms = int((time.time() - t_pipeline) * 1000)
    print(f"[pipeline] complete: {len(steps)} steps {total_ms}ms {total_tokens}tok",
          file=sys.stderr, flush=True)

    return {
        "content": accumulated_context[-1] if accumulated_context else "",
        "steps_log": steps_log,
        "total_ms": total_ms,
        "total_tokens": total_tokens,
        "partial": False,
    }


def _send_request(backend_url: str, payload: dict, timeout: int) -> dict:
    """Send a single completion request to llama-server."""
    body = json.dumps(payload).encode()
    parsed = urlparse(backend_url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    try:
        conn.request("POST", "/v1/chat/completions", body=body, headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        })
        resp = conn.getresponse()
        data = json.loads(resp.read())
    finally:
        conn.close()
    return data


def should_use_pipeline(pipeline: dict, payload: dict) -> bool:
    """Check if this request should use multi-step pipeline execution.

    Pipeline execution activates when:
      1. The pipeline YAML has a non-empty 'pipeline' steps list, AND
      2. The request has "pipeline": true in the payload, OR the route
         has pipeline_auto: true in the YAML.
    """
    steps = pipeline.get("pipeline")
    if not steps or not isinstance(steps, list) or len(steps) < 2:
        return False

    # Explicit request-level toggle
    if payload.get("pipeline") is True:
        return True

    # Route-level auto toggle
    if pipeline.get("pipeline_auto", False):
        return True

    return False
