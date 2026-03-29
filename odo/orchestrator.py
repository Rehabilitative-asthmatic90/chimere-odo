#!/usr/bin/env python3
"""
ODO — One Door Orchestrator.

Intent-based routing proxy that classifies incoming requests and applies
pipeline-specific parameters (sampling, LoRA, engram, GBNF) before
forwarding to the appropriate backend.

Architecture:
  Client → ODO (8085) → think_router (8084) → chimere/llama (8081)
  Client → ODO (8085) → chimere/llama (8081)  [direct mode]

Pipeline configs are loaded from ~/.chimere/odo/pipelines/{route_id}.yaml.
If no pipeline config exists for a route, sensible defaults are used.

Install:
  systemctl --user enable --now odo
  # Update chimere.json: baseUrl → http://127.0.0.1:8085/v1

Usage:
  python3 orchestrator.py                  # default: proxy through think_router
  ODO_DIRECT=1 python3 orchestrator.py     # bypass think_router, go direct
"""

import json
import http.client
import os
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
from urllib.parse import urlparse

# Import sibling classifier
sys.path.insert(0, str(Path(__file__).parent))
from classifier import classify, IMAGE_EXTS

# ── Config ───────────────────────────────────────────────────────────────────

LISTEN_PORT = int(os.environ.get("ODO_PORT", "8085"))
DIRECT_MODE = os.environ.get("ODO_DIRECT", "0") == "1"

# Backend targets
THINK_ROUTER = os.environ.get("ODO_THINK_ROUTER", "http://127.0.0.1:8084")
CHIMERE_DIRECT = os.environ.get("ODO_CHIMERE", "http://127.0.0.1:8081")

BACKEND = CHIMERE_DIRECT if DIRECT_MODE else THINK_ROUTER
FORWARD_TIMEOUT = int(os.environ.get("ODO_TIMEOUT", "300"))

PIPELINES_DIR = Path(__file__).parent / "pipelines"

# ── Pipeline loading ─────────────────────────────────────────────────────────

_pipeline_cache: dict[str, dict] = {}
_pipeline_mtime: dict[str, float] = {}

# Default pipeline config (used when no YAML exists for a route)
DEFAULT_PIPELINE = {
    "temperature": None,       # None = let downstream decide
    "top_p": None,
    "top_k": None,
    "max_tokens": None,
    "presence_penalty": None,
    "thinking": None,          # None = let think_router decide
    "lora": None,              # LoRA adapter path
    "engram_table": None,      # Engram table name
    "gbnf": None,              # GBNF grammar string
    "system_prefix": None,     # Prepended to system message
    "direct": False,           # Bypass think_router for this route
}


def _load_yaml(path: Path) -> dict:
    """Load a YAML file using PyYAML if available, else basic parser."""
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # Minimal YAML parser for simple key: value files
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
                                # Strip quotes
                                if (val.startswith('"') and val.endswith('"')) or \
                                   (val.startswith("'") and val.endswith("'")):
                                    val = val[1:-1]
                    result[key.strip()] = val
        return result


def load_pipeline(route_id: str) -> dict:
    """Load pipeline config from YAML, with mtime-based caching (hot reload)."""
    yaml_path = PIPELINES_DIR / f"{route_id}.yaml"

    if not yaml_path.exists():
        return dict(DEFAULT_PIPELINE)

    try:
        mtime = yaml_path.stat().st_mtime
    except OSError:
        return dict(DEFAULT_PIPELINE)

    cached_mtime = _pipeline_mtime.get(route_id, 0)
    if route_id in _pipeline_cache and mtime <= cached_mtime:
        return dict(_pipeline_cache[route_id])

    try:
        data = _load_yaml(yaml_path)
        merged = dict(DEFAULT_PIPELINE)
        merged.update(data)
        _pipeline_cache[route_id] = merged
        _pipeline_mtime[route_id] = mtime
        return dict(merged)
    except Exception as e:
        print(f"[odo] warning: failed to load {yaml_path}: {e}",
              file=sys.stderr, flush=True)
        return dict(DEFAULT_PIPELINE)


# ── Request analysis helpers ─────────────────────────────────────────────────

def _extract_user_text(payload: dict) -> str:
    """Extract the last user message text from the payload."""
    messages = payload.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # Multimodal content array
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
                return " ".join(parts)
    return ""


def _detect_image(payload: dict) -> bool:
    """Check if any message contains base64 or URL images."""
    for msg in payload.get("messages", []):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                return True
    return False


def _extract_files(payload: dict) -> list[str]:
    """Extract filenames from the request (metadata or content references)."""
    files = []
    # Check for ODO-specific metadata
    meta = payload.get("odo_metadata", {})
    if meta.get("files"):
        files.extend(meta["files"])
    return files


def _apply_pipeline(payload: dict, pipeline: dict, route_info: dict) -> tuple[dict, str]:
    """Apply pipeline-specific parameters to the request payload.

    Returns (modified_payload, backend_url).
    """
    result = dict(payload)
    # Remove ODO metadata before forwarding
    result.pop("odo_metadata", None)

    # Sampling parameters — only override if pipeline specifies them
    for param in ("temperature", "top_p", "top_k", "max_tokens", "presence_penalty"):
        val = pipeline.get(param)
        if val is not None:
            result[param] = val

    # Thinking mode override
    thinking = pipeline.get("thinking")
    if thinking is not None:
        result["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}

    # LoRA adapter
    lora = pipeline.get("lora")
    if lora:
        result["lora"] = lora

    # Engram table
    engram = pipeline.get("engram_table")
    if engram:
        result["engram_table"] = engram

    # GBNF grammar
    gbnf = pipeline.get("gbnf")
    if gbnf:
        result["grammar"] = gbnf

    # System prefix — prepend to existing system message
    prefix = pipeline.get("system_prefix")
    if prefix and "messages" in result:
        messages = result["messages"]
        if messages and messages[0].get("role") == "system":
            original = messages[0].get("content", "")
            messages[0]["content"] = f"{prefix}\n\n{original}"
        else:
            messages.insert(0, {"role": "system", "content": prefix})

    # Inject classification metadata as a header hint (for logging/debugging)
    result.setdefault("odo_route", route_info.get("route", "general"))

    # Backend selection: per-pipeline direct mode overrides global setting
    if pipeline.get("direct") or DIRECT_MODE:
        backend = CHIMERE_DIRECT
    else:
        backend = THINK_ROUTER

    return result, backend


# ── HTTP forwarding ──────────────────────────────────────────────────────────

def _forward_raw(path: str, body: bytes, headers: dict,
                 backend: str, timeout: int = 300):
    """Forward a raw request to the specified backend."""
    parsed = urlparse(backend)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    conn.request("POST", path, body=body, headers=headers)
    resp = conn.getresponse()
    return resp, conn


def _proxy_get(path: str, backend: str):
    """Forward a GET request to the backend."""
    parsed = urlparse(backend)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=30)
    conn.request("GET", path)
    resp = conn.getresponse()
    resp_body = resp.read()
    conn.close()
    return resp.status, resp.getheader("Content-Type", "application/json"), resp_body


# ── Handler ──────────────────────────────────────────────────────────────────

class ODOHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass  # suppress default access logs

    def do_GET(self):
        if self.path == "/health":
            self._json_response(200, {
                "status": "ok",
                "port": LISTEN_PORT,
                "backend": BACKEND,
                "direct_mode": DIRECT_MODE,
            })
        elif self.path == "/routes":
            self._json_response(200, {
                "available": list(load_pipeline("__list__").keys())
                if False else self._list_routes(),
            })
        elif self.path.startswith("/v1/models"):
            try:
                status, ct, body = _proxy_get(self.path, BACKEND)
                self.send_response(status)
                self.send_header("Content-Type", ct)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                self.send_error(502, str(e))
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

        # ── Classify ──
        user_text = _extract_user_text(payload)
        has_image = _detect_image(payload)
        files = _extract_files(payload)

        route_info = classify(user_text, files=files or None, has_image=has_image)
        route_id = route_info["route"]

        # ── Load pipeline ──
        pipeline = load_pipeline(route_id)

        # ── Apply pipeline params ──
        modified_payload, target_backend = _apply_pipeline(payload, pipeline, route_info)

        # Remove internal fields before forwarding
        modified_payload.pop("odo_route", None)

        classify_ms = int((time.time() - t0) * 1000)
        print(f"[odo] route={route_id} conf={route_info['confidence']:.2f} "
              f"strategy={route_info['strategy']} classify={classify_ms}ms "
              f"backend={target_backend}",
              file=sys.stderr, flush=True)

        # ── Forward ──
        fwd_body = json.dumps(modified_payload).encode()
        is_streaming = modified_payload.get("stream", False)

        try:
            resp, conn = _forward_raw(self.path, fwd_body, {
                "Content-Type": "application/json",
                "Content-Length": str(len(fwd_body)),
            }, backend=target_backend, timeout=FORWARD_TIMEOUT)
        except Exception as e:
            self.send_error(502, f"Backend error: {e}")
            return

        if is_streaming:
            self._stream_response(resp, conn)
        else:
            self._buffer_response(resp, conn)

        total_ms = int((time.time() - t0) * 1000)
        print(f"[odo] completed route={route_id} total={total_ms}ms",
              file=sys.stderr, flush=True)

    def _stream_response(self, resp, conn):
        """Stream SSE response from backend to client."""
        self.send_response(resp.status)
        self.send_header("Content-Type",
                         resp.getheader("Content-Type", "text/event-stream"))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            while True:
                line = resp.readline()
                if not line:
                    break
                self.wfile.write(line)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            conn.close()

    def _buffer_response(self, resp, conn):
        """Buffer and forward a non-streaming response."""
        try:
            resp_body = resp.read()
        finally:
            conn.close()
        self.send_response(resp.status)
        self.send_header("Content-Type",
                         resp.getheader("Content-Type", "application/json"))
        self.send_header("Content-Length", str(len(resp_body)))
        self.end_headers()
        self.wfile.write(resp_body)

    def _proxy_post(self, body):
        """Transparent POST proxy for non-chat endpoints."""
        try:
            resp, conn = _forward_raw(self.path, body, {
                "Content-Type": self.headers.get("Content-Type", "application/json"),
                "Content-Length": str(len(body)),
            }, backend=BACKEND, timeout=FORWARD_TIMEOUT)
            resp_body = resp.read()
            conn.close()
        except Exception as e:
            self.send_error(502, str(e))
            return
        self.send_response(resp.status)
        self.send_header("Content-Type",
                         resp.getheader("Content-Type", "application/json"))
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
        """List available pipeline configs and their status."""
        routes = {}
        for yaml_file in sorted(PIPELINES_DIR.glob("*.yaml")):
            route_id = yaml_file.stem
            try:
                cfg = load_pipeline(route_id)
                routes[route_id] = {
                    "configured": True,
                    "direct": cfg.get("direct", False),
                    "lora": cfg.get("lora") is not None,
                    "gbnf": cfg.get("gbnf") is not None,
                }
            except Exception:
                routes[route_id] = {"configured": False}
        return routes

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()


# ── Server ───────────────────────────────────────────────────────────────────

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    PIPELINES_DIR.mkdir(parents=True, exist_ok=True)

    server = ThreadedHTTPServer(("127.0.0.1", LISTEN_PORT), ODOHandler)
    mode = "DIRECT" if DIRECT_MODE else f"via think_router ({THINK_ROUTER})"
    print(f"[odo] listening on 127.0.0.1:{LISTEN_PORT}", flush=True)
    print(f"[odo] mode: {mode}", flush=True)
    print(f"[odo] pipelines: {PIPELINES_DIR}", flush=True)
    print(f"[odo] forward timeout: {FORWARD_TIMEOUT}s", flush=True)

    n_pipelines = len(list(PIPELINES_DIR.glob("*.yaml")))
    print(f"[odo] loaded {n_pipelines} pipeline config(s)", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[odo] shutting down", flush=True)
        server.shutdown()


if __name__ == "__main__":
    main()
