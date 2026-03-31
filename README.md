# Chimere ODO

**Unified inference orchestrator for local LLM deployments — intent classification, adaptive routing, quality-gated self-improvement, and SOTA web search.**

ODO sits between user requests and a local llama-server, intelligently routing, enriching, and quality-gating every interaction.

## Powered by Chimere Distilled

ODO is designed to work with [Chimere Distilled](https://huggingface.co/Kevletesteur/Qwen3.5-35B-A3B-Chimere-Distilled-GGUF) — our Claude Opus 4.6 distillation of Qwen3.5-35B-A3B (MoE).

| Metric | Score |
|--------|-------|
| HumanEval | **97%** |
| BFCL tool-calling | **85%** (+18 pts vs base) |
| IFEval | **80%** |
| GGUF size | **15 GB** (fits 16 GB VRAM) |

## Architecture

```
User → ODO (intent classify → enrich → route) → llama-server → ODO (quality gate) → User
```

### Pipeline

1. **Intent Classification**: 3-strategy cascade (regex → filetype → LLM)
2. **Context Enrichment**: Web search, ChromaDB RAG, tool injection
3. **Adaptive Routing**: Entropy-based compute profiles (thinking vs no-think)
4. **Quality Assessment**: Scoring + continuous improvement
5. **Search Pipeline**: 8-stage SOTA web search (QueryExpand → WebSearch → RRF → DeepFetch → Diversity → CRAG → Contradictions → Synthesis)

### Features

- Zero-config intent handling
- YAML-based pipeline definitions (hot-reload)
- Adaptive compute allocation (think/no-think routing)
- Autonomous self-improvement (overnight LoRA + DSPy)
- Engram memory management (semantic few-shot, n-gram bias)
- DVTS tree search with PRM scoring
- Knowledge ingestion (YouTube, Instagram, GLM-OCR)

## Quick Start

### Docker

```bash
docker compose up -d
# ODO on port 8084, llama-server on port 8081
```

### Standalone

```bash
pip install -r requirements.txt
export ODO_BACKEND=http://127.0.0.1:8081
python odo.py
```

## Configuration

All via environment variables:
- `ODO_BACKEND`: llama-server URL (default: `http://127.0.0.1:8081`)
- `ODO_PORT`: ODO listening port (default: `8084`)
- `CHIMERE_HOME`: Data directory (default: `~/.chimere`)

Pipeline YAMLs in `pipelines/` for per-route customization.

## Related Projects

- [chimere](https://github.com/AIdevsmartdata/chimere) — Rust inference runtime + distilled models
- [Chimere Distilled GGUF](https://huggingface.co/Kevletesteur/Qwen3.5-35B-A3B-Chimere-Distilled-GGUF) — 15 GB GGUF for local inference

## License

Apache 2.0 — Kevin Remondiere
