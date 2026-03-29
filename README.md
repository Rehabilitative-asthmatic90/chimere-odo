# Chimere ODO

**Unified inference orchestrator for local LLM deployments: intent classification, adaptive compute routing, quality-gated self-improvement, and SOTA web search pipeline.**

ODO sits between user requests and a local llama-server, adding intelligence layers: it classifies intent, enriches context with RAG and web search, routes to the right compute profile, and logs quality scores for nightly self-improvement.

## Architecture

```
User request → ODO (port 8084)
                 ├── Intent classifier (keyword → filetype → LLM cascade)
                 ├── Enricher (web search, RAG, tool injection)
                 ├── Entropy router (fast/quality/ultra compute modes)
                 ├── Semantic few-shot (FAISS, domain-specific)
                 ├── Confidence RAG trigger (logprob probe)
                 ├── Dynamic Engram (web → n-gram bias per query)
                 ├── DVTS tree search (K candidates + PRM scoring)
                 └── Quality gate (ThinkPRM step-level verification)
               → llama-server (port 8081)
               → quality_scores.jsonl → nightly training loop
```

## What's in this repo

```
odo/                    Core orchestrator
  odo.py                Main proxy server (~1500 lines)
  classifier.py         Intent classification (3-strategy cascade)
  enricher.py           Context enrichment (web, RAG, tools)
  entropy_router.py     Adaptive compute allocation
  quality_gate.py       Response quality scoring
  dvts.py               Diverse Verifier Tree Search
  semantic_fewshot.py   FAISS-based example selection
  dynamic_engram.py     Per-query n-gram bias injection
  confidence_rag_trigger.py  Logprob-based RAG triggering
  pipeline_executor.py  Multi-step pipeline execution
  orchestrator.py       Core orchestration engine
  pipelines/            YAML pipeline definitions (code, kine, cyber, research)
  few_shot/             Domain-specific few-shot examples

engram/                 Engram memory management
  engram_ingest.py      Build .engr hash tables from documents
  engram_write_nightly.py  Quality-gated nightly Engram updates
  engram_query.py       Query Engram tables

search/                 Web search pipeline
  deep_search_sota.py   8-stage SOTA search (expand → search → RRF → fetch → rerank → CRAG → synthesize)
  search_router.py      Brave/SearXNG/Perplexica routing with budget management
  brave_search.py       Brave Search API client
  searxng_search.py     SearXNG integration

quality/                Self-improvement loop
  soul_improver.py      Autonomous system prompt optimization (6-phase cycle)
  dspy_optimize.py      DSPy MIPROv2 weekly prompt optimization
  nightly_lora.py       Overnight LoRA training from quality-filtered pairs
  debate_router.py      Multi-agent debate (advocate/critic/synthesizer)

knowledge/              Knowledge ingestion
  ingest_pipeline.py    YouTube, Instagram, web article ingestion
  ocr_glm.py            GLM-OCR document processing (94.6% OmniDocBench)

think_router.py         Entropy-based thinking/no-think proxy
```

## Key features

- **Zero-config intent classification**: 3-strategy cascade (regex → filetype → LLM) handles 99% of requests in <1ms
- **Adaptive compute**: entropy-based routing between fast (no-think), quality (think), and ultra (DVTS) modes
- **Self-improving**: quality scores → training pairs → nightly LoRA/SPIN → Engram WRITE
- **SOTA web search**: query expansion → parallel search → RRF fusion → deep fetch → CRAG filtering → LLM synthesis
- **Pipeline YAML**: define multi-step agent workflows in YAML with hot-reload

## Usage

```bash
# Start ODO (requires a running llama-server on port 8081)
python odo/odo.py --port 8084

# Query Engram tables
python engram/engram_query.py ~/.chimere/data/engram/kine.engr "tendon achille"

# Run SOTA web search
python search/deep_search_sota.py "latest advances in MoE quantization" --depth deep
```

## Related repos

- [chimere](https://github.com/AIdevsmartdata/chimere) — Rust inference runtime + DFlash drafter
- [ramp-quant](https://github.com/AIdevsmartdata/ramp-quant) — RAMP mixed-precision quantization

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Author

**Kevin Remondiere** — Independent ML researcher, Bayonne, France
