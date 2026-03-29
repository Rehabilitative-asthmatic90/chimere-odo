#!/usr/bin/env python3
"""
knowledge_rag_query.py — Query ChromaDB knowledge base.

Usage:
    python3 knowledge_rag_query.py "lombalgie chronique exercices"
    python3 knowledge_rag_query.py --collection medical "coiffe rotateurs"
    python3 knowledge_rag_query.py --collection code "vllm lora adapter"
    python3 knowledge_rag_query.py --json "tendinopathie" --max 5
    python3 knowledge_rag_query.py --no-rerank "tendinopathie"
    python3 knowledge_rag_query.py --rerank-model cross-encoder/ms-marco-multilingual-MiniLM-L-12-v2 "query"
"""

import argparse
import json
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = Path.home() / ".chimere/data/chromadb"
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-multilingual-MiniLM-L-12-v2"

# How many results to pull from ChromaDB before re-ranking
RERANK_FETCH_K = 20

# RRF constant (higher = less weight to rank position)
RRF_K = 60

# Singleton for embedder (expensive to load)
_embedder = None

# Singleton for cross-encoder (lazy-loaded on first use)
_cross_encoder = None
_cross_encoder_model_name = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_cross_encoder(model_name: str = DEFAULT_RERANK_MODEL):
    """Lazy-load the cross-encoder singleton. Returns None on failure."""
    global _cross_encoder, _cross_encoder_model_name

    # If already loaded with the same model, reuse it
    if _cross_encoder is not None and _cross_encoder_model_name == model_name:
        return _cross_encoder

    try:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(model_name)
        _cross_encoder_model_name = model_name
        return _cross_encoder
    except ImportError:
        print(
            "[WARNING] CrossEncoder not available in sentence_transformers. "
            "Re-ranking disabled.",
            file=sys.stderr,
        )
        return None
    except Exception as e:
        print(
            f"[WARNING] Failed to load cross-encoder model '{model_name}': {e}. "
            "Re-ranking disabled.",
            file=sys.stderr,
        )
        return None


def rerank_results(
    query: str,
    results: list[dict],
    model_name: str = DEFAULT_RERANK_MODEL,
    top_n: int = 5,
) -> list[dict]:
    """
    Re-rank results with a cross-encoder.

    Each result dict must have a "text" key.
    Returns up to top_n results sorted by cross-encoder score (descending).
    The "score" field is replaced with the cross-encoder score; the original
    cosine similarity is preserved as "embed_score".
    """
    if not results:
        return results

    cross_encoder = get_cross_encoder(model_name)
    if cross_encoder is None:
        # Graceful degradation: return cosine-sorted results truncated to top_n
        return results[:top_n]

    pairs = [(query, r["text"]) for r in results]
    ce_scores = cross_encoder.predict(pairs)

    for r, score in zip(results, ce_scores):
        r["embed_score"] = r["score"]   # preserve original cosine similarity
        r["score"] = round(float(score), 4)

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_n]


def _bm25_search(client, query: str, collections: list[str], fetch_k: int) -> list[dict]:
    """BM25 keyword search using ChromaDB's where_document $contains.

    Simple but effective: splits query into keywords, searches for documents
    containing those keywords. Scores by keyword hit count.
    """
    # Extract meaningful keywords (>= 3 chars, no stopwords)
    stopwords_fr = {"les", "des", "une", "pour", "dans", "avec", "sur", "par", "est", "sont",
                    "pas", "que", "qui", "quoi", "quel", "quelle", "quels", "quelles",
                    "comment", "cette", "ces"}
    keywords = [w.lower() for w in query.split() if len(w) >= 3 and w.lower() not in stopwords_fr]

    if not keywords:
        return []

    bm25_results = []
    for coll_name in collections:
        try:
            coll = client.get_collection(coll_name)
        except Exception:
            continue

        for kw in keywords[:4]:  # limit to 4 keywords to avoid too many queries
            try:
                results = coll.get(
                    where_document={"$contains": kw},
                    limit=fetch_k,
                    include=["documents", "metadatas"]
                )
                if not results["documents"]:
                    continue

                for doc, meta in zip(results["documents"], results["metadatas"]):
                    # Score by how many keywords appear in the document
                    doc_lower = doc.lower()
                    hits = sum(1 for k in keywords if k in doc_lower)
                    bm25_results.append({
                        "text": doc,
                        "bm25_hits": hits,
                        "collection": coll_name,
                        "title": meta.get("title", ""),
                        "section": meta.get("section", ""),
                        "category": meta.get("category", ""),
                        "account": meta.get("account", ""),
                        "source": meta.get("source", ""),
                        "file_path": meta.get("file_path", ""),
                    })
            except Exception:
                continue

    # Deduplicate by text content (same chunk may match multiple keywords)
    seen = set()
    deduped = []
    for r in bm25_results:
        key = r["text"][:100]
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    # Sort by keyword hit count (descending)
    deduped.sort(key=lambda r: -r["bm25_hits"])
    return deduped[:fetch_k]


def _rrf_fusion(dense_results: list[dict], sparse_results: list[dict],
                k: int = RRF_K) -> list[dict]:
    """Reciprocal Rank Fusion of dense (embedding) and sparse (BM25) results.

    RRF score = sum(1 / (k + rank_i)) for each result list containing the doc.
    """
    scores: dict[str, float] = {}
    result_map: dict[str, dict] = {}

    for rank, r in enumerate(dense_results):
        key = r["text"][:100]
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
        result_map[key] = r

    for rank, r in enumerate(sparse_results):
        key = r["text"][:100]
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
        if key not in result_map:
            result_map[key] = r

    # Sort by RRF score
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    fused = []
    for key, rrf_score in ranked:
        r = dict(result_map[key])
        r["rrf_score"] = round(rrf_score, 6)
        r.pop("bm25_hits", None)
        fused.append(r)

    return fused


def query_rag(
    query: str,
    collection: str = "auto",
    max_results: int = 5,
    min_score: float = 0.25,
    rerank: bool = True,
    rerank_model: str = DEFAULT_RERANK_MODEL,
    hybrid: bool = True,
) -> list[dict]:
    """
    Query ChromaDB with hybrid search (dense + BM25 keyword, RRF fusion).

    Pipeline:
      1. Dense search (Qwen3-Embedding cosine similarity)
      2. BM25 keyword search ($contains on ChromaDB)
      3. RRF fusion of both result lists
      4. Optional cross-encoder reranking on top

    When hybrid=False, falls back to dense-only (original behaviour).
    """
    if not CHROMA_DIR.exists():
        return []

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedder = get_embedder()

    query_emb = embedder.encode(
        [query], normalize_embeddings=True
    ).tolist()

    collections_to_query = ["medical", "code"] if collection == "auto" else [collection]
    fetch_k = RERANK_FETCH_K if rerank else max_results

    # 1. Dense search (embedding cosine similarity)
    dense_results = []
    for coll_name in collections_to_query:
        try:
            coll = client.get_collection(coll_name)
        except Exception:
            continue

        results = coll.query(
            query_embeddings=query_emb,
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"]
        )

        if not results["documents"] or not results["documents"][0]:
            continue

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            score = 1.0 - dist
            if score >= min_score:
                dense_results.append({
                    "text": doc,
                    "score": round(score, 4),
                    "collection": coll_name,
                    "title": meta.get("title", ""),
                    "section": meta.get("section", ""),
                    "category": meta.get("category", ""),
                    "account": meta.get("account", ""),
                    "source": meta.get("source", ""),
                    "file_path": meta.get("file_path", ""),
                })

    dense_results.sort(key=lambda r: r["score"], reverse=True)

    # 2. BM25 keyword search + RRF fusion
    if hybrid:
        sparse_results = _bm25_search(client, query, collections_to_query, fetch_k)
        if sparse_results:
            all_results = _rrf_fusion(dense_results, sparse_results)
            # Preserve a score field for compatibility
            for r in all_results:
                if "score" not in r:
                    r["score"] = r.get("rrf_score", 0)
        else:
            all_results = dense_results
    else:
        all_results = dense_results

    if rerank:
        return rerank_results(query, all_results, model_name=rerank_model, top_n=max_results)

    return all_results[:max_results]


def format_results_text(results: list[dict]) -> str:
    """Format results as readable text for LLM context injection."""
    if not results:
        return ""
    parts = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "?")
        score = r.get("score", 0)
        embed_score = r.get("embed_score")  # present only when re-ranked
        section = r.get("section", "")
        text = r.get("text", "")[:300]
        account = r.get("account", "")
        source_info = f" ({account})" if account else ""
        section_info = f" | {section}" if section else ""
        if embed_score is not None:
            score_str = f"ce: {score:.4f}, emb: {embed_score:.2f}"
        else:
            score_str = f"{score:.2f}"
        parts.append(
            f"**{i}. {title}**{source_info} (score: {score_str}{section_info})\n{text}"
        )
    return "\n\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Query ChromaDB knowledge base")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--collection", "-c", default="auto",
                        choices=["auto", "medical", "code"],
                        help="Collection to query (default: auto)")
    parser.add_argument("--max", "-m", type=int, default=5,
                        help="Max results (default: 5)")
    parser.add_argument("--json", "-j", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--min-score", type=float, default=0.25,
                        help="Minimum similarity score (default: 0.25)")

    # Re-ranking flags
    rerank_group = parser.add_mutually_exclusive_group()
    rerank_group.add_argument(
        "--rerank",
        dest="rerank",
        action="store_true",
        default=True,
        help="Enable cross-encoder re-ranking (default: enabled)",
    )
    rerank_group.add_argument(
        "--no-rerank",
        dest="rerank",
        action="store_false",
        help="Disable cross-encoder re-ranking",
    )
    parser.add_argument(
        "--rerank-model",
        default=DEFAULT_RERANK_MODEL,
        metavar="MODEL",
        help=(
            f"Cross-encoder model to use for re-ranking "
            f"(default: {DEFAULT_RERANK_MODEL})"
        ),
    )

    # Hybrid search flags
    hybrid_group = parser.add_mutually_exclusive_group()
    hybrid_group.add_argument(
        "--hybrid", dest="hybrid", action="store_true", default=True,
        help="Enable hybrid search: dense + BM25 keyword + RRF fusion (default: enabled)",
    )
    hybrid_group.add_argument(
        "--no-hybrid", dest="hybrid", action="store_false",
        help="Disable hybrid search (dense-only)",
    )

    args = parser.parse_args()

    results = query_rag(
        args.query,
        args.collection,
        args.max,
        args.min_score,
        rerank=args.rerank,
        rerank_model=args.rerank_model,
        hybrid=args.hybrid,
    )

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        if not results:
            print("Aucun résultat trouvé.")
        else:
            print(format_results_text(results))


if __name__ == "__main__":
    main()
