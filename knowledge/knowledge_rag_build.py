#!/usr/bin/env python3
"""
knowledge_rag_build.py — Index all knowledge files into ChromaDB.

Usage:
    python3 knowledge_rag_build.py                # full rebuild
    python3 knowledge_rag_build.py --incremental  # only new/modified files
    python3 knowledge_rag_build.py --stats         # show collection stats
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
KNOWLEDGE_DIR = Path.home() / ".chimere/workspaces/main/knowledge"
CHROMA_DIR = Path.home() / ".chimere/data/chromadb"
MANIFEST_PATH = CHROMA_DIR / "index_manifest.json"
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"

CHUNK_SIZE = 600      # chars per chunk
CHUNK_OVERLAP = 100
BATCH_SIZE = 32       # embedding batch size

# Directory -> collection mapping
COLLECTION_MAP = {
    "kine-sante": "medical",
    "sport-performance": "medical",
    "syntheses": "medical",
    "dev-ia": "code",
    "web": "code",
    "youtube": "code",
    "instagram": "medical",  # mostly kine content
}

# Chimère config docs — indexed into "chimere" collection
CHIMERE_DOC_DIRS = [
    Path.home() / "Bureau",
    Path.home() / ".chimere/workspaces/kevin",
    Path.home() / ".claude/projects/-home-remondiere/memory",
]
# Files to skip in chimere doc dirs (noisy or not useful for RAG)
CHIMERE_SKIP_PATTERNS = {"HEARTBEAT.md", "BOOTSTRAP.md"}


# ── Metadata extraction ──────────────────────────────────────────────────────
def extract_metadata(text: str, file_path: Path) -> dict:
    """Extract structured metadata from knowledge file header."""
    meta = {
        "file_path": str(file_path),
        "title": file_path.stem.replace("-", " ").replace("_", " "),
    }

    # Parse header fields: - **Key** : value
    patterns = {
        "source": r'\*\*Source\*\*\s*:\s*(.+)',
        "date": r'\*\*Date de publication\*\*\s*:\s*(.+)',
        "category": r'\*\*Categorie\*\*\s*:\s*(.+)',
        "account": r'\*\*Compte\*\*\s*:\s*(.+)',
        "type": r'\*\*Type\*\*\s*:\s*(.+)',
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, text[:1000])
        if m:
            meta[key] = m.group(1).strip()

    # Extract title from first # heading
    title_match = re.match(r'^#\s+(.+)', text, re.MULTILINE)
    if title_match:
        meta["title"] = title_match.group(1).strip()

    return meta


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_file(text: str, file_key: str, meta: dict) -> list[dict]:
    """Chunk a markdown file by ## sections, then by size."""
    chunks = []

    # Skip files with no useful content
    if "ne contient aucun contenu scientifique" in text.lower():
        return []
    if "aucune fiche de connaissance ne peut" in text.lower():
        return []

    # Remove raw content blocks (usually duplicated)
    text = re.sub(
        r'<details>\s*<summary>.*?</summary>.*?</details>',
        '', text, flags=re.DOTALL
    )

    # Split on ## headings
    sections = re.split(r'\n(?=##\s)', text)
    chunk_idx = 0

    for section in sections:
        # Extract section title
        section_title = ""
        sec_match = re.match(r'##\s+(.+)', section)
        if sec_match:
            section_title = sec_match.group(1).strip()

        # Split section into paragraphs
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', section) if p.strip()]
        current = ""

        for para in paragraphs:
            if len(current) + len(para) > CHUNK_SIZE and current:
                chunk_id = f"{file_key}_{chunk_idx:04d}"
                chunks.append({
                    "id": chunk_id,
                    "text": current.strip(),
                    "metadata": {
                        **{k: str(v)[:200] for k, v in meta.items()},
                        "section": section_title[:100],
                        "chunk_idx": chunk_idx,
                    }
                })
                current = current[-CHUNK_OVERLAP:] + " " + para if CHUNK_OVERLAP > 0 else para
                chunk_idx += 1
            else:
                current = (current + "\n\n" + para).strip()

        # Last chunk of section
        if current.strip() and len(current.strip()) > 50:
            chunks.append({
                "id": f"{file_key}_{chunk_idx:04d}",
                "text": current.strip(),
                "metadata": {
                    **{k: str(v)[:200] for k, v in meta.items()},
                    "section": section_title[:100],
                    "chunk_idx": chunk_idx,
                }
            })
            chunk_idx += 1

    return chunks


def get_collection_for_file(file_path: Path) -> str:
    """Determine which collection a file belongs to."""
    rel = file_path.relative_to(KNOWLEDGE_DIR)
    top_dir = rel.parts[0] if rel.parts else ""
    return COLLECTION_MAP.get(top_dir, "code")


def make_file_key(file_path: Path) -> str:
    """Create a unique key for a file."""
    rel = file_path.relative_to(KNOWLEDGE_DIR)
    return str(rel).replace("/", "_").replace(".md", "")


# ── Indexing ──────────────────────────────────────────────────────────────────
def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def save_manifest(manifest: dict):
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def build_index(incremental: bool = False):
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading embedding model: {EMBED_MODEL} (CPU)")
    embedder = SentenceTransformer(EMBED_MODEL, device="cpu")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    manifest = load_manifest() if incremental else {}

    if not incremental:
        # Full rebuild: delete existing collections
        for name in ["medical", "code", "chimere"]:
            try:
                client.delete_collection(name)
            except Exception:
                pass

    collections = {}
    for name in ["medical", "code", "chimere"]:
        collections[name] = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    # Find all markdown files
    md_files = sorted(KNOWLEDGE_DIR.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files")

    total_chunks = 0
    skipped = 0
    errors = 0
    new_manifest = {}

    for i, fpath in enumerate(md_files):
        file_key = make_file_key(fpath)
        mtime = fpath.stat().st_mtime
        collection_name = get_collection_for_file(fpath)

        # Incremental: skip unchanged files
        if incremental and file_key in manifest:
            prev = manifest[file_key]
            if prev.get("mtime") == mtime:
                new_manifest[file_key] = prev
                skipped += 1
                continue
            else:
                # File changed: delete old chunks
                try:
                    old_ids = [f"{file_key}_{j:04d}" for j in range(prev.get("chunks", 100))]
                    collections[prev.get("collection", collection_name)].delete(ids=old_ids)
                except Exception:
                    pass

        try:
            text = fpath.read_text(encoding="utf-8")
            meta = extract_metadata(text, fpath)
            chunks = chunk_file(text, file_key, meta)

            if not chunks:
                skipped += 1
                continue

            coll = collections[collection_name]

            # Embed and add in batches
            for b in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[b:b + BATCH_SIZE]
                texts = [c["text"] for c in batch]
                ids = [c["id"] for c in batch]
                metas = [c["metadata"] for c in batch]
                embeddings = embedder.encode(texts, normalize_embeddings=True).tolist()
                coll.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metas)

            total_chunks += len(chunks)
            new_manifest[file_key] = {
                "mtime": mtime,
                "chunks": len(chunks),
                "collection": collection_name,
            }

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(md_files)}] {total_chunks} chunks indexed...")

        except Exception as e:
            print(f"  ERROR {fpath.name}: {e}", file=sys.stderr)
            errors += 1

    # ── Index Chimère config docs into "chimere" collection ──
    oc_coll = collections["chimere"]
    oc_files = []
    for doc_dir in CHIMERE_DOC_DIRS:
        if doc_dir.exists():
            oc_files.extend(sorted(doc_dir.glob("*.md")))

    for fpath in oc_files:
        if fpath.name in CHIMERE_SKIP_PATTERNS:
            continue
        file_key = "oc_" + fpath.stem.replace("-", "_").replace(" ", "_")
        mtime = fpath.stat().st_mtime

        if incremental and file_key in manifest:
            prev = manifest[file_key]
            if prev.get("mtime") == mtime:
                new_manifest[file_key] = prev
                skipped += 1
                continue
            else:
                try:
                    old_ids = [f"{file_key}_{j:04d}" for j in range(prev.get("chunks", 50))]
                    oc_coll.delete(ids=old_ids)
                except Exception:
                    pass

        try:
            text = fpath.read_text(encoding="utf-8")
            meta = {"file_path": str(fpath), "title": fpath.stem, "source": "chimere-config"}
            chunks = chunk_file(text, file_key, meta)
            if not chunks:
                continue
            for b in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[b:b + BATCH_SIZE]
                embeddings = embedder.encode(
                    [c["text"] for c in batch], normalize_embeddings=True
                ).tolist()
                oc_coll.add(
                    ids=[c["id"] for c in batch],
                    embeddings=embeddings,
                    documents=[c["text"] for c in batch],
                    metadatas=[c["metadata"] for c in batch],
                )
            total_chunks += len(chunks)
            new_manifest[file_key] = {"mtime": mtime, "chunks": len(chunks), "collection": "chimere"}
        except Exception as e:
            print(f"  ERROR chimere {fpath.name}: {e}", file=sys.stderr)
            errors += 1

    save_manifest(new_manifest)

    print(f"\nDone:")
    print(f"  Files processed: {len(md_files) - skipped}")
    print(f"  Files skipped: {skipped}")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Errors: {errors}")

    for name, coll in collections.items():
        count = coll.count()
        print(f"  Collection '{name}': {count} chunks")

    # Quick test
    print(f"\nTest query: 'tendinopathie coiffe rotateurs'")
    query_emb = embedder.encode(
        ["tendinopathie coiffe rotateurs"],
        normalize_embeddings=True
    ).tolist()
    results = collections["medical"].query(query_embeddings=query_emb, n_results=3)
    if results["documents"] and results["documents"][0]:
        for j, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            score = 1.0 - dist
            print(f"  [{j+1}] score={score:.3f} | {meta.get('title', '?')[:50]}")
            print(f"       {doc[:100]}...")


def show_stats():
    if not CHROMA_DIR.exists():
        print("No ChromaDB index found. Run without --stats first.")
        return

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    for name in ["medical", "code", "chimere"]:
        try:
            coll = client.get_collection(name)
            print(f"Collection '{name}': {coll.count()} chunks")
        except Exception:
            print(f"Collection '{name}': not found")

    manifest = load_manifest()
    print(f"Manifest: {len(manifest)} files tracked")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index knowledge files into ChromaDB")
    parser.add_argument("--incremental", action="store_true", help="Only index new/modified files")
    parser.add_argument("--stats", action="store_true", help="Show collection statistics")
    args = parser.parse_args()

    if args.stats:
        show_stats()
    else:
        t0 = time.time()
        build_index(incremental=args.incremental)
        print(f"\nCompleted in {time.time() - t0:.0f}s")
