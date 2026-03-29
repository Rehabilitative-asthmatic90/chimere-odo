#!/usr/bin/env python3
"""Engram Ingestion Pipeline — Build .engr hash tables from text corpora.

Tokenizes documents with the Qwen3.5 tokenizer, extracts n-grams, and writes
a binary hash table compatible with the Rust EngramLookup (engram_lookup.rs).

Usage:
    python3 engram_ingest.py --input document.txt --output table.engr --order 5
    python3 engram_ingest.py --input corpus/ --output table.engr  # directory of files
    python3 engram_ingest.py --input doc.pdf --output table.engr  # PDF support

File format (must match Rust exactly):
    Header (20 bytes): magic(u32) version(u32) order(u32) table_size(u32) num_entries(u32)
    Hash Table [table_size x 16 bytes]: hash(u64) offset(u32) count(u32)
    Data Section: per slot: num_nexts(u32) then [token(u32) freq(u32)] pairs
"""

import argparse
import os
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants (must match engram_lookup.rs)
# ---------------------------------------------------------------------------

MAGIC = 0x454E4752       # ASCII "ENGR"
VERSION = 1
HEADER_SIZE = 20         # 5 x u32
SLOT_SIZE = 16           # u64 + u32 + u32
EMPTY_HASH = 0           # sentinel for empty slots

FNV_OFFSET_BASIS = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK_64 = 0xFFFFFFFFFFFFFFFF

# ---------------------------------------------------------------------------
# FNV-1a hash (must match Rust EngramLookup::hash_ngram exactly)
# ---------------------------------------------------------------------------

def fnv1a_hash(tokens):
    """FNV-1a hash over a sequence of token IDs (u32, little-endian bytes).

    Matches the Rust implementation in engram_lookup.rs byte-for-byte:
        for each token:
            for each byte of token.to_le_bytes():
                hash ^= byte
                hash = hash.wrapping_mul(FNV_PRIME)
    """
    h = FNV_OFFSET_BASIS
    for t in tokens:
        for byte in t.to_bytes(4, 'little'):
            h ^= byte
            h = (h * FNV_PRIME) & MASK_64
    return h


def next_power_of_two(n):
    """Smallest power of two >= n (minimum 1). Matches Rust."""
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


# ---------------------------------------------------------------------------
# Tokenizer loading
# ---------------------------------------------------------------------------

def load_tokenizer():
    """Load the Qwen3.5 tokenizer. Try local paths first, then HuggingFace."""
    from tokenizers import Tokenizer

    local_paths = [
        os.path.expanduser("~/.chimere/models/Qwen3.5-35B-A3B-BF16/tokenizer.json"),
        os.path.expanduser("~/.chimere/models/Qwen3.5-35B-A3B-GGUF/tokenizer.json"),
        os.path.expanduser("~/.chimere/models/qwopus-27b-bf16/tokenizer.json"),
    ]

    for path in local_paths:
        if os.path.isfile(path):
            print(f"[ENGRAM] Loading tokenizer from {path}", file=sys.stderr)
            return Tokenizer.from_file(path)

    # Fall back to HuggingFace
    print("[ENGRAM] No local tokenizer found, downloading from HuggingFace...", file=sys.stderr)
    return Tokenizer.from_pretrained("Qwen/Qwen3.5-35B-A3B")


# ---------------------------------------------------------------------------
# Document reading
# ---------------------------------------------------------------------------

def read_text_file(path):
    """Read a plain text or markdown file."""
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()


def read_pdf_file(path):
    """Read a PDF file using PyMuPDF (fitz). Returns extracted text."""
    try:
        import fitz  # pymupdf
    except ImportError:
        print(f"[ENGRAM] WARNING: pymupdf not installed, cannot read PDF: {path}", file=sys.stderr)
        print("[ENGRAM] Install with: pip install pymupdf", file=sys.stderr)
        return ""

    text_parts = []
    doc = fitz.open(path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        text_parts.append(page.get_text())
    doc.close()
    return "\n".join(text_parts)


def read_document(path):
    """Read a document and return its text content."""
    path = str(path)
    ext = os.path.splitext(path)[1].lower()

    if ext == '.pdf':
        return read_pdf_file(path)
    elif ext in ('.txt', '.md', '.rst', '.py', '.rs', '.c', '.cpp', '.h',
                 '.js', '.ts', '.json', '.jsonl', '.csv', '.xml', '.html',
                 '.yml', '.yaml', '.toml', '.cfg', '.ini', '.sh', '.bash',
                 '.log', '.tex'):
        return read_text_file(path)
    else:
        # Try as text
        try:
            return read_text_file(path)
        except UnicodeDecodeError:
            print(f"[ENGRAM] WARNING: Cannot read binary file: {path}", file=sys.stderr)
            return ""


def collect_input_files(input_path):
    """Collect all readable files from a path (file or directory)."""
    input_path = Path(input_path)

    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        files = []
        supported_exts = {
            '.txt', '.md', '.rst', '.py', '.rs', '.c', '.cpp', '.h',
            '.js', '.ts', '.json', '.jsonl', '.csv', '.xml', '.html',
            '.yml', '.yaml', '.toml', '.cfg', '.ini', '.sh', '.bash',
            '.log', '.tex', '.pdf',
        }
        for f in sorted(input_path.rglob('*')):
            if f.is_file() and f.suffix.lower() in supported_exts:
                files.append(f)
        return files

    print(f"[ENGRAM] ERROR: Input path does not exist: {input_path}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# N-gram extraction
# ---------------------------------------------------------------------------

def extract_ngrams(corpus, order):
    """Extract n-gram frequencies from a token corpus.

    Returns: dict mapping FNV-1a hash -> dict(next_token -> frequency)
    """
    if len(corpus) <= order:
        print(f"[ENGRAM] WARNING: Corpus too short ({len(corpus)} tokens) for order {order}",
              file=sys.stderr)
        return {}

    ngram_map = defaultdict(lambda: defaultdict(int))

    for i in range(len(corpus) - order):
        key_tokens = tuple(corpus[i:i + order])
        next_token = corpus[i + order]
        h = fnv1a_hash(key_tokens)
        ngram_map[h][next_token] += 1

    return ngram_map


# ---------------------------------------------------------------------------
# Binary file writer
# ---------------------------------------------------------------------------

def write_engram_file(ngram_map, order, output_path):
    """Write the .engr binary file matching the Rust EngramLookup format.

    Layout:
        Header (20 bytes)
        Hash Table [table_size x 16 bytes]
        Data Section [variable]
    """
    num_entries = len(ngram_map)
    if num_entries == 0:
        print("[ENGRAM] ERROR: No n-grams to write", file=sys.stderr)
        sys.exit(1)

    # Table size: next power of two >= 2 * num_entries, min 16
    table_size = next_power_of_two(max(num_entries * 2, 16))

    # Build data section and slot assignments
    slots = [(EMPTY_HASH, 0, 0)] * table_size  # (hash, data_offset, total_count)
    data_bytes = bytearray()

    for h, nexts_map in ngram_map.items():
        total_count = sum(nexts_map.values())
        data_offset = len(data_bytes)

        # Data entry: num_nexts(u32) then sorted (token u32, freq u32) pairs
        nexts_sorted = sorted(nexts_map.items(), key=lambda x: -x[1])
        num_nexts = len(nexts_sorted)

        data_bytes.extend(struct.pack('<I', num_nexts))
        for tok, freq in nexts_sorted:
            data_bytes.extend(struct.pack('<II', tok, freq))

        # Insert into hash table with linear probing
        probe = h % table_size
        while True:
            if slots[probe][0] == EMPTY_HASH:
                slots[probe] = (h, data_offset, total_count)
                break
            probe = (probe + 1) % table_size

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Write the file
    with open(output_path, 'wb') as f:
        # Header (20 bytes)
        f.write(struct.pack('<I', MAGIC))
        f.write(struct.pack('<I', VERSION))
        f.write(struct.pack('<I', order))
        f.write(struct.pack('<I', table_size))
        f.write(struct.pack('<I', num_entries))

        # Hash table [table_size x 16 bytes]
        for h, offset, count in slots:
            f.write(struct.pack('<Q', h))    # u64 hash
            f.write(struct.pack('<I', offset))  # u32 data offset
            f.write(struct.pack('<I', count))   # u32 total count

        # Data section
        f.write(data_bytes)

    return table_size, num_entries, len(data_bytes)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_engram_file(output_path, ngram_map, order):
    """Read back the .engr file and verify a sample of entries."""
    with open(output_path, 'rb') as f:
        data = f.read()

    # Parse header
    magic, version, file_order, table_size, num_entries = struct.unpack_from('<IIIII', data, 0)

    assert magic == MAGIC, f"Bad magic: 0x{magic:08X}"
    assert version == VERSION, f"Bad version: {version}"
    assert file_order == order, f"Order mismatch: {file_order} != {order}"
    assert num_entries == len(ngram_map), f"Entry count mismatch: {num_entries} != {len(ngram_map)}"

    data_section_start = HEADER_SIZE + table_size * SLOT_SIZE

    # Verify a sample of lookups
    verified = 0
    sample_hashes = list(ngram_map.keys())[:min(100, len(ngram_map))]

    for target_hash in sample_hashes:
        probe = target_hash % table_size
        found = False
        start_probe = probe

        while True:
            slot_offset = HEADER_SIZE + probe * SLOT_SIZE
            slot_hash = struct.unpack_from('<Q', data, slot_offset)[0]
            slot_data_offset = struct.unpack_from('<I', data, slot_offset + 8)[0]

            if slot_hash == EMPTY_HASH:
                break

            if slot_hash == target_hash:
                # Verify data section
                abs_offset = data_section_start + slot_data_offset
                num_nexts = struct.unpack_from('<I', data, abs_offset)[0]
                assert num_nexts == len(ngram_map[target_hash]), \
                    f"num_nexts mismatch for hash 0x{target_hash:016X}"
                found = True
                verified += 1
                break

            probe = (probe + 1) % table_size
            if probe == start_probe:
                break

        assert found, f"Hash 0x{target_hash:016X} not found in table"

    return verified


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Engram Ingestion — Build .engr n-gram hash tables from text corpora",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python3 engram_ingest.py --input document.txt --output table.engr --order 5
  python3 engram_ingest.py --input corpus/ --output table.engr
  python3 engram_ingest.py --input doc.pdf --output table.engr --order 3
""")
    parser.add_argument('--input', '-i', required=True,
                        help='Input file or directory of files (.txt, .md, .pdf, etc.)')
    parser.add_argument('--output', '-o', required=True,
                        help='Output .engr file path')
    parser.add_argument('--order', '-n', type=int, default=5,
                        help='N-gram order (default: 5)')
    parser.add_argument('--min-freq', type=int, default=1,
                        help='Minimum frequency to keep an n-gram (default: 1)')
    parser.add_argument('--max-nexts', type=int, default=64,
                        help='Max distinct next-tokens per n-gram (default: 64)')
    parser.add_argument('--verify', action='store_true', default=True,
                        help='Verify output file after writing (default: on)')
    parser.add_argument('--no-verify', action='store_true',
                        help='Skip verification')

    args = parser.parse_args()

    if args.order < 1:
        print("[ENGRAM] ERROR: order must be >= 1", file=sys.stderr)
        sys.exit(1)

    t_start = time.time()

    # Step 1: Collect input files
    files = collect_input_files(args.input)
    if not files:
        print(f"[ENGRAM] ERROR: No readable files found in {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"[ENGRAM] Found {len(files)} file(s) to process", file=sys.stderr)

    # Step 2: Load tokenizer
    tokenizer = load_tokenizer()

    # Step 3: Tokenize all documents
    corpus = []
    total_chars = 0

    for i, fpath in enumerate(files):
        text = read_document(fpath)
        if not text.strip():
            print(f"[ENGRAM]   Skipping empty file: {fpath}", file=sys.stderr)
            continue

        total_chars += len(text)
        encoding = tokenizer.encode(text, add_special_tokens=False)
        token_ids = encoding.ids
        corpus.extend(token_ids)

        if (i + 1) % 100 == 0 or (i + 1) == len(files):
            print(f"[ENGRAM]   Tokenized {i + 1}/{len(files)} files "
                  f"({len(corpus):,} tokens so far)", file=sys.stderr)

    t_tokenized = time.time()
    print(f"[ENGRAM] Tokenization: {total_chars:,} chars -> {len(corpus):,} tokens "
          f"in {t_tokenized - t_start:.1f}s", file=sys.stderr)

    if len(corpus) <= args.order:
        print(f"[ENGRAM] ERROR: Corpus too short ({len(corpus)} tokens) for order {args.order}",
              file=sys.stderr)
        sys.exit(1)

    # Step 4: Extract n-grams
    print(f"[ENGRAM] Extracting {args.order}-grams...", file=sys.stderr)
    ngram_map = extract_ngrams(corpus, args.order)

    # Apply frequency filter
    if args.min_freq > 1:
        before = len(ngram_map)
        filtered = {}
        for h, nexts in ngram_map.items():
            total = sum(nexts.values())
            if total >= args.min_freq:
                filtered[h] = nexts
        ngram_map = filtered
        print(f"[ENGRAM] Frequency filter (>= {args.min_freq}): "
              f"{before:,} -> {len(ngram_map):,} n-grams", file=sys.stderr)

    # Apply max-nexts pruning (keep top-k next tokens per n-gram)
    if args.max_nexts > 0:
        pruned_count = 0
        for h in ngram_map:
            nexts = ngram_map[h]
            if len(nexts) > args.max_nexts:
                top_nexts = sorted(nexts.items(), key=lambda x: -x[1])[:args.max_nexts]
                ngram_map[h] = dict(top_nexts)
                pruned_count += 1
        if pruned_count > 0:
            print(f"[ENGRAM] Pruned next-tokens in {pruned_count:,} n-grams "
                  f"(max {args.max_nexts})", file=sys.stderr)

    t_extracted = time.time()
    total_nexts = sum(len(v) for v in ngram_map.values())
    print(f"[ENGRAM] Extracted {len(ngram_map):,} unique {args.order}-grams "
          f"({total_nexts:,} next-token entries) in {t_extracted - t_tokenized:.1f}s",
          file=sys.stderr)

    # Step 5: Write binary file
    print(f"[ENGRAM] Writing {args.output}...", file=sys.stderr)
    table_size, num_entries, data_size = write_engram_file(ngram_map, args.order, args.output)

    file_size = os.path.getsize(args.output)
    t_written = time.time()

    print(f"[ENGRAM] Written: {file_size:,} bytes ({file_size / 1048576:.2f} MB)", file=sys.stderr)
    print(f"[ENGRAM]   table_size: {table_size:,} slots "
          f"(load factor: {num_entries / table_size:.2%})", file=sys.stderr)
    print(f"[ENGRAM]   num_entries: {num_entries:,}", file=sys.stderr)
    print(f"[ENGRAM]   data_section: {data_size:,} bytes", file=sys.stderr)

    # Step 6: Verify
    if args.verify and not args.no_verify:
        print("[ENGRAM] Verifying...", file=sys.stderr)
        verified = verify_engram_file(args.output, ngram_map, args.order)
        print(f"[ENGRAM] Verified {verified} lookups OK", file=sys.stderr)

    t_end = time.time()
    print(f"[ENGRAM] Done in {t_end - t_start:.1f}s total", file=sys.stderr)
    print(f"[ENGRAM] Output: {os.path.abspath(args.output)}", file=sys.stderr)


if __name__ == '__main__':
    main()
