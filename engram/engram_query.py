#!/usr/bin/env python3
"""Engram Query Tool — Query .engr n-gram hash tables.

Reads a binary .engr file (built by engram_ingest.py or the Rust EngramLookup::build),
tokenizes a query string, and shows hit/miss per n-gram, top predictions, and
coverage statistics.

Usage:
    python3 engram_query.py --table table.engr --query "the patient has chronic lower back pain"
    python3 engram_query.py --table table.engr --query "def fibonacci(n):" --top-k 10
    python3 engram_query.py --table table.engr --stats
"""

import argparse
import os
import struct
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants (must match engram_lookup.rs and engram_ingest.py)
# ---------------------------------------------------------------------------

MAGIC = 0x454E4752       # ASCII "ENGR"
VERSION = 1
HEADER_SIZE = 20         # 5 x u32
SLOT_SIZE = 16           # u64 + u32 + u32
EMPTY_HASH = 0

FNV_OFFSET_BASIS = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK_64 = 0xFFFFFFFFFFFFFFFF

# ---------------------------------------------------------------------------
# FNV-1a hash (must match Rust exactly)
# ---------------------------------------------------------------------------

def fnv1a_hash(tokens):
    """FNV-1a hash over a sequence of token IDs (u32, little-endian bytes)."""
    h = FNV_OFFSET_BASIS
    for t in tokens:
        for byte in t.to_bytes(4, 'little'):
            h ^= byte
            h = (h * FNV_PRIME) & MASK_64
    return h


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
            return Tokenizer.from_file(path)

    return Tokenizer.from_pretrained("Qwen/Qwen3.5-35B-A3B")


# ---------------------------------------------------------------------------
# EngramTable — Python reader for .engr files
# ---------------------------------------------------------------------------

class EngramTable:
    """Read-only accessor for .engr binary files.

    Matches the Rust EngramLookup memory layout exactly.
    """

    def __init__(self, path):
        with open(path, 'rb') as f:
            self.data = f.read()

        if len(self.data) < HEADER_SIZE:
            raise ValueError(f"File too small ({len(self.data)} bytes), need >= {HEADER_SIZE}")

        magic, version, order, table_size, num_entries = struct.unpack_from('<IIIII', self.data, 0)

        if magic != MAGIC:
            raise ValueError(f"Bad magic: 0x{magic:08X}, expected 0x{MAGIC:08X}")
        if version != VERSION:
            raise ValueError(f"Unsupported version: {version}, expected {VERSION}")

        self.order = order
        self.table_size = table_size
        self.num_entries = num_entries
        self.data_section_start = HEADER_SIZE + table_size * SLOT_SIZE

        min_size = HEADER_SIZE + table_size * SLOT_SIZE
        if len(self.data) < min_size:
            raise ValueError(
                f"File truncated: {len(self.data)} bytes, need >= {min_size}")

    def _read_slot(self, idx):
        """Read (hash, data_offset, total_count) for slot idx."""
        base = HEADER_SIZE + idx * SLOT_SIZE
        h = struct.unpack_from('<Q', self.data, base)[0]
        offset = struct.unpack_from('<I', self.data, base + 8)[0]
        count = struct.unpack_from('<I', self.data, base + 12)[0]
        return h, offset, count

    def lookup(self, context):
        """Look up next-token predictions for a context (list of token IDs).

        Uses the last `self.order` tokens as the key.
        Returns list of (token_id, probability) sorted by descending probability,
        or empty list if not found.
        """
        if len(context) < self.order:
            return []

        key = context[len(context) - self.order:]
        target_hash = fnv1a_hash(key)

        probe = target_hash % self.table_size
        start = probe

        while True:
            slot_hash, data_offset, _total_count = self._read_slot(probe)

            if slot_hash == EMPTY_HASH:
                return []

            if slot_hash == target_hash:
                return self._read_predictions(data_offset)

            probe = (probe + 1) % self.table_size
            if probe == start:
                return []

    def _read_predictions(self, data_offset):
        """Read next-token predictions from the data section."""
        base = self.data_section_start + data_offset

        if base + 4 > len(self.data):
            return []

        num_nexts = struct.unpack_from('<I', self.data, base)[0]
        if num_nexts == 0:
            return []

        needed = base + 4 + num_nexts * 8
        if needed > len(self.data):
            return []

        total_freq = 0
        entries = []

        for i in range(num_nexts):
            off = base + 4 + i * 8
            token = struct.unpack_from('<I', self.data, off)[0]
            freq = struct.unpack_from('<I', self.data, off + 4)[0]
            total_freq += freq
            entries.append((token, freq))

        if total_freq == 0:
            return []

        result = [(tok, freq / total_freq) for tok, freq in entries]
        result.sort(key=lambda x: -x[1])
        return result

    def stats(self):
        """Compute statistics about the hash table."""
        occupied = 0
        total_count = 0
        max_count = 0
        total_nexts = 0
        max_nexts = 0

        # Probe chain lengths
        chain_lengths = []

        for i in range(self.table_size):
            h, data_offset, count = self._read_slot(i)
            if h != EMPTY_HASH:
                occupied += 1
                total_count += count
                max_count = max(max_count, count)

                # Read num_nexts from data section
                base = self.data_section_start + data_offset
                if base + 4 <= len(self.data):
                    num_nexts = struct.unpack_from('<I', self.data, base)[0]
                    total_nexts += num_nexts
                    max_nexts = max(max_nexts, num_nexts)

        # Measure probe chain lengths (sample)
        max_chain = 0
        total_chain = 0
        chain_count = 0

        for i in range(self.table_size):
            h, _, _ = self._read_slot(i)
            if h != EMPTY_HASH:
                # Find where this hash would start probing
                ideal = h % self.table_size
                chain = (i - ideal) % self.table_size
                total_chain += chain
                max_chain = max(max_chain, chain)
                chain_count += 1

        avg_chain = total_chain / max(chain_count, 1)

        return {
            'order': self.order,
            'table_size': self.table_size,
            'num_entries': self.num_entries,
            'occupied_slots': occupied,
            'load_factor': occupied / self.table_size if self.table_size > 0 else 0,
            'total_frequency': total_count,
            'max_frequency': max_count,
            'total_next_tokens': total_nexts,
            'max_nexts_per_slot': max_nexts,
            'avg_nexts_per_slot': total_nexts / max(occupied, 1),
            'avg_probe_chain': avg_chain,
            'max_probe_chain': max_chain,
            'file_size': len(self.data),
            'data_section_size': len(self.data) - self.data_section_start,
        }


# ---------------------------------------------------------------------------
# Query display
# ---------------------------------------------------------------------------

def format_token(tokenizer, token_id):
    """Decode a single token ID to its string representation."""
    try:
        text = tokenizer.decode([token_id])
        # Show control chars as repr
        if not text or all(c in '\x00\x01\x02\x03' for c in text):
            return f"<{token_id}>"
        return text
    except Exception:
        return f"<{token_id}>"


def query_and_display(table, tokenizer, query_text, top_k=5):
    """Tokenize query, look up n-grams, display results."""
    encoding = tokenizer.encode(query_text, add_special_tokens=False)
    token_ids = list(encoding.ids)

    if not token_ids:
        print("Query produced no tokens.")
        return

    # Decode tokens for display
    token_strs = []
    for tid in token_ids:
        token_strs.append(format_token(tokenizer, tid))

    print(f"\nQuery: {query_text!r}")
    print(f"Tokens ({len(token_ids)}): {token_ids}")
    print(f"Decoded: {token_strs}")
    print(f"Table order: {table.order}")
    print()

    # Try each n-gram window
    hits = 0
    misses = 0
    total_windows = 0

    if len(token_ids) < table.order:
        print(f"Query too short ({len(token_ids)} tokens) for order-{table.order} table "
              f"(need >= {table.order} tokens)")
        return

    print(f"{'Window':>8}  {'Context':40}  {'Status':8}  {'Top Predictions'}")
    print("-" * 100)

    for i in range(len(token_ids) - table.order + 1):
        window = token_ids[i:i + table.order]
        # The context is the n-gram key; the "prediction" is what follows
        context_str = " ".join(format_token(tokenizer, t) for t in window)

        # Check if there is an actual next token in the query
        actual_next = None
        actual_next_str = ""
        if i + table.order < len(token_ids):
            actual_next = token_ids[i + table.order]
            actual_next_str = format_token(tokenizer, actual_next)

        predictions = table.lookup(window)
        total_windows += 1

        if predictions:
            hits += 1
            status = "HIT"

            # Show top-k predictions
            top = predictions[:top_k]
            pred_strs = []
            for tok, prob in top:
                tok_str = format_token(tokenizer, tok)
                marker = ""
                if actual_next is not None and tok == actual_next:
                    marker = " <-- MATCH"
                pred_strs.append(f"{tok_str!r} ({prob:.3f}){marker}")

            pred_display = ", ".join(pred_strs)
            if len(predictions) > top_k:
                pred_display += f" ... (+{len(predictions) - top_k} more)"
        else:
            misses += 1
            status = "MISS"
            pred_display = "-"

        # Truncate context string for display
        if len(context_str) > 38:
            context_str = context_str[:35] + "..."

        print(f"  [{i:>3}]   {context_str:40}  {status:8}  {pred_display}")

    # Summary
    print()
    print("=" * 60)
    print(f"  Coverage: {hits}/{total_windows} n-grams found "
          f"({hits / max(total_windows, 1) * 100:.1f}%)")
    print(f"  Hits: {hits}  |  Misses: {misses}")

    # Check if any actual next tokens were correctly predicted
    if len(token_ids) > table.order:
        correct_top1 = 0
        correct_top5 = 0
        predictable = 0

        for i in range(len(token_ids) - table.order):
            window = token_ids[i:i + table.order]
            actual_next = token_ids[i + table.order]
            predictions = table.lookup(window)

            if predictions:
                predictable += 1
                pred_tokens = [t for t, _ in predictions]
                if pred_tokens and pred_tokens[0] == actual_next:
                    correct_top1 += 1
                if actual_next in pred_tokens[:5]:
                    correct_top5 += 1

        if predictable > 0:
            print(f"  Top-1 accuracy: {correct_top1}/{predictable} "
                  f"({correct_top1 / predictable * 100:.1f}%)")
            print(f"  Top-5 accuracy: {correct_top5}/{predictable} "
                  f"({correct_top5 / predictable * 100:.1f}%)")

    print("=" * 60)


def display_stats(table):
    """Display table statistics."""
    s = table.stats()

    print()
    print("=" * 50)
    print("  ENGRAM TABLE STATISTICS")
    print("=" * 50)
    print(f"  Order:              {s['order']}")
    print(f"  Table size:         {s['table_size']:,} slots")
    print(f"  Occupied slots:     {s['occupied_slots']:,}")
    print(f"  Load factor:        {s['load_factor']:.2%}")
    print(f"  Total frequency:    {s['total_frequency']:,}")
    print(f"  Max frequency:      {s['max_frequency']:,}")
    print(f"  Total next-tokens:  {s['total_next_tokens']:,}")
    print(f"  Max nexts/slot:     {s['max_nexts_per_slot']}")
    print(f"  Avg nexts/slot:     {s['avg_nexts_per_slot']:.2f}")
    print(f"  Avg probe chain:    {s['avg_probe_chain']:.2f}")
    print(f"  Max probe chain:    {s['max_probe_chain']}")
    print(f"  File size:          {s['file_size']:,} bytes "
          f"({s['file_size'] / 1048576:.2f} MB)")
    print(f"  Data section:       {s['data_section_size']:,} bytes")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Engram Query Tool — Query .engr n-gram hash tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python3 engram_query.py --table table.engr --query "the patient has chronic lower back pain"
  python3 engram_query.py --table table.engr --query "def fibonacci(n):" --top-k 10
  python3 engram_query.py --table table.engr --stats
""")
    parser.add_argument('--table', '-t', required=True,
                        help='Path to .engr file')
    parser.add_argument('--query', '-q',
                        help='Query text to look up')
    parser.add_argument('--top-k', '-k', type=int, default=5,
                        help='Number of top predictions to show (default: 5)')
    parser.add_argument('--stats', '-s', action='store_true',
                        help='Show table statistics')
    parser.add_argument('--raw-tokens', nargs='+', type=int,
                        help='Query with raw token IDs instead of text')

    args = parser.parse_args()

    if not os.path.isfile(args.table):
        print(f"ERROR: Table file not found: {args.table}", file=sys.stderr)
        sys.exit(1)

    # Load table
    try:
        table = EngramTable(args.table)
    except Exception as e:
        print(f"ERROR: Failed to load table: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[ENGRAM] Loaded: {args.table} (order={table.order}, "
          f"{table.num_entries:,} entries, "
          f"{len(table.data) / 1048576:.2f} MB)", file=sys.stderr)

    if args.stats:
        display_stats(table)
        if not args.query and not args.raw_tokens:
            return

    if args.raw_tokens:
        # Query with raw token IDs (no tokenizer needed for the query itself,
        # but we still need it for decoding predictions)
        tokenizer = load_tokenizer()
        token_ids = args.raw_tokens

        print(f"\nRaw token query: {token_ids}")
        predictions = table.lookup(token_ids)
        if predictions:
            print(f"Found {len(predictions)} predictions:")
            for tok, prob in predictions[:args.top_k]:
                tok_str = format_token(tokenizer, tok)
                print(f"  {tok_str!r:20} (id={tok:>6}, p={prob:.4f})")
        else:
            print("No predictions found for this context.")
        return

    if args.query:
        tokenizer = load_tokenizer()
        query_and_display(table, tokenizer, args.query, args.top_k)
    elif not args.stats:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
