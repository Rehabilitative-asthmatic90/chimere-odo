#!/usr/bin/env python3
"""
OCR via GLM-OCR (0.9B, 94.62% OmniDocBench) on GPU.

Swap workflow:
  1. Stop qwen35-custom (free GPU)
  2. Load GLM-OCR FP16 (~2.1 GB VRAM)
  3. OCR all pages
  4. Unload GLM-OCR
  5. Restart qwen35-custom

Usage:
  ocr_glm.py image.png                     # single image
  ocr_glm.py document.pdf                  # full PDF
  ocr_glm.py document.pdf --pages 1-10     # page range
  ocr_glm.py document.pdf --pages 3,5,7    # specific pages
  ocr_glm.py document.pdf --engram kine    # OCR + ingest into Engram table
  ocr_glm.py document.pdf --json           # output JSON instead of text
  ocr_glm.py document.pdf --no-swap        # don't swap models (GLM-OCR already loaded)

Output: stdout (text) or JSON. Also writes to /tmp/ocr_glm_output.txt
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

VENV_PYTHON = str(Path.home() / ".chimere/venvs/paddleocr/bin/python3")
MODEL_ID = "zai-org/GLM-OCR"
DPI = 150
MAX_TOKENS = 8192


def parse_pages(spec: str, total: int) -> list[int]:
    """Parse page spec like '1-10', '3,5,7', or '1-5,8,10-12'."""
    pages = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            pages.extend(range(int(a), min(int(b) + 1, total + 1)))
        else:
            p = int(part)
            if 1 <= p <= total:
                pages.append(p)
    return sorted(set(pages))


def extract_pages_as_png(pdf_path: str, pages: list[int], tmpdir: str) -> list[str]:
    """Extract PDF pages as PNG images."""
    sys.path.insert(0, str(Path.home() / ".local/share/pipx/venvs/pymupdf/lib/python3.12/site-packages"))
    import pymupdf
    doc = pymupdf.open(pdf_path)
    paths = []
    for p in pages:
        pix = doc[p - 1].get_pixmap(dpi=DPI)
        out = os.path.join(tmpdir, f"page_{p:04d}.png")
        pix.save(out)
        paths.append(out)
    doc.close()
    return paths


def swap_stop():
    """Stop Qwen3.5 to free GPU."""
    subprocess.run(
        ["systemctl", "--user", "stop", "qwen35-custom.service", "qwen-watchdog.service"],
        capture_output=True
    )
    time.sleep(3)


def swap_start():
    """Restart Qwen3.5."""
    subprocess.run(
        ["systemctl", "--user", "start", "qwen35-custom.service", "qwen-watchdog.service"],
        capture_output=True
    )


def ocr_images(image_paths: list[str], page_nums: list[int]) -> dict:
    """Run GLM-OCR on images. Returns {page_num: text}."""
    import torch
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="cuda"
    )
    load_time = time.time() - t0
    vram = torch.cuda.memory_allocated() // 1024 // 1024
    print(f"[glm-ocr] loaded in {load_time:.1f}s, VRAM: {vram} MB", file=sys.stderr)

    results = {}
    total_chars = 0
    total_time = 0

    for img_path, pn in zip(image_paths, page_nums):
        image = Image.open(img_path).convert("RGB")

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Text Recognition:"}
        ]}]

        t1 = time.time()
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device)
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False)

        text = processor.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        gen_time = time.time() - t1

        results[pn] = text.strip()
        total_chars += len(text)
        total_time += gen_time
        print(f"[glm-ocr] page {pn}: {gen_time:.1f}s, {len(text)} chars", file=sys.stderr)

    # Free GPU
    del model, processor
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[glm-ocr] total: {total_time:.1f}s, {total_chars} chars, {len(results)} pages",
          file=sys.stderr)
    return results


def main():
    parser = argparse.ArgumentParser(description="GLM-OCR: SOTA document OCR (0.9B, GPU swap)")
    parser.add_argument("input", help="Image or PDF file")
    parser.add_argument("--pages", default=None, help="Page range: 1-10, 3,5,7")
    parser.add_argument("--engram", default=None, help="Engram table name for ingestion")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--no-swap", action="store_true", help="Don't swap models")
    parser.add_argument("--output", "-o", default=None, help="Output file (default: stdout)")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    is_pdf = input_path.lower().endswith(".pdf")
    tmpdir = "/tmp/ocr_glm_work"
    os.makedirs(tmpdir, exist_ok=True)

    # Prepare images
    if is_pdf:
        sys.path.insert(0, str(Path.home() / ".local/share/pipx/venvs/pymupdf/lib/python3.12/site-packages"))
        import pymupdf
        doc = pymupdf.open(input_path)
        total_pages = len(doc)
        doc.close()

        if args.pages:
            page_nums = parse_pages(args.pages, total_pages)
        else:
            page_nums = list(range(1, total_pages + 1))

        print(f"[glm-ocr] PDF: {total_pages} pages, processing {len(page_nums)}", file=sys.stderr)
        image_paths = extract_pages_as_png(input_path, page_nums, tmpdir)
    else:
        page_nums = [1]
        image_paths = [input_path]

    # Swap if needed
    if not args.no_swap:
        print("[glm-ocr] stopping Qwen3.5...", file=sys.stderr)
        swap_stop()

    # OCR
    results = ocr_images(image_paths, page_nums)

    # Swap back
    if not args.no_swap:
        print("[glm-ocr] restarting Qwen3.5...", file=sys.stderr)
        swap_start()

    # Output
    if args.json:
        output = json.dumps({"pages": {str(k): v for k, v in results.items()},
                             "source": input_path, "model": MODEL_ID}, ensure_ascii=False, indent=2)
    else:
        output = "\n\n".join(f"--- Page {p} ---\n{t}" for p, t in sorted(results.items()))

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"[glm-ocr] saved to {args.output}", file=sys.stderr)
    else:
        print(output)

    # Also save to /tmp
    with open("/tmp/ocr_glm_output.txt", "w") as f:
        f.write(output)

    # Engram ingestion
    if args.engram:
        engram_script = str(Path.home() / ".chimere/bin/engram_ingest.py")
        if os.path.exists(engram_script):
            text_file = "/tmp/ocr_glm_for_engram.txt"
            with open(text_file, "w") as f:
                for t in results.values():
                    f.write(t + "\n")
            subprocess.run([
                sys.executable, engram_script,
                "--input", text_file,
                "--table", args.engram,
            ])
            print(f"[glm-ocr] ingested into Engram table '{args.engram}'", file=sys.stderr)


if __name__ == "__main__":
    main()
