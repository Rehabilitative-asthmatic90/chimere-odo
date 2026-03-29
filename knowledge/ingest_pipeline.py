#!/usr/bin/env python3
"""
Knowledge Ingestion Pipeline — PAI Level 3

Ingests content from YouTube, Instagram, and web articles into structured
knowledge files. Uses local tools only: yt-dlp, Whisper.cpp, Cobalt,
and Qwen3.5-35B (vision + summarization) via llama-server at port 8081.

Usage:
    python3 ingest_pipeline.py "https://youtube.com/watch?v=xxx"
    python3 ingest_pipeline.py --url "https://..." --category cyber
    python3 ingest_pipeline.py --init
    python3 ingest_pipeline.py --list
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

# ── Paths ──

KNOWLEDGE_DIR = Path(os.path.expanduser("~/.chimere/workspaces/main/knowledge"))
INDEX_PATH = KNOWLEDGE_DIR / "index.json"
WHISPER_SCRIPT = os.path.expanduser("~/.chimere/bin/whisper_gpu.sh")
VENV_PYTHON = os.path.expanduser("~/.chimere/venvs/pipeline/bin/python")
LLM_URL = "http://127.0.0.1:8084/v1/chat/completions"
COBALT_URL = "http://localhost:9000/"
COOKIES_FILE = os.path.expanduser("~/.chimere/cookies/youtube.txt")
CHANNELS_YAML = KNOWLEDGE_DIR / "channels.yaml"

# Use venv python if available, else system
PYTHON = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable

# Ensure Deno is in PATH (required by yt-dlp for YouTube JS extraction)
DENO_BIN = os.path.expanduser("~/.deno/bin")
if os.path.isdir(DENO_BIN) and DENO_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = DENO_BIN + ":" + os.environ.get("PATH", "")


# ── Channel Config ──

def _load_channels() -> dict:
    """Load channels.yaml and build lookup tables."""
    if not CHANNELS_YAML.exists():
        return {}
    import yaml
    with open(CHANNELS_YAML) as f:
        config = yaml.safe_load(f)
    lookup = {}
    for ch in config.get("channels", []):
        cid = ch.get("channel_id", "")
        if cid:
            lookup[cid] = ch
        handle = ch.get("handle", "").lstrip("@").lower()
        if handle:
            lookup[handle] = ch
        name_lower = ch.get("name", "").lower()
        if name_lower:
            lookup[name_lower] = ch
        slug = ch.get("slug", "")
        if slug:
            lookup[slug] = ch
    return lookup

_CHANNEL_LOOKUP = _load_channels()


def _resolve_channel(channel_id: str = "", channel_name: str = "") -> Optional[dict]:
    """Find channel config by ID or name. Returns channel dict or None."""
    if channel_id and channel_id in _CHANNEL_LOOKUP:
        return _CHANNEL_LOOKUP[channel_id]
    if channel_name:
        name_lower = channel_name.lower()
        if name_lower in _CHANNEL_LOOKUP:
            return _CHANNEL_LOOKUP[name_lower]
        # Fuzzy: check if any key is contained in the name
        for key, ch in _CHANNEL_LOOKUP.items():
            if isinstance(key, str) and key in name_lower:
                return ch
    return None


def _get_ingested_urls() -> set:
    """Load already-ingested URLs from index.json."""
    if not INDEX_PATH.exists():
        return set()
    try:
        entries = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        return {e.get("url", "") for e in entries}
    except (json.JSONDecodeError, OSError):
        return set()


def _find_channel_by_slug(slug: str) -> Optional[dict]:
    """Find channel config by slug."""
    for key, ch in _CHANNEL_LOOKUP.items():
        if isinstance(ch, dict) and ch.get("slug") == slug:
            return ch
    return None


RSS_URL_TEMPLATE = "https://www.youtube.com/feeds/videos.xml?channel_id={}"


def channel_ingest(slug: str, max_videos: int = 3, dry_run: bool = False) -> list:
    """Ingest latest videos from a channel by slug. Returns list of result dicts."""
    import feedparser

    ch = _find_channel_by_slug(slug)
    if not ch:
        return [{"error": f"Channel '{slug}' not found in channels.yaml"}]

    channel_id = ch.get("channel_id", "")
    platform = ch.get("platform", "youtube")

    if platform != "youtube" or not channel_id:
        return [{"error": f"Channel '{slug}' has no YouTube channel_id (platform: {platform})"}]

    # Fetch RSS
    rss_url = RSS_URL_TEMPLATE.format(channel_id)
    feed = feedparser.parse(rss_url)
    if not feed.entries:
        return [{"error": f"No entries in RSS feed for {ch.get('name', slug)}"}]

    # Filter already-ingested
    ingested_urls = _get_ingested_urls()
    new_entries = []
    for entry in feed.entries:
        vid = entry.get("yt_videoid", "")
        if not vid:
            link = entry.get("link", "")
            if "v=" in link:
                vid = link.split("v=")[-1].split("&")[0]
        if vid:
            url = f"https://www.youtube.com/watch?v={vid}"
            if url not in ingested_urls:
                new_entries.append({"video_id": vid, "title": entry.get("title", ""), "url": url})

    if not new_entries:
        return [{"info": f"No new videos for {ch.get('name', slug)} — all already ingested"}]

    results = []
    for entry in new_entries[:max_videos]:
        print(f"[channel] {entry['title'][:60]}")
        if dry_run:
            results.append({"title": entry["title"], "url": entry["url"], "dry_run": True})
        else:
            print(f"[channel] Ingesting {entry['url']}...")
            result = ingest(entry["url"])
            results.append(result)

    return results


# ── URL Classification ──

def classify_url(url: str) -> tuple:
    """Classify URL into (source_type, parsed_url).
    Returns: ("youtube", url) | ("instagram", url) | ("web", url)
    """
    parsed = urlparse(url)
    host = parsed.hostname or ""
    host = host.lower().removeprefix("www.")

    if host in ("youtube.com", "youtu.be", "m.youtube.com"):
        return ("youtube", url)
    if host in ("instagram.com", "m.instagram.com"):
        return ("instagram", url)
    return ("web", url)


# ── Summarization ──

def llm_summarize(content: str, context: str = "", source_type: str = "article") -> str:
    """Summarize content using Qwen3.5-35B via llama-server HTTP API."""
    import requests

    # Truncate content to ~24000 chars (~6000 tokens) to fit in context
    max_chars = 24000
    if len(content) > max_chars:
        content = content[:max_chars] + "\n\n[... tronqué]"

    system_prompt = (
        "Tu es un assistant qui résume du contenu en français. "
        "Extrais les points clés, concepts importants et informations utiles. "
        "Format de réponse :\n"
        "1. Un titre clair\n"
        "2. Un résumé de 3-5 phrases\n"
        "3. Des bullet points avec les points clés\n"
        "Sois concis et factuel."
    )

    user_prompt = f"Résume ce contenu ({source_type})"
    if context:
        user_prompt += f" — contexte : {context}"
    user_prompt += f" :\n\n{content}"

    try:
        resp = requests.post(
            LLM_URL,
            json={
                "model": "qwen3.5-35b",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 2048,
                "temperature": 1.0,
                "top_p": 0.95,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[Erreur résumé Qwen3.5: {e}]"


def _parse_llm_summary(summary: str) -> tuple:
    """Parse LLM summary into (title, resume_text, bullet_points).
    Best-effort extraction — falls back gracefully.
    """
    lines = summary.strip().split("\n")
    title = ""
    resume_lines = []
    bullets = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Title: first line starting with # or first non-empty line
        if not title:
            title = stripped.lstrip("#").strip()
            continue
        if stripped.startswith(("- ", "* ", "• ")):
            bullets.append(stripped)
        else:
            resume_lines.append(stripped)

    resume_text = "\n".join(resume_lines) if resume_lines else summary
    return title, resume_text, bullets


# ── YouTube Handler ──

def youtube_handler(url: str) -> dict:
    """Download YouTube audio, transcribe with Whisper, summarize with Qwen3.5."""
    with tempfile.TemporaryDirectory(prefix="ingest_yt_") as tmpdir:
        info_path = os.path.join(tmpdir, "info.json")
        audio_path = os.path.join(tmpdir, "audio.mp3")

        # 1. Download audio + metadata with yt-dlp (Python API for reliable PATH/runtime detection)
        print("[youtube] Downloading audio...")
        import yt_dlp
        ydl_opts = {
            'format': 'bestaudio/best',
            'noplaylist': True,
            'outtmpl': os.path.join(tmpdir, 'audio.%(ext)s'),
            'writeinfojson': True,
            'quiet': True,
            'no_warnings': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
            }],
        }
        if os.path.exists(COOKIES_FILE):
            ydl_opts['cookiefile'] = COOKIES_FILE

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                metadata = ydl.extract_info(url, download=True)
        except Exception as e:
            return {"error": f"yt-dlp failed: {str(e)[:500]}"}

        title = metadata.get("title", "Sans titre")
        channel = metadata.get("channel", metadata.get("uploader", ""))
        duration = metadata.get("duration", 0)
        description = metadata.get("description", "")[:500]

        # Find the audio file (yt-dlp may name it differently)
        audio_files = list(Path(tmpdir).glob("audio.*"))
        audio_files = [f for f in audio_files if f.suffix != ".json"]
        if not audio_files:
            return {"error": "No audio file produced by yt-dlp"}
        audio_file = str(audio_files[0])

        # 2. Try youtube-transcript-api first (fast, uses existing subtitles)
        transcript = ""
        video_id = metadata.get("id", "")
        if video_id:
            try:
                from youtube_transcript_api import YouTubeTranscriptApi
                print("[youtube] Trying subtitle transcript...")
                fetched = YouTubeTranscriptApi.get_transcript(video_id, languages=["fr", "en"])
                transcript = " ".join(seg["text"] for seg in fetched)
                if transcript:
                    print(f"[youtube] Got subtitles ({len(transcript)} chars)")
            except Exception:
                pass  # No subtitles available, fall back to Whisper

        # 2b. Fall back to Whisper if no subtitles
        if not transcript:
            print("[youtube] Transcribing with Whisper...")
            try:
                whisper_result = subprocess.run(
                    ["bash", WHISPER_SCRIPT, audio_file],
                    capture_output=True, text=True, timeout=600,
                )
                transcript = whisper_result.stdout.strip()
            except subprocess.TimeoutExpired:
                transcript = "[Transcription timeout]"
            except Exception as e:
                transcript = f"[Transcription error: {e}]"

        if not transcript:
            transcript = "[Transcription vide]"

        # 3. Summarize with Qwen3.5
        print("[youtube] Summarizing...")
        context = f"Vidéo YouTube: {title}"
        if channel:
            context += f" par {channel}"
        if duration:
            context += f" ({duration // 60}min)"

        summary = llm_summarize(transcript, context=context, source_type="vidéo YouTube")

        # Resolve channel from channels.yaml
        channel_id = metadata.get("channel_id", "")
        ch_config = _resolve_channel(channel_id=channel_id, channel_name=channel)

        return {
            "title": title,
            "channel": channel,
            "channel_id": channel_id,
            "duration": duration,
            "description": description,
            "transcript": transcript,
            "summary": summary,
            "source_type": "YouTube",
            "_channel_config": ch_config,  # internal: for domain-first routing
        }


# ── Instagram Handler ──

def instagram_handler(url: str) -> dict:
    """Download Instagram content via Cobalt, OCR images, transcribe video, summarize."""
    import requests

    # 0. Extract caption via yt-dlp metadata (no download) — used as fallback if OCR/audio fails
    ig_caption = ""
    try:
        import yt_dlp
        ydl_opts = {
            "skip_download": True,
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            meta = ydl.extract_info(url, download=False)
        ig_caption = meta.get("description", "") or ""
        if ig_caption:
            print(f"[instagram] Got caption via yt-dlp ({len(ig_caption)} chars)")
    except Exception as e:
        print(f"[instagram] yt-dlp caption extraction failed (non-fatal): {e}")

    with tempfile.TemporaryDirectory(prefix="ingest_ig_") as tmpdir:
        # 1. Get download URL from Cobalt
        print("[instagram] Fetching via Cobalt...")
        try:
            resp = requests.post(
                COBALT_URL,
                json={"url": url},
                headers={"Accept": "application/json", "Content-Type": "application/json"},
                timeout=30,
            )
            data = resp.json()
        except Exception as e:
            return {"error": f"Cobalt request failed: {e}"}

        if data.get("status") == "error":
            return {"error": f"Cobalt error: {data.get('error', {}).get('code', 'unknown')}"}

        # Cobalt returns either a direct URL or a picker (multiple images)
        download_url = data.get("url")
        picker = data.get("picker", [])

        content_parts = []
        media_type = "image"

        if picker:
            # Multiple images (carousel)
            for i, item in enumerate(picker[:10]):  # max 10 items
                item_url = item.get("url", "")
                if not item_url:
                    continue
                ext = ".jpg" if "image" in item.get("type", "image") else ".mp4"
                media_path = os.path.join(tmpdir, f"media_{i}{ext}")
                try:
                    media_resp = requests.get(item_url, timeout=60)
                    with open(media_path, "wb") as f:
                        f.write(media_resp.content)
                    if ext == ".jpg":
                        text = _ocr_image(media_path)
                        if text:
                            content_parts.append(f"[Image {i+1}] {text}")
                    else:
                        media_type = "video"
                        text = _extract_video_text(media_path, tmpdir)
                        if text:
                            content_parts.append(f"[Video {i+1}] {text}")
                except Exception as e:
                    content_parts.append(f"[Media {i+1} error: {e}]")

        elif download_url:
            # Single media
            media_path = os.path.join(tmpdir, "media")
            try:
                media_resp = requests.get(download_url, timeout=120)
                content_type = media_resp.headers.get("content-type", "")
                if "video" in content_type:
                    media_type = "video"
                    media_path += ".mp4"
                    with open(media_path, "wb") as f:
                        f.write(media_resp.content)
                    text = _extract_video_text(media_path, tmpdir)
                    if text:
                        content_parts.append(text)
                else:
                    media_path += ".jpg"
                    with open(media_path, "wb") as f:
                        f.write(media_resp.content)
                    text = _ocr_image(media_path)
                    if text:
                        content_parts.append(text)
            except Exception as e:
                content_parts.append(f"[Download error: {e}]")
        else:
            return {"error": "Cobalt returned no download URL"}

        # Prepend caption to content_parts (as ig_batch.py does), or use as fallback
        if ig_caption:
            content_parts.insert(0, f"[Caption] {ig_caption}")

        content = "\n\n".join(content_parts) if content_parts else "[Aucun contenu extrait]"

        # Summarize
        print("[instagram] Summarizing...")
        summary = llm_summarize(content, context="Post Instagram", source_type="post Instagram")

        return {
            "title": f"Instagram {media_type}",
            "transcript": content,
            "summary": summary,
            "source_type": "Instagram",
            "media_type": media_type,
        }


def _ocr_image(image_path: str) -> str:
    """Extract text from an image using Qwen3.5-35B vision via llama-server."""
    import base64
    import requests

    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        # Detect mime type from extension
        ext = Path(image_path).suffix.lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        resp = requests.post(
            LLM_URL,
            json={
                "model": "qwen3.5-35b",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{img_b64}"},
                            },
                            {
                                "type": "text",
                                "text": "Extract all visible text from this image. Return only the extracted text, nothing else.",
                            },
                        ],
                    }
                ],
                "max_tokens": 1024,
                "temperature": 0.0,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        pass
    return ""


def _extract_video_text(video_path: str, tmpdir: str) -> str:
    """Extract audio from video with ffmpeg, then transcribe with Whisper."""
    audio_path = os.path.join(tmpdir, "extracted_audio.wav")
    try:
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-ar", "16000", "-ac", "1",
             "-c:a", "pcm_s16le", "-y", audio_path],
            capture_output=True, timeout=120,
        )
        if os.path.exists(audio_path):
            result = subprocess.run(
                ["bash", WHISPER_SCRIPT, audio_path],
                capture_output=True, text=True, timeout=600,
            )
            return result.stdout.strip()
    except Exception:
        pass
    return ""


# ── Web Handler ──

def web_handler(url: str) -> dict:
    """Fetch web article, extract text with readability, summarize with Qwen3.5."""
    import requests
    from readability import Document
    from bs4 import BeautifulSoup

    print("[web] Fetching article...")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
        }
        resp = requests.get(url, timeout=30, headers=headers)
        resp.raise_for_status()
    except Exception as e:
        return {"error": f"HTTP fetch failed: {e}"}

    # Extract readable content
    doc = Document(resp.text)
    title = doc.title() or "Sans titre"
    soup = BeautifulSoup(doc.summary(), "html.parser")
    text = soup.get_text(separator="\n", strip=True)

    if not text or len(text) < 50:
        # Fallback: extract all text from page
        soup_full = BeautifulSoup(resp.text, "html.parser")
        # Remove script/style
        for tag in soup_full(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup_full.get_text(separator="\n", strip=True)

    # Truncate raw content for storage
    raw_content = text[:8000]

    # Summarize
    print("[web] Summarizing...")
    summary = llm_summarize(text, context=f"Article: {title}", source_type="article web")

    return {
        "title": title,
        "transcript": raw_content,
        "summary": summary,
        "source_type": "Article web",
    }


# ── Knowledge Storage ──

def _slugify(text: str) -> str:
    """Convert text to filesystem-safe slug."""
    text = text.lower().strip()
    text = re.sub(r'[àáâã]', 'a', text)
    text = re.sub(r'[èéêë]', 'e', text)
    text = re.sub(r'[ìíîï]', 'i', text)
    text = re.sub(r'[òóôõ]', 'o', text)
    text = re.sub(r'[ùúûü]', 'u', text)
    text = re.sub(r'[ç]', 'c', text)
    text = re.sub(r'[^a-z0-9]+', '-', text)
    text = text.strip('-')
    return text[:60]


def _auto_category(summary: str) -> str:
    """Auto-detect category from summary content."""
    summary_lower = summary.lower()
    if any(w in summary_lower for w in ("cyber", "malware", "vulnerability", "cve", "exploit", "ransomware", "phishing")):
        return "cyber"
    if any(w in summary_lower for w in ("kinésithérap", "kinesitherapie", "rééducation", "musculo", "articulation")):
        return "kinesitherapie"
    if any(w in summary_lower for w in ("python", "javascript", "code", "programm", "developer", "api", "framework", "git")):
        return "dev"
    return "general"


def save_knowledge(result: dict, category: str, url: str, source_type_dir: str) -> Path:
    """Save knowledge file and update index. Uses domain-first layout if channel is known."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%Y-%m-%d %H:%M")

    title = result.get("title", "Sans titre")
    slug = _slugify(title)
    if not slug:
        slug = "untitled"

    # Domain-first routing: known channel → domain/channel-slug/
    ch_config = result.get("_channel_config")
    if ch_config:
        domain = ch_config.get("domain", "general")
        ch_slug = ch_config.get("slug", source_type_dir)
        subdir = KNOWLEDGE_DIR / domain / ch_slug
        category = domain  # override category with domain
    else:
        subdir = KNOWLEDGE_DIR / source_type_dir
    subdir.mkdir(parents=True, exist_ok=True)

    filename = f"{date_str}_{slug}.md"
    filepath = subdir / filename

    # Avoid collisions
    counter = 1
    while filepath.exists():
        counter += 1
        filename = f"{date_str}_{slug}_{counter}.md"
        filepath = subdir / filename

    # Parse summary
    llm_title, resume_text, bullets = _parse_llm_summary(result.get("summary", ""))
    display_title = llm_title or title

    # Build markdown
    bullet_text = "\n".join(bullets) if bullets else "- Voir résumé ci-dessus"
    raw_content = result.get("transcript", "")[:8000]
    source_type_label = result.get("source_type", source_type_dir.capitalize())

    md = f"""# {display_title}

- **Source** : {url}
- **Date d'ingestion** : {time_str}
- **Type** : {source_type_label}
- **Catégorie** : {category}

## Résumé

{resume_text}

## Points clés

{bullet_text}

## Contenu brut

<details>
<summary>Voir le contenu complet</summary>

{raw_content}

</details>
"""

    filepath.write_text(md, encoding="utf-8")
    print(f"[save] {filepath}")

    # Update index
    _update_index(url, display_title, category, date_str, str(filepath.relative_to(KNOWLEDGE_DIR)), source_type_dir)

    return filepath


def _update_index(url: str, title: str, category: str, date: str, path: str, source_type: str):
    """Append entry to index.json."""
    entries = []
    if INDEX_PATH.exists():
        try:
            entries = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            entries = []

    entries.append({
        "url": url,
        "title": title,
        "category": category,
        "date": date,
        "path": path,
        "source_type": source_type,
        "ingested_at": datetime.now().isoformat(),
    })

    INDEX_PATH.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ── Main Pipeline ──

def ingest(url: str, category: str = "") -> dict:
    """Main ingestion entry point. Returns result dict with summary."""
    source_type, url = classify_url(url)
    print(f"[ingest] URL classified as: {source_type}")

    handlers = {
        "youtube": youtube_handler,
        "instagram": instagram_handler,
        "web": web_handler,
    }

    handler = handlers[source_type]
    result = handler(url)

    if result.get("error"):
        print(f"[ingest] ERROR: {result['error']}", file=sys.stderr)
        return result

    # Auto-categorize if not specified
    if not category:
        category = _auto_category(result.get("summary", ""))

    # Save to knowledge base
    filepath = save_knowledge(result, category, url, source_type)

    result["filepath"] = str(filepath)
    result["category"] = category
    result["source_url"] = url
    return result


def init_dirs():
    """Create knowledge directory structure (domain-first + fallback platform dirs)."""
    # Platform fallback dirs for unknown channels
    for subdir in ("youtube", "instagram", "web"):
        (KNOWLEDGE_DIR / subdir).mkdir(parents=True, exist_ok=True)
    # Domain-first dirs from channels.yaml
    if CHANNELS_YAML.exists():
        import yaml
        with open(CHANNELS_YAML) as f:
            config = yaml.safe_load(f)
        for ch in config.get("channels", []):
            domain = ch.get("domain", "general")
            ch_slug = ch.get("slug", "")
            if domain and ch_slug:
                (KNOWLEDGE_DIR / domain / ch_slug).mkdir(parents=True, exist_ok=True)
    if not INDEX_PATH.exists():
        INDEX_PATH.write_text("[]\n", encoding="utf-8")
    print(f"[init] Knowledge directory ready: {KNOWLEDGE_DIR}")


def list_knowledge():
    """List ingested knowledge entries."""
    if not INDEX_PATH.exists():
        print("No index found. Run --init first.")
        return
    entries = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    if not entries:
        print("No knowledge entries yet.")
        return
    print(f"{'Date':<12} {'Type':<10} {'Category':<15} {'Title'}")
    print("-" * 70)
    for e in entries:
        print(f"{e.get('date', '?'):<12} {e.get('source_type', '?'):<10} {e.get('category', '?'):<15} {e.get('title', '?')[:40]}")
    print(f"\nTotal: {len(entries)} entries")


def format_summary(result: dict) -> str:
    """Format result for display (Telegram or CLI)."""
    if result.get("error"):
        return f"Erreur d'ingestion : {result['error']}"

    title = result.get("title", "Sans titre")
    category = result.get("category", "general")
    source_type = result.get("source_type", "?")
    summary = result.get("summary", "Pas de résumé")

    return (
        f"**{title}**\n"
        f"Type: {source_type} | Catégorie: {category}\n\n"
        f"{summary}"
    )


# ── CLI ──

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Ingestion Pipeline")
    parser.add_argument("url", nargs="?", help="URL to ingest")
    parser.add_argument("--url", dest="url_flag", help="URL to ingest (alternative)")
    parser.add_argument("--category", default="", help="Force category (cyber, dev, kinesitherapie, general)")
    parser.add_argument("--channel", default="", help="Ingest latest videos from channel by slug")
    parser.add_argument("--max", type=int, default=3, dest="max_videos", help="Max videos to ingest per channel (default: 3)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be ingested")
    parser.add_argument("--init", action="store_true", help="Create knowledge directory structure")
    parser.add_argument("--list", action="store_true", help="List ingested knowledge")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    if args.init:
        init_dirs()
        sys.exit(0)

    if args.list:
        list_knowledge()
        sys.exit(0)

    # Channel mode: ingest latest from a channel
    if args.channel:
        results = channel_ingest(args.channel, max_videos=args.max_videos, dry_run=args.dry_run)
        if args.json:
            output = []
            for r in results:
                cleaned = {k: v for k, v in r.items() if k not in ("transcript", "_channel_config")}
                if "transcript" in r:
                    cleaned["transcript_length"] = len(r.get("transcript", ""))
                output.append(cleaned)
            print(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            for r in results:
                if r.get("error"):
                    print(f"ERROR: {r['error']}")
                elif r.get("info"):
                    print(r["info"])
                elif r.get("dry_run"):
                    print(f"[dry-run] {r.get('title', '?')} — {r.get('url', '?')}")
                else:
                    print(f"\n{format_summary(r)}")
                    if r.get("filepath"):
                        print(f"Saved to: {r['filepath']}")
            ingested = [r for r in results if r.get("filepath")]
            errors = [r for r in results if r.get("error")]
            print(f"\nDone. Ingested: {len(ingested)}, Errors: {len(errors)}")
        sys.exit(0)

    target_url = args.url or args.url_flag
    if not target_url:
        parser.print_help()
        sys.exit(1)

    # Validate URL
    if not target_url.startswith(("http://", "https://")):
        print(f"Error: Invalid URL: {target_url}", file=sys.stderr)
        sys.exit(1)

    result = ingest(target_url, category=args.category)

    if args.json:
        # Remove large fields for JSON output
        output = {k: v for k, v in result.items() if k not in ("transcript", "_channel_config")}
        output["transcript_length"] = len(result.get("transcript", ""))
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print("\n" + format_summary(result))
        if result.get("filepath"):
            print(f"\nSaved to: {result['filepath']}")
