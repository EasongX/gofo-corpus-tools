"""Ingest a Feishu doc into a corpus repo.

Pipeline:
  fetch (kb-bot) → download embedded images → BGE dedup vs existing corpus
  → Claude generates frontmatter + dedup judgment → write file → git commit.

Usage:
  corpus-ingest <feishu_url> --uploaded-by "宋宜烜"
  corpus-ingest <feishu_url> --repo ~/gofo_hr_corpus --target-dir hr

If --repo not given, walks up from cwd looking for a dir containing
knowledge/ and bot_configs/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path

DEFAULT_TARGET_DIR = "ops"  # overridden per-repo via --target-dir / Claude
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
VALID_LEVELS = {"public", "internal", "confidential"}

TITLE_RE = re.compile(r"<title>([^<]+)</title>")
IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

# Set once in main() so module-level functions can reference them. Avoids
# threading repo through every helper.
REPO: Path = Path()
KNOWLEDGE: Path = Path()
DATA: Path = Path()


@dataclass
class Chunk:
    source: str
    text: str


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from cwd looking for a corpus repo (has knowledge/ + bot_configs/)."""
    p = (start or Path.cwd()).resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "knowledge").is_dir() and (candidate / "bot_configs").is_dir():
            return candidate
    sys.exit(
        "not inside a corpus repo (looking for a dir with knowledge/ and bot_configs/). "
        "pass --repo PATH explicitly."
    )


def _split_text(text: str, *, max_chars: int = 600, overlap: int = 100) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    out: list[str] = []
    current = ""
    for para in paragraphs:
        if len(para) > max_chars:
            if current:
                out.append(current.strip())
                current = ""
            for i in range(0, len(para), max_chars - overlap):
                out.append(para[i : i + max_chars].strip())
            continue
        projected = (current + "\n\n" + para) if current else para
        if len(projected) > max_chars and current:
            out.append(current.strip())
            tail = current[-overlap:] if overlap and len(current) > overlap else ""
            current = (tail + "\n\n" + para).strip() if tail else para
        else:
            current = projected
    if current.strip():
        out.append(current.strip())
    return [c for c in out if c]


def load_corpus_chunks() -> list[Chunk]:
    out: list[Chunk] = []
    for path in sorted(KNOWLEDGE.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        if text.startswith("---"):
            end = text.find("\n---", 3)
            if end != -1:
                text = text[end + 4 :]
        rel = str(path.relative_to(KNOWLEDGE))
        for piece in _split_text(text):
            out.append(Chunk(source=rel, text=piece))
    return out


def fetch_doc(url: str) -> dict:
    cmd = [
        "lark-cli", "--profile", "kb-bot", "docs", "+fetch",
        "--api-version", "v2", "--doc", url, "--as", "bot",
        "--doc-format", "markdown",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        sys.exit(f"lark-cli fetch failed: {r.stderr or r.stdout}")
    data = json.loads(r.stdout)
    if not data.get("ok"):
        sys.exit(f"fetch error: {data.get('error')}")
    return data["data"]["document"]


def extract_title(content: str) -> str:
    m = TITLE_RE.search(content)
    return m.group(1).strip() if m else "untitled"


def slugify(title: str) -> str:
    s = re.sub(r"[\s\\/]+", "-", title.strip())
    s = re.sub(r"[<>:\"|?*]", "", s)
    return s[:80] or "untitled"


def download_image(url: str, out_dir: Path, index: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1(url.encode()).hexdigest()[:10]
    name = f"img-{index:03d}-{h}.png"
    dst = out_dir / name
    if dst.exists():
        return dst
    req = urllib.request.Request(url, headers={"User-Agent": "gofo-corpus-tools/ingest"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        ct = resp.headers.get("Content-Type", "")
        ext = ".png"
        if "jpeg" in ct or "jpg" in ct:
            ext = ".jpg"
        elif "gif" in ct:
            ext = ".gif"
        elif "webp" in ct:
            ext = ".webp"
        if ext != ".png":
            dst = dst.with_suffix(ext)
        dst.write_bytes(resp.read())
    return dst


def process_content(content: str, media_dir: Path, md_dir: Path) -> tuple[str, int]:
    body = TITLE_RE.sub("", content, count=1).lstrip()
    counter = {"i": 0}

    def repl(m: re.Match[str]) -> str:
        alt = m.group(1)
        url = m.group(2)
        if url.startswith("http"):
            counter["i"] += 1
            try:
                local = download_image(url, media_dir, counter["i"])
                rel = os.path.relpath(local, md_dir)
                return f"![{alt}]({rel})"
            except Exception as e:
                return f"![{alt} (image download failed: {e})]({url})"
        return m.group(0)

    body = IMG_RE.sub(repl, body)
    return body, counter["i"]


def embed_dedup_check(new_text: str) -> dict:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    chunks = load_corpus_chunks()
    if not chunks:
        return {"max_score": 0.0, "top_hits": []}

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    new_pieces = _split_text(new_text)
    new_emb = model.encode(new_pieces, normalize_embeddings=True).astype("float32")
    old_emb = model.encode([c.text for c in chunks], normalize_embeddings=True).astype("float32")
    sims = new_emb @ old_emb.T
    flat = sims.flatten()
    top_idx = (-flat).argsort()[:5]
    hits = []
    for ix in top_idx:
        nc, oc = int(ix // sims.shape[1]), int(ix % sims.shape[1])
        score = float(sims[nc, oc])
        if score < 0.4:
            continue
        hits.append((chunks[oc].source, score, chunks[oc].text[:200]))
    return {"max_score": float(sims.max()), "top_hits": hits}


def collect_existing_tags() -> list[str]:
    tags: set[str] = set()
    try:
        import yaml
    except ImportError:
        return []
    for path in KNOWLEDGE.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---"):
            continue
        end = text.find("\n---", 3)
        if end == -1:
            continue
        try:
            meta = yaml.safe_load(text[3:end]) or {}
        except yaml.YAMLError:
            continue
        for t in meta.get("tags") or []:
            if isinstance(t, str):
                tags.add(t)
    return sorted(tags)


def claude_judge_and_frontmatter(
    *,
    title_hint: str,
    body: str,
    source_url: str,
    uploaded_by: str,
    dedup: dict,
    level_override: str | None,
    target_subdirs_hint: list[str],
) -> dict:
    from anthropic import Anthropic

    api_key = os.environ.get("AGENT_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("missing AGENT_ANTHROPIC_API_KEY (see ~/.claude/secrets/anthropic.env)")
    client = Anthropic(api_key=api_key)

    hits_text = "\n\n".join(
        f"[{src}] score={score:.2f}\n{text}" for src, score, text in dedup["top_hits"]
    ) or "(no relevant existing chunks above 0.4)"

    BODY_BUDGET = 8000
    if len(body) <= BODY_BUDGET:
        body_excerpt = body
    else:
        half = BODY_BUDGET // 2
        body_excerpt = body[:half] + "\n\n...[middle truncated]...\n\n" + body[-half:]

    existing_tags = collect_existing_tags()
    known_tags_str = ", ".join(existing_tags) if existing_tags else "(none yet)"
    level_directive = (
        f"Use level={level_override!r} (operator override; do not reclassify)."
        if level_override else
        "Classify level yourself: public (no internal context required), "
        "internal (default — team / operational content), "
        "confidential (PII, credentials, salary, legal, security incidents)."
    )
    target_options = " | ".join(repr(s) for s in target_subdirs_hint) or "'shared' | 'ops'"

    prompt = f"""You analyze a Lark/Feishu doc that a teammate just submitted to a team
knowledge base and produce metadata + a duplicate/conflict judgment.

NEW DOC (title hint: {title_hint!r}):
---
{body_excerpt}
---

TOP RELEVANT EXISTING CHUNKS (max similarity: {dedup['max_score']:.2f}):
---
{hits_text}
---

Return JSON only, with these fields:
- title: clean title in source language
- tags: 3-6 lowercase kebab-case tags. **Prefer reusing existing tags below.**
        Only invent a new tag when no existing one fits.
        Existing tags (alphabetical): {known_tags_str}
- summary: 1-3 sentence summary in the source language
- key_points: 3-5 bullet strings, no leading dashes
- target_subdir: one of {target_options}
- level: "public" | "internal" | "confidential". {level_directive}
- dedup_decision: exactly one of "new" | "update_of" | "conflict_with"
- dedup_target: if decision is update_of/conflict_with, the existing source filename from the hits; else null
- dedup_reason: 1 sentence explaining the decision

Rules:
- "update_of" when same topic, new info supersedes/extends
- "conflict_with" when same fact stated differently (a real contradiction)
- "new" when the topic is genuinely new despite surface similarity
- Be conservative: only flag "conflict_with" if there's a real factual disagreement.
- Don't invent a Chinese tag if an English equivalent already exists in the existing list.
"""

    resp = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        sys.exit(f"claude returned no JSON: {text[:500]}")
    return json.loads(m.group(0))


def list_target_subdirs() -> list[str]:
    """Existing first-level + 'foo/bar' two-level subdirs of knowledge/."""
    out: list[str] = []
    for child in sorted(KNOWLEDGE.iterdir()):
        if not child.is_dir() or child.name.startswith((".", "_")):
            continue
        out.append(child.name)
        for sub in sorted(child.iterdir()):
            if sub.is_dir() and not sub.name.startswith((".", "_")):
                out.append(f"{child.name}/{sub.name}")
    return out


def build_frontmatter(meta: dict, source_url: str, source_doc_id: str, uploaded_by: str) -> str:
    today = date.today().isoformat()
    tags_str = "[" + ", ".join(meta["tags"]) + "]"
    kp = "\n".join(f"  - {p}" for p in meta["key_points"])
    level = meta.get("level") if meta.get("level") in VALID_LEVELS else "internal"
    return (
        "---\n"
        f"title: {meta['title']}\n"
        f"tags: {tags_str}\n"
        f"level: {level}\n"
        f"source: {source_url}\n"
        f"source_doc_id: {source_doc_id}\n"
        f"learned_date: {today}\n"
        f"uploaded_by: {uploaded_by}\n"
        f"summary: {json.dumps(meta['summary'], ensure_ascii=False)}\n"
        f"key_points:\n{kp}\n"
        "---\n\n"
    )


def git_commit(repo: Path, files: list[Path], message: str) -> None:
    subprocess.run(["git", "add"] + [str(f) for f in files], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=repo, check=True)


def main() -> None:
    global REPO, KNOWLEDGE, DATA

    ap = argparse.ArgumentParser(prog="corpus-ingest")
    ap.add_argument("url")
    ap.add_argument("--repo", type=Path, default=None,
                    help="Corpus repo root. Default: walk up from cwd.")
    ap.add_argument("--uploaded-by", default="unknown")
    ap.add_argument("--target-dir", default=None,
                    help="Override Claude's choice of target subdir")
    ap.add_argument("--level", choices=sorted(VALID_LEVELS), default=None,
                    help="Force sensitivity level; otherwise Claude classifies.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    REPO = (args.repo.resolve() if args.repo else find_repo_root())
    KNOWLEDGE = REPO / "knowledge"
    DATA = REPO / "data"
    if not KNOWLEDGE.is_dir():
        sys.exit(f"no knowledge/ found in {REPO}")
    DATA.mkdir(exist_ok=True)

    print(f"→ Repo: {REPO}", file=sys.stderr)
    print(f"→ Fetching {args.url}", file=sys.stderr)
    doc = fetch_doc(args.url)
    raw = doc["content"]
    doc_id = doc["document_id"]
    title_hint = extract_title(raw)
    slug = slugify(title_hint)

    print(f"→ Title: {title_hint!r}; slug={slug}", file=sys.stderr)

    print("→ Embedding + duplicate check (loading BGE model, ~5s)...", file=sys.stderr)
    body_only = TITLE_RE.sub("", raw, count=1).lstrip()
    dedup = embed_dedup_check(body_only)
    print(f"  max similarity: {dedup['max_score']:.3f}", file=sys.stderr)
    for src, score, _ in dedup["top_hits"]:
        print(f"  hit: {src} ({score:.2f})", file=sys.stderr)

    print("→ Asking Claude for metadata + dedup judgment...", file=sys.stderr)
    meta = claude_judge_and_frontmatter(
        title_hint=title_hint,
        body=body_only,
        source_url=args.url,
        uploaded_by=args.uploaded_by,
        dedup=dedup,
        level_override=args.level,
        target_subdirs_hint=list_target_subdirs(),
    )
    if args.level:
        meta["level"] = args.level
    print(f"  decision: {meta['dedup_decision']} ({meta.get('dedup_reason', '')})", file=sys.stderr)
    print(f"  level: {meta.get('level', 'internal')}", file=sys.stderr)

    target_dir = args.target_dir or meta.get("target_subdir") or DEFAULT_TARGET_DIR
    md_dir = KNOWLEDGE / target_dir

    md_filename = slug + ".md"
    md_path = md_dir / md_filename
    if md_path.exists() and meta["dedup_decision"] == "new":
        md_path = md_dir / f"{slug}-{date.today().isoformat()}.md"
    if meta["dedup_decision"] == "conflict_with":
        md_path = md_dir / f"_conflict_{slug}-{date.today().isoformat()}.md"

    if args.dry_run:
        # Emit structured proposal so callers (kb-bot) can render previews
        # without re-running the analysis. No image download, no write.
        print(json.dumps({
            "ok": True,
            "dry_run": True,
            "target_path": str(md_path.relative_to(REPO)),
            "title": meta["title"],
            "decision": meta["dedup_decision"],
            "decision_reason": meta.get("dedup_reason", ""),
            "dedup_target": meta.get("dedup_target"),
            "max_similarity": dedup["max_score"],
            "tags": meta["tags"],
            "level": meta.get("level", "internal"),
            "summary": meta.get("summary", ""),
            "key_points": meta.get("key_points") or [],
            "source_url": args.url,
            "source_doc_id": doc_id,
        }, ensure_ascii=False, indent=2))
        return

    md_dir.mkdir(parents=True, exist_ok=True)
    media_dir = md_dir / "_media" / slug

    print(f"→ Downloading embedded images → {media_dir}", file=sys.stderr)
    body, n_images = process_content(raw, media_dir, md_dir)
    print(f"  {n_images} image(s) downloaded", file=sys.stderr)

    fm = build_frontmatter(meta, args.url, doc_id, args.uploaded_by)
    full = fm + body

    md_path.write_text(full, encoding="utf-8")
    print(f"→ Wrote {md_path}", file=sys.stderr)

    files_to_commit = [md_path]
    if media_dir.exists():
        files_to_commit.extend(p for p in media_dir.rglob("*") if p.is_file())

    commit_msg = (
        f"learn: {meta['title']} (from {args.uploaded_by})\n\n"
        f"source: {args.url}\n"
        f"dedup: {meta['dedup_decision']}"
        + (f" → {meta['dedup_target']}" if meta.get("dedup_target") else "")
    )
    git_commit(REPO, files_to_commit, commit_msg)
    print(json.dumps({
        "ok": True,
        "path": str(md_path.relative_to(REPO)),
        "decision": meta["dedup_decision"],
        "dedup_target": meta.get("dedup_target"),
        "max_similarity": dedup["max_score"],
        "n_images": n_images,
        "tags": meta["tags"],
        "level": meta.get("level", "internal"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
