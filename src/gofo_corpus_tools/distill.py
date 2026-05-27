"""One-shot (and re-runnable) topic-based distillation of a corpus repo.

Reads all knowledge/**/*.md files (source-organized), asks Claude to:
  1. Propose a topic taxonomy
  2. For each topic, synthesize a single markdown body from the relevant
     sources, with inline citations

Outputs new `knowledge/<area>/<topic_slug>.md` files (topic-organized).
Original source files are moved to `knowledge/_archive/sources/`.

Usage:
  corpus-distill --dry-run             # print taxonomy + per-topic preview to stdout
  corpus-distill --apply                # write files + git commit (uncommitted = abort)

Re-runnable: subsequent runs read both topic files and archived sources; if
new sources have appeared (via corpus-ingest), they're picked up and merged.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from .ingest import find_repo_root

MODEL = "claude-opus-4-7"
MAX_OUTPUT_TOKENS = 64000  # opus-4-7 cap; we emit many topic bodies in one call


def _read_md(path: Path) -> tuple[dict, str]:
    """Return (frontmatter, body) for a markdown file. Frontmatter may be empty."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    try:
        meta = yaml.safe_load(text[3:end]) or {}
    except yaml.YAMLError:
        meta = {}
    body = text[end + 4:].lstrip()
    return meta if isinstance(meta, dict) else {}, body


def _collect_sources(repo: Path) -> list[dict]:
    """Read every existing knowledge/**/*.md (skipping _archive/ and topic files
    if any) and return [{path, meta, body}, ...]."""
    knowledge = repo / "knowledge"
    out: list[dict] = []
    for path in sorted(knowledge.rglob("*.md")):
        rel = path.relative_to(repo)
        # Skip archived sources to avoid double-counting on re-run
        if "_archive" in rel.parts:
            continue
        meta, body = _read_md(path)
        # Skip files that already look like topic files (have topic_slug)
        # — these are re-runs and will be regenerated.
        out.append({
            "path": str(rel),
            "title": meta.get("title", path.stem),
            "tags": meta.get("tags") or [],
            "summary": meta.get("summary", ""),
            "source": meta.get("source", ""),
            "is_topic_file": bool(meta.get("topic_slug")),
            "body": body,
        })
    return out


_SYSTEM = """\
You reorganize a corpus of source markdown files into a topic-based knowledge
base. A "topic" is a coherent unit of knowledge a teammate might ask about
(e.g. "VPN 申请", "POD 拍照标准", "退件流程"). One topic = one output file.

Coverage rules:
1. Cover ALL source content. No facts dropped. If two sources contradict,
   keep both and flag as `⚠️ 冲突: <旧文 says X / 新文 says Y / 时间戳>`.
2. One canonical fact per topic file. Don't duplicate identical info across
   topic files; cross-reference: `see [vpn.md]`.
3. Inline citation after each substantive paragraph or list block:
   `[来源: <source-path>]`.
4. Topic slug: short kebab-case English (e.g. "vpn", "pod-standard").
5. Default area "ops" unless clearly shared/.
6. Aim for 10–18 topics. Don't over-fragment; don't under-fragment.

Voice & style (CRITICAL — these docs are read by ops/IT staff in a hurry,
and by LLMs answering teammates' questions; bias hard toward direct + concrete):

- **答先于解释**: every section opens with the answer / action / SOP,
  then a 1-2 sentence "why" if non-obvious. Never write a section that
  builds up before the answer.
- **短段**: 2-3 sentences per paragraph. Frequent paragraph breaks. No
  wall-of-text. Use bullets liberally when steps are discrete.
- **第二人称 ("你")**, never公文体 ("操作人员应该"、"相关人员需要").
- **Concrete > abstract**: prefer specific numbers, station codes,
  addresses, button names, exact button paths.  "Click 【签入】on the PDA"
  beats "use the system to register".
- **Why before how when it matters**: if the policy has a non-obvious reason
  ("POD 三张图是法律证据"), state the reason in 1 sentence; don't expand.
- **Honest about edges**: if a source says something only applies to NE
  but not LGA01, write that out — never paper over differences for flow.
- **Site differences**: when an SOP varies by station (NE / BWI / LGA01
  / PHL etc.), put "**适用范围**: <range>" line above the relevant block.
- **Binding policy language**: when source text has legal/compliance
  weight (e.g., "凡是没有符合以上两条 POD 投诉都可能被认为成立"),
  preserve the original wording verbatim in a quote block:
    > <original verbatim>
  then add a short plain-language gloss if helpful.
- **No filler**: skip "在本节中，我们将...", "首先值得注意的是...",
  "总而言之...". Get to the point.
- **No ML metaphors, no English jargon for Chinese-only operational concepts,
  no self-deprecating humor.** Keep tone serious but conversational.
- **Length**: each topic body should be 200–1200 Chinese characters typically.
  Long topics (POD standard, returns process) can go to 2000+ if they cover
  multi-step procedures, but don't pad.
"""


def _build_distill_prompt(sources: list[dict]) -> str:
    """Pack all sources into a single user-turn prompt."""
    parts: list[str] = []
    parts.append("Here are the source files. Each one starts with `=== <path> ===`.")
    parts.append("")
    for s in sources:
        parts.append(f"=== {s['path']} ===")
        parts.append(f"title: {s['title']}")
        if s['tags']:
            parts.append(f"tags: {s['tags']}")
        if s['summary']:
            parts.append(f"summary: {s['summary']}")
        parts.append("")
        parts.append(s['body'][:8000])  # cap each source body
        parts.append("")
    parts.append("---")
    parts.append("OUTPUT FORMAT — strict; we machine-parse this.")
    parts.append("Use the following delimited format. Do NOT wrap in code fences.")
    parts.append("Do NOT add prose before, between, or after.")
    parts.append("Body content is free-form markdown — no escaping needed.\n")
    parts.append("""\
===TAXONOMY===
- slug: vpn
  title: VPN 申请与使用
  area: ops
- slug: pod-standard
  title: POD 拍照标准
  area: ops
- ... (all topics)
===END===

===TOPIC vpn===
title: VPN 申请与使用
area: ops
tags: ops, faq, it, vpn
level: internal
summary: 一两句概述（中文，可含「」引号但避免 ASCII " 引号）
key_points:
- 要点 1
- 要点 2
sources:
- faq/VPN 怎么申请.md
- faq/foo.md
===BODY===
## 怎么申请

按 X 流程，找 Y 部门。[来源: faq/VPN 怎么申请.md]

## 为什么需要 VPN

简要解释 why。

> 原文约束话术（如有）保留在 quote block 里
===ENDTOPIC===

===TOPIC pod-standard===
... (same shape)
===ENDTOPIC===

(Repeat for every topic listed in taxonomy. Order doesn't matter.)
""")
    return "\n".join(parts)


def call_distill(sources: list[dict]) -> dict:
    from anthropic import Anthropic
    key = os.environ.get("AGENT_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        sys.exit("missing AGENT_ANTHROPIC_API_KEY")
    client = Anthropic(api_key=key)

    user_prompt = _build_distill_prompt(sources)
    print(f"→ distill: {len(sources)} sources, prompt ~{len(user_prompt)} chars", file=sys.stderr)

    # Streaming required for max_tokens this large (SDK rule for jobs > 10min).
    chunks: list[str] = []
    with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_OUTPUT_TOKENS,
        system=_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    ) as stream:
        for text_chunk in stream.text_stream:
            chunks.append(text_chunk)
            # crude progress indicator: dot every 1KB output
            if sum(len(c) for c in chunks) // 1000 > (sum(len(c) for c in chunks[:-1]) // 1000):
                print(".", end="", file=sys.stderr, flush=True)
    print("", file=sys.stderr)
    text = "".join(chunks)
    debug_path = Path("/tmp/distill_raw_output.txt")
    debug_path.write_text(text, encoding="utf-8")
    try:
        return parse_delimited(text)
    except Exception as e:
        sys.exit(f"parse failed ({e}). raw saved to {debug_path}")


def parse_delimited(text: str) -> dict:
    """Parse the ===TAXONOMY===/===TOPIC slug===/===BODY===/===ENDTOPIC=== format."""
    # Taxonomy section
    tax_m = re.search(r"===TAXONOMY===\s*(.*?)\s*===END===", text, re.DOTALL)
    if not tax_m:
        raise ValueError("no ===TAXONOMY===...===END=== block found")
    tax_yaml = tax_m.group(1)
    try:
        taxonomy = yaml.safe_load(tax_yaml) or []
    except yaml.YAMLError as e:
        raise ValueError(f"taxonomy yaml parse failed: {e}") from e
    if not isinstance(taxonomy, list):
        raise ValueError(f"taxonomy not a list: {type(taxonomy)}")

    # Topic blocks
    topics: list[dict] = []
    topic_pat = re.compile(
        r"===TOPIC\s+([^\s=]+)===\s*\n(.*?)\n===BODY===\s*\n(.*?)\n===ENDTOPIC===",
        re.DOTALL,
    )
    for m in topic_pat.finditer(text):
        slug = m.group(1).strip()
        header_yaml = m.group(2)
        body = m.group(3).strip()
        try:
            header = yaml.safe_load(header_yaml) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"topic {slug!r} header yaml failed: {e}") from e
        if not isinstance(header, dict):
            raise ValueError(f"topic {slug!r} header not a dict")
        # Normalize tags (may come as comma-separated string)
        tags = header.get("tags")
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        topics.append({
            "slug": slug,
            "title": header.get("title", slug),
            "area": header.get("area", "ops"),
            "tags": tags or [],
            "level": header.get("level", "internal"),
            "summary": header.get("summary", ""),
            "key_points": header.get("key_points") or [],
            "sources": header.get("sources") or [],
            "body": body,
        })
    if not topics:
        raise ValueError("no ===TOPIC...===ENDTOPIC=== blocks parsed")
    return {"taxonomy": taxonomy, "topics": topics}


def _topic_filename(slug: str) -> str:
    safe = re.sub(r"[^a-z0-9-]+", "-", slug.lower()).strip("-") or "untitled"
    return safe + ".md"


def _render_topic_md(topic: dict, today: str) -> str:
    """Render a topic dict into a markdown file (frontmatter + body)."""
    tags = topic.get("tags") or []
    tags_str = "[" + ", ".join(tags) + "]"
    key_points = topic.get("key_points") or []
    kp_yaml = "\n".join(f"  - {p}" for p in key_points) if key_points else ""
    sources_yaml = "\n".join(
        f"  - file: {s!r}\n    distilled_at: {today}" for s in (topic.get("sources") or [])
    ) or "  []"

    fm = (
        "---\n"
        f"title: {topic['title']}\n"
        f"topic_slug: {topic['slug']}\n"
        f"area: {topic.get('area', 'ops')}\n"
        f"tags: {tags_str}\n"
        f"level: {topic.get('level', 'internal')}\n"
        f"distilled_date: {today}\n"
        f"last_updated: {today}\n"
        f"summary: {json.dumps(topic.get('summary', ''), ensure_ascii=False)}\n"
        + (f"key_points:\n{kp_yaml}\n" if key_points else "")
        + f"sources:\n{sources_yaml}\n"
        + "---\n\n"
    )
    return fm + (topic.get("body") or "").rstrip() + "\n"


def write_distilled(repo: Path, plan: dict, archive: bool = True) -> dict:
    knowledge = repo / "knowledge"
    today = date.today().isoformat()
    written: list[str] = []
    archived: list[str] = []

    # Write each topic file
    for topic in plan.get("topics", []):
        area = topic.get("area") or "ops"
        out_path = knowledge / area / _topic_filename(topic["slug"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(_render_topic_md(topic, today), encoding="utf-8")
        written.append(str(out_path.relative_to(repo)))

    # Archive original sources (skip _archive itself, skip topic files we just wrote)
    if archive:
        archive_root = knowledge / "_archive" / "sources"
        archive_root.mkdir(parents=True, exist_ok=True)
        topic_paths = {Path(p) for p in written}
        for path in sorted(knowledge.rglob("*.md")):
            rel = path.relative_to(repo)
            if "_archive" in rel.parts:
                continue
            if rel in topic_paths:
                continue
            dst = archive_root / rel.relative_to("knowledge")
            dst.parent.mkdir(parents=True, exist_ok=True)
            # also move adjacent _media folder if any
            os.replace(path, dst)
            archived.append(str(rel))
        # archive _media folders + other non-md assets (e.g. reference/labels/)
        for asset_dir_name in ("_media", "reference"):
            for asset_dir in sorted(knowledge.rglob(asset_dir_name)):
                rel = asset_dir.relative_to(knowledge)
                if "_archive" in rel.parts:
                    continue
                if not asset_dir.is_dir():
                    continue
                dst = archive_root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    for child in list(asset_dir.iterdir()):
                        os.replace(child, dst / child.name)
                    try:
                        asset_dir.rmdir()
                    except OSError:
                        pass
                else:
                    os.replace(asset_dir, dst)
        # clean empty subdirs left behind
        for child in list(knowledge.rglob("*")):
            if child.is_dir() and not any(child.iterdir()):
                # skip _archive itself
                if "_archive" in child.relative_to(knowledge).parts:
                    continue
                try:
                    child.rmdir()
                except OSError:
                    pass

    return {"written": written, "archived": archived}


def git_assert_clean(repo: Path) -> None:
    r = subprocess.run(["git", "status", "--porcelain"], cwd=repo,
                       capture_output=True, text=True, check=True)
    if r.stdout.strip():
        sys.exit("working tree not clean — commit or stash first, distill is destructive")


def git_commit(repo: Path, msg: str) -> None:
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", msg], cwd=repo, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(prog="corpus-distill")
    ap.add_argument("--repo", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="print taxonomy + topic count; do not write files")
    ap.add_argument("--apply", action="store_true",
                    help="write files, archive sources, git commit")
    ap.add_argument("--save-plan", type=Path, default=None,
                    help="dump Claude's plan JSON to this path (for debugging / replay)")
    ap.add_argument("--load-plan", type=Path, default=None,
                    help="skip Claude call, apply this previously-saved plan JSON")
    args = ap.parse_args()

    if not args.dry_run and not args.apply and not args.load_plan:
        sys.exit("pass --dry-run or --apply (--load-plan implies write)")

    repo = (args.repo.resolve() if args.repo else find_repo_root())
    knowledge = repo / "knowledge"
    if not knowledge.is_dir():
        sys.exit(f"no knowledge/ in {repo}")

    if args.apply:
        git_assert_clean(repo)

    if args.load_plan:
        plan = json.loads(args.load_plan.read_text(encoding="utf-8"))
        print(f"→ loaded plan from {args.load_plan} ({len(plan.get('topics', []))} topics)",
              file=sys.stderr)
    else:
        sources = _collect_sources(repo)
        print(f"→ {len(sources)} sources found", file=sys.stderr)
        plan = call_distill(sources)
        print(f"→ Claude proposed {len(plan.get('topics', []))} topics", file=sys.stderr)

    if args.save_plan:
        args.save_plan.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"→ plan saved to {args.save_plan}", file=sys.stderr)

    # Taxonomy preview
    print("\n=== TAXONOMY ===")
    for t in plan.get("taxonomy", []):
        topic_obj = next((tt for tt in plan["topics"] if tt["slug"] == t["slug"]), {})
        n_sources = len(topic_obj.get("sources") or [])
        print(f"  · {t['area']}/{t['slug']}.md — {t['title']}  ({n_sources} sources)")

    if args.dry_run and not args.apply and not args.load_plan:
        # show first topic body as a sample
        if plan.get("topics"):
            first = plan["topics"][0]
            print(f"\n=== SAMPLE: {first['area']}/{first['slug']}.md ===")
            print((first.get("body") or "")[:1500])
        return

    if args.apply or args.load_plan:
        result = write_distilled(repo, plan, archive=True)
        print(f"\n→ wrote {len(result['written'])} topic files", file=sys.stderr)
        print(f"→ archived {len(result['archived'])} source files", file=sys.stderr)
        n_topics = len(plan.get("topics", []))
        n_sources_archived = len(result['archived'])
        git_commit(repo, f"distill: regroup {n_sources_archived} source files into {n_topics} topic files\n\nMoved source-organized markdown to knowledge/_archive/sources/.\nNew topic-organized files under knowledge/<area>/<topic>.md, each with\n`sources:` frontmatter tracking provenance.")
        print("→ committed", file=sys.stderr)


if __name__ == "__main__":
    main()
