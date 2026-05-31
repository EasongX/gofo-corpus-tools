"""Canonical frontmatter schema for all corpus topic files.

ONE schema, used by every write path:
  - corpus-ingest (URL / --from-md)         → make_frontmatter
  - corpus-distill (bulk topic synthesis)   → make_frontmatter
  - kb-bot verbal mode (create_new)         → make_frontmatter (imported)

Fields (order is stable for clean diffs):
  title         human title
  topic_slug    filename stem (no .md), e.g. 16_virtual_number_metrics
  area          business-line dir, e.g. 1x_收派作业
  tags          list[str], 3-6 kebab-case
  level         public | internal | confidential
  source        provenance: a URL, "file:<name>", "verbal (kb-bot)",
                or "distilled" — free text
  learned_date  date first added (YYYY-MM-DD)
  last_updated  date last modified (YYYY-MM-DD)
  uploaded_by   who contributed (open_id or name)
  summary       1-3 sentence summary (REQUIRED, never empty)
  key_points    list[str] (REQUIRED, never empty)
  sources       optional list of {file, distilled_at} — only distill sets this
"""
from __future__ import annotations

import json
from datetime import date

VALID_LEVELS = {"public", "internal", "confidential"}


def _yaml_str(v: str) -> str:
    """Quote a scalar safely for YAML (handles colons, quotes, etc.)."""
    return json.dumps(v or "", ensure_ascii=False)


def make_frontmatter(
    *,
    title: str,
    topic_slug: str,
    area: str,
    tags: list[str],
    level: str,
    summary: str,
    key_points: list[str],
    source: str = "",
    uploaded_by: str = "unknown",
    learned_date: str | None = None,
    last_updated: str | None = None,
    sources: list[dict] | None = None,
) -> str:
    """Build a canonical frontmatter block (including the trailing '---\\n\\n')."""
    today = date.today().isoformat()
    learned_date = learned_date or today
    last_updated = last_updated or today
    level = level if level in VALID_LEVELS else "internal"
    tags_str = "[" + ", ".join(tags or []) + "]"

    lines = ["---"]
    lines.append(f"title: {_yaml_str(title)}")
    lines.append(f"topic_slug: {topic_slug}")
    lines.append(f"area: {area}")
    lines.append(f"tags: {tags_str}")
    lines.append(f"level: {level}")
    lines.append(f"source: {_yaml_str(source)}")
    lines.append(f"learned_date: {learned_date}")
    lines.append(f"last_updated: {last_updated}")
    lines.append(f"uploaded_by: {_yaml_str(uploaded_by)}")
    lines.append(f"summary: {_yaml_str(summary)}")
    if key_points:
        lines.append("key_points:")
        for p in key_points:
            lines.append(f"  - {_yaml_str(p)}")
    else:
        lines.append("key_points: []")
    if sources:
        lines.append("sources:")
        for s in sources:
            lines.append(f"  - file: {_yaml_str(s.get('file', ''))}")
            lines.append(f"    distilled_at: {s.get('distilled_at', today)}")
    lines.append("---")
    return "\n".join(lines) + "\n\n"
