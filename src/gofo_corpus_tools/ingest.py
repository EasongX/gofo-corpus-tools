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
from typing import Any

import yaml

from .frontmatter import make_frontmatter

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


LARK_BASE = ["lark-cli", "--profile", "kb-bot"]
WIKI_RE = re.compile(r"/wiki/([A-Za-z0-9]+)")
SHEETS_RE = re.compile(r"/sheets/([A-Za-z0-9]+)")
DOCX_RE = re.compile(r"/(?:docx|docs|doc)/([A-Za-z0-9]+)")
MAX_SHEET_ROWS = 5000  # per-tab safety cap so a runaway sheet can't hang ingest


def _col_letter(n: int) -> str:
    """1-based column index to A1 letters: 1->A, 26->Z, 27->AA, 30->AD."""
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s or "A"


def _lark(args: list[str]) -> dict:
    """Run a lark-cli subcommand and return parsed JSON. lark-cli exits non-zero
    on API errors but still prints the JSON error envelope to stdout, so parse
    stdout regardless of return code and let the caller inspect ok/code/error
    (this lets callers turn e.g. a missing-scope error into actionable help).
    Only hard-exit when there's no JSON to inspect at all."""
    r = subprocess.run(LARK_BASE + args, capture_output=True, text=True, check=False)
    # On success the envelope lands on stdout; on API errors lark-cli exits
    # non-zero and writes the {ok:false,error:...} envelope to stderr instead.
    for stream in (r.stdout, r.stderr):
        if not stream.strip():
            continue
        try:
            return json.loads(stream)
        except json.JSONDecodeError:
            continue
    sys.exit(f"lark-cli {' '.join(args)} failed:\n{r.stderr or r.stdout}")


def _resolve_url(url: str) -> tuple[str, str]:
    """Resolve a Feishu URL to (obj_type, token).

    obj_type is Feishu's object type ('docx', 'sheet', 'bitable', ...). Wiki
    URLs wrap an underlying object, so they're resolved through the wiki node
    API to find the real obj_type/obj_token (a /wiki/ link can point at a docx,
    a sheet, a bitable, etc.). For direct /docx/ and /sheets/ links the token in
    the path is the object token itself.
    """
    m = WIKI_RE.search(url)
    if m:
        data = _lark(["api", "GET", "/open-apis/wiki/v2/spaces/get_node",
                      "--params", json.dumps({"token": m.group(1)}), "--as", "bot"])
        node = (data.get("data") or {}).get("node")
        if data.get("code") not in (0, None) or not node:
            sys.exit(f"wiki node resolve failed: {data.get('msg') or data.get('error')}")
        return node["obj_type"], node["obj_token"]
    m = SHEETS_RE.search(url)
    if m:
        return "sheet", m.group(1)
    m = DOCX_RE.search(url)
    if m:
        return "docx", m.group(1)
    return "docx", url  # unknown shape: let docx fetch resolve / error


def fetch_doc(url: str) -> dict:
    """Fetch a Feishu doc as markdown. Supports docx (native or wiki-wrapped)
    and sheet (every tab rendered as a markdown table). Returns the uniform
    shape {content, document_id} the rest of the pipeline expects, where
    content begins with a <title>…</title> line."""
    obj_type, token = _resolve_url(url)
    if obj_type == "sheet":
        return fetch_sheet(token)
    # docx and everything else: docs +fetch resolves wiki wrappers itself, so
    # pass the original URL through unchanged.
    cmd = [
        "docs", "+fetch", "--api-version", "v2", "--doc", url, "--as", "bot",
        "--doc-format", "markdown",
    ]
    data = _lark(cmd)
    if not data.get("ok"):
        sys.exit(f"fetch error: {data.get('error')}")
    return data["data"]["document"]


def _sheet_scope_exit(error: Any) -> None:
    """Sheet read failed: turn Feishu's opaque scope error into actionable help."""
    msg = error.get("message", "") if isinstance(error, dict) else str(error)
    code = error.get("code") if isinstance(error, dict) else None
    blob = f"{code} {msg}".lower()
    if code == 99991672 or "scope" in blob or "permission" in blob:
        sys.exit(
            "读取电子表格失败：kb-bot 飞书应用缺少电子表格读取权限。\n"
            "请在飞书开发者后台为该自建应用开通 scope `sheets:spreadsheet:readonly`"
            "（开通后需发布新版本生效），再重新发送链接。\n"
            f"(原始错误: {msg})"
        )
    sys.exit(f"读取电子表格失败: {msg}")


def _values_to_markdown(values: list) -> str:
    """Render a 2D cell array as a GitHub-flavored markdown table. First
    non-empty row becomes the header. Empty trailing rows/cols are dropped."""
    rows = [["" if c is None else str(c) for c in (r or [])] for r in (values or [])]
    rows = [r for r in rows if any(cell.strip() for cell in r)]
    if not rows:
        return "*(空表)*"
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    # Drop fully-empty columns: wide grids carry many blank columns that are pure
    # noise once flattened to markdown (and inflate the embedded text).
    keep = [i for i in range(width) if any(r[i].strip() for r in rows)]
    if keep:
        rows = [[r[i] for i in keep] for r in rows]

    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")

    out = [
        "| " + " | ".join(esc(c) for c in rows[0]) + " |",
        "| " + " | ".join("---" for _ in rows[0]) + " |",
    ]
    for r in rows[1:]:
        out.append("| " + " | ".join(esc(c) for c in r) + " |")
    return "\n".join(out)


PAGE = 200  # lark-cli returns at most ~200 rows per +read call


def _read_range(token: str, rng: str) -> tuple[list, str]:
    """One +read call. Returns (values, echoed_range). Cell values live under
    data.valueRange.{values,range} (camelCase, no truncated/total_rows fields)."""
    rd = _lark(["sheets", "+read", "--spreadsheet-token", token, "--range", rng,
                "--value-render-option", "ToString", "--as", "bot"])
    if not rd.get("ok"):
        _sheet_scope_exit(rd.get("error"))
    vr = (rd.get("data") or {}).get("valueRange") or {}
    return list(vr.get("values") or []), vr.get("range") or ""


def _read_tab_rows(token: str, sid: str, row_count: int, col_count: int) -> tuple[list, bool]:
    """Read all rows of one tab in <=PAGE-row windows. Always uses explicit
    "<sheetId>!A1:<col><row>" ranges: a bare sheetId that looks like A1 notation
    (e.g. 'PKSQ42') is otherwise misparsed by lark-cli as a cell reference. Reads
    up to the tab's grid row_count (capped at MAX_SHEET_ROWS), stopping early at
    the first all-empty window. Trailing empty rows are dropped when rendering.
    Returns (values, capped)."""
    last_col = _col_letter(col_count or 1)
    limit = min(row_count or MAX_SHEET_ROWS, MAX_SHEET_ROWS)
    rows: list = []
    start = 1
    while start <= limit:
        end = min(start + PAGE - 1, limit)
        vals, _ = _read_range(token, f"{sid}!A{start}:{last_col}{end}")
        if not any(any((c not in (None, "")) for c in (r or [])) for r in vals):
            break  # an all-empty window means we're past the real data
        rows.extend(vals)
        start = end + 1
    return rows, (row_count or 0) > limit


def fetch_sheet(token: str) -> dict:
    """Read every tab of a spreadsheet and render it as markdown tables, in the
    same {content, document_id} shape as a docx fetch so downstream ingest code
    (title extraction, image scan, dedup, frontmatter) stays type-agnostic.
    All tabs are read; each tab is paginated to its full row count."""
    info = _lark(["sheets", "+info", "--spreadsheet-token", token, "--as", "bot"])
    if not info.get("ok"):
        _sheet_scope_exit(info.get("error"))
    data = info.get("data") or {}
    # lark-cli wraps both under an extra key: data.spreadsheet.spreadsheet.title
    # and data.sheets.sheets[]. Unwrap defensively so a flatter future shape
    # still works.
    sp = data.get("spreadsheet") or {}
    if isinstance(sp.get("spreadsheet"), dict):
        sp = sp["spreadsheet"]
    title = sp.get("title") or "电子表格"
    sheets = data.get("sheets") or []
    if isinstance(sheets, dict):
        sheets = sheets.get("sheets") or []

    parts = [f"<title>{title}</title>", ""]
    for sh in sheets:
        sid = sh.get("sheet_id") or sh.get("sheetId")
        tab = sh.get("title") or sid
        if not sid or sh.get("hidden"):
            continue
        gp = sh.get("grid_properties") or {}
        row_count = gp.get("row_count") or gp.get("rowCount") or 0
        col_count = gp.get("column_count") or gp.get("columnCount") or 26
        values, capped = _read_tab_rows(token, sid, row_count, col_count)
        if len(sheets) > 1:
            parts.append(f"## {tab}")
            parts.append("")
        parts.append(_values_to_markdown(values))
        if capped:
            parts.append(f"\n*（表「{tab}」行数超过 {MAX_SHEET_ROWS} 行上限，仅含前 {len(values)} 行）*")
        parts.append("")
    return {"content": "\n".join(parts).rstrip() + "\n", "document_id": f"sheet:{token}"}


SHEET_DISTILL_MODEL = "claude-opus-4-7"

SHEET_DISTILL_PROMPT = """\
你在为一个用于**语义检索（RAG）**的知识库整理资料。下面是从飞书电子表格《{title}》\
导出的原始内容，每个工作表（tab）渲染成了一张 markdown 表格。

原始表格直接入库检索效果很差：空单元格、合并单元格碎片、宽表布局都是噪声，按字符\
切块还会把表头和数据切散、丢失语义。你的任务是把表格里的**信息**提炼成一篇结构化的\
中文知识文档，让每条信息都成为**自包含、可被独立检索命中**的陈述。

要求：
- 忠实转写，**不编造、不遗漏**任何有效信息（指标定义、计算/研发逻辑、口径、字段含义、\
查询条件、规则、已知问题、备注等都要保留）。
- 把表格的行列关系还原成完整句子。例：不要写「待分拣 | 见概况」，要写\
「『待分拣』指标：定义见『概况』工作表；…（把该行其它列的信息也并入）」。
- 按表格自身的工作表/分区组织成小节，用 `##` / `###` 标题；同类条目可归并。
- 丢弃纯空白、纯排版、无信息量的内容；不要保留原始表格。
- 保留具体的名称、条件、公式、字段名、数值口径等关键细节，不要泛化成空话。
- 看不懂或语义不明的内容，原样保留并标注「（原文，含义待确认）」，不要臆测。
- 只输出整理后的 markdown 正文，**不要**输出 `<title>` 行、不要加任何前言或说明。

原始内容如下：

{body}
"""


def distill_sheet(title: str, raw_content: str) -> str:
    """Turn a spreadsheet's raw markdown tables into retrieval-friendly knowledge
    prose. Raw tables embed and chunk badly (empty cells, merged-cell fragments,
    headers split from data); this rewrites the *information* as self-contained
    statements. Returns markdown beginning with the <title> line, ready to take
    the place of the raw body. Falls back to the raw content if Claude fails."""
    from anthropic import Anthropic

    api_key = os.environ.get("AGENT_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("missing AGENT_ANTHROPIC_API_KEY (see ~/.claude/secrets/anthropic.env)")
    body = TITLE_RE.sub("", raw_content, count=1).lstrip()
    try:
        resp = Anthropic(api_key=api_key).messages.create(
            model=SHEET_DISTILL_MODEL,
            max_tokens=16000,
            messages=[{"role": "user",
                       "content": SHEET_DISTILL_PROMPT.format(title=title, body=body)}],
        )
        distilled = "".join(
            b.text for b in resp.content if getattr(b, "type", None) == "text"
        ).strip()
    except Exception as e:
        print(f"  ⚠️ 表格提炼失败，回退原始表格: {e}", file=sys.stderr)
        return raw_content
    if not distilled:
        print("  ⚠️ 表格提炼返回空，回退原始表格", file=sys.stderr)
        return raw_content
    return f"<title>{title}</title>\n\n{distilled}\n"


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Return (frontmatter_dict, body) for a corpus markdown file."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    try:
        fm = yaml.safe_load(text[3:end]) or {}
    except yaml.YAMLError:
        fm = {}
    return (fm if isinstance(fm, dict) else {}), text[end + 4:].lstrip()


def _source_doc_id(source_field: str) -> str:
    """Pull the doc_id out of a frontmatter source string like
    'https://… (doc_id=sheet:XXX)' or 'file:… (doc_id=file:abc)'."""
    m = re.search(r"doc_id=([^\s)]+)", source_field or "")
    return m.group(1) if m else ""


_LEVEL_RANK = {"public": 0, "internal": 1, "confidential": 2}


def _max_level(a: str | None, b: str | None) -> str:
    """Return the more sensitive of two levels (never downgrade on merge)."""
    ra, rb = _LEVEL_RANK.get(a or "", 1), _LEVEL_RANK.get(b or "", 1)
    return a if ra >= rb else b


MERGE_PROMPT = """\
你在维护一个用于语义检索的知识库。下面有一篇**已有知识文档**和一条**新的更新**\
（来自同主题的另一份资料），二者被判定为同一主题、新内容是对旧文档的更新/扩展。

请把它们合并成一篇文档，规则：
- 整合新信息；新内容明确取代旧说法的，用新的、删掉过时的。
- **保留**旧文档里仍然有效、新内容未涉及的所有信息，不要丢。
- **真正的口径冲突**（同一件事，新旧给出不同定义/数值/做法且无法判断谁对）：\
**不要静默二选一**。两个版本都保留，在该处用 `> ⚠️ 口径冲突（待人工确认）：旧=…；新=…` 显式标注。
- 保持适合检索的结构（小标题、自包含陈述句），不要无谓改写没冲突的内容。
- 不编造、不臆测。

只返回一个 JSON 对象（不要任何额外文字），字段：
- body: 合并后的 markdown 正文（不含 frontmatter、不含 <title> 行）
- summary: 1-3 句中文摘要
- key_points: 3-5 条要点（字符串数组，不带前导符号）
- conflicts: 数组，每个元素 {{"point":"冲突的点","existing":"旧口径","new":"新口径"}}；无冲突则为空数组

【已有知识文档】标题：{title}
{existing}

【新的更新】
{new}
"""


def merge_docs(title: str, existing_body: str, new_body: str) -> dict:
    """Merge a new update into an existing doc body via Claude. Returns
    {body, summary, key_points, conflicts}. Raises on hard failure so the caller
    can refuse to write (we must never silently drop the existing content)."""
    from anthropic import Anthropic

    api_key = os.environ.get("AGENT_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("missing AGENT_ANTHROPIC_API_KEY (see ~/.claude/secrets/anthropic.env)")
    resp = Anthropic(api_key=api_key).messages.create(
        model=SHEET_DISTILL_MODEL,
        max_tokens=16000,
        messages=[{"role": "user", "content": MERGE_PROMPT.format(
            title=title, existing=existing_body, new=new_body)}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        sys.exit(f"merge: claude returned no JSON: {text[:400]}")
    data = json.loads(m.group(0))
    if not data.get("body"):
        sys.exit("merge: claude returned empty body; refusing to overwrite existing doc")
    data.setdefault("conflicts", [])
    return data


def extract_title(content: str) -> str:
    m = TITLE_RE.search(content)
    return m.group(1).strip() if m else "untitled"


def slugify(title: str) -> str:
    """Underscore-separated slug. Convention: knowledge/<line>/<NN>_<slug>.md
    uses underscore everywhere, so slug itself shouldn't introduce hyphens."""
    s = re.sub(r"[\s\\/-]+", "_", title.strip())
    s = re.sub(r"[<>:\"|?*]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:80] or "untitled"


def next_line_seq(line_dir: Path) -> str:
    """Compute next <line><seq> prefix for a new file in a business-line dir.

    Line dirs are named like '1x_收派作业'. The seq within a line starts at 0:
      first file = '10', second = '11', ..., tenth = '19', eleventh = '110'
      (no carry to 20, which belongs to line 2). Returns '' if dir name is
      not a recognized line (e.g. '_archive') — caller skips the prefix.
    """
    name = line_dir.name
    m = re.match(r"^(\d)x_", name)
    if not m:
        return ""
    line_prefix = m.group(1)
    used: set[int] = set()
    for p in line_dir.glob("*.md"):
        m2 = re.match(rf"^{line_prefix}(\d+)_", p.name)
        if m2:
            used.add(int(m2.group(1)))
    seq = 0
    while seq in used:
        seq += 1
    return f"{line_prefix}{seq}"


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
    user_hint: str = "",
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

    user_hint_block = (
        f"\n**USER HINT (the teammate's instructions about this doc — "
        f"FOLLOW THIS over your default classification)**:\n{user_hint}\n"
    ) if user_hint.strip() else ""

    prompt = f"""You analyze a Lark/Feishu doc that a teammate just submitted to a team
knowledge base and produce metadata + a duplicate/conflict judgment.

NEW DOC (title hint: {title_hint!r}):
---
{body_excerpt}
---
{user_hint_block}
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


def build_frontmatter(meta: dict, source_url: str, source_doc_id: str,
                      uploaded_by: str, *, topic_slug: str = "", area: str = "",
                      learned_date: str | None = None) -> str:
    from .frontmatter import make_frontmatter
    src = source_url
    if source_doc_id:
        src = f"{source_url} (doc_id={source_doc_id})" if source_url else source_doc_id
    return make_frontmatter(
        title=meta["title"],
        topic_slug=topic_slug,
        area=area,
        tags=meta.get("tags") or [],
        level=meta.get("level", "internal"),
        summary=meta.get("summary", ""),
        key_points=meta.get("key_points") or [],
        source=src,
        uploaded_by=uploaded_by,
        learned_date=learned_date,
    )


def git_commit(repo: Path, files: list[Path], message: str) -> None:
    subprocess.run(["git", "add"] + [str(f) for f in files], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=repo, check=True)


def main() -> None:
    global REPO, KNOWLEDGE, DATA

    ap = argparse.ArgumentParser(prog="corpus-ingest")
    ap.add_argument("url", nargs="?", default=None,
                    help="Feishu doc/wiki URL (omit when using --from-md)")
    ap.add_argument("--from-md", type=Path, default=None,
                    help="Read body from a local markdown file instead of fetching "
                         "from Feishu. Requires --title.")
    ap.add_argument("--title", default=None,
                    help="Title for --from-md mode (used as the doc title)")
    ap.add_argument("--repo", type=Path, default=None,
                    help="Corpus repo root. Default: walk up from cwd.")
    ap.add_argument("--uploaded-by", default="unknown")
    ap.add_argument("--target-dir", default=None,
                    help="Override Claude's choice of target subdir")
    ap.add_argument("--level", choices=sorted(VALID_LEVELS), default=None,
                    help="Force sensitivity level; otherwise Claude classifies.")
    ap.add_argument("--hint", default="",
                    help="User's instruction about this doc (e.g. 'only extract "
                         "metric calc rules; put under 2x_操作运输'). Claude "
                         "follows this over its default classification.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.url and not args.from_md:
        sys.exit("must provide either <url> or --from-md FILE")
    if args.from_md and not args.title:
        sys.exit("--from-md requires --title 'doc title'")

    REPO = (args.repo.resolve() if args.repo else find_repo_root())
    KNOWLEDGE = REPO / "knowledge"
    DATA = REPO / "data"
    if not KNOWLEDGE.is_dir():
        sys.exit(f"no knowledge/ found in {REPO}")
    DATA.mkdir(exist_ok=True)

    print(f"→ Repo: {REPO}", file=sys.stderr)
    if args.from_md:
        # Local-markdown path: skip Feishu fetch.
        print(f"→ Reading local markdown: {args.from_md}", file=sys.stderr)
        body_md = args.from_md.read_text(encoding="utf-8")
        title_hint = args.title
        # Synthesize "raw" so downstream code (which strips <title>) stays uniform
        raw = f"<title>{title_hint}</title>\n\n{body_md}"
        doc_id = f"file:{hashlib.sha256(body_md.encode()).hexdigest()[:12]}"
        source_url = args.url or f"file:{args.from_md.name}"
    else:
        print(f"→ Fetching {args.url}", file=sys.stderr)
        doc = fetch_doc(args.url)
        raw = doc["content"]
        doc_id = doc["document_id"]
        title_hint = extract_title(raw)
        source_url = args.url
        # Spreadsheets are stored as distilled knowledge prose, not raw tables:
        # raw tables embed/chunk poorly for retrieval. Done before dedup + meta so
        # those run on the same high-quality text that gets stored.
        if doc_id.startswith("sheet:"):
            print("→ Distilling spreadsheet into knowledge prose (Claude)...", file=sys.stderr)
            raw = distill_sheet(title_hint, raw)
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
        source_url=source_url,
        uploaded_by=args.uploaded_by,
        dedup=dedup,
        level_override=args.level,
        target_subdirs_hint=list_target_subdirs(),
        user_hint=args.hint,
    )
    if args.level:
        meta["level"] = args.level
    print(f"  decision: {meta['dedup_decision']} ({meta.get('dedup_reason', '')})", file=sys.stderr)
    print(f"  level: {meta.get('level', 'internal')}", file=sys.stderr)

    # "update_of" writes into the existing file in place rather than dropping a
    # second numbered file next to it (which would pile up 42_/43_/44_ dupes).
    # HOW it writes depends on whether it's the SAME source doc:
    #   - same source (re-fetch of the same sheet/docx) → overwrite: the new
    #     content is the authoritative complete version.
    #   - different source (e.g. a Scout Q&A card extending an existing topic
    #     file) → MERGE: blindly overwriting would wipe the target down to just
    #     the new card. We must combine and flag any 口径 conflicts.
    update_target = None
    update_mode = None  # None | "overwrite" | "merge"
    existing_fm: dict = {}
    if meta["dedup_decision"] == "update_of" and meta.get("dedup_target"):
        cand = KNOWLEDGE / meta["dedup_target"]
        if cand.is_file():
            update_target = cand
            existing_fm, _ = split_frontmatter(cand.read_text(encoding="utf-8"))
            old_id = _source_doc_id(str(existing_fm.get("source", "")))
            update_mode = "overwrite" if (old_id and old_id == doc_id) else "merge"

    if update_target is not None:
        md_path = update_target
        md_dir = md_path.parent
        target_dir = str(md_dir.relative_to(KNOWLEDGE))
    else:
        target_dir = args.target_dir or meta.get("target_subdir") or DEFAULT_TARGET_DIR
        md_dir = KNOWLEDGE / target_dir
        md_dir.mkdir(parents=True, exist_ok=True)

        # Auto-prepend business-line numbered prefix (e.g. '16_' for the 7th file
        # in 1x_收派作业). Skips if target_dir doesn't match the <N>x_ pattern.
        seq_prefix = next_line_seq(md_dir)
        fname_core = f"{seq_prefix}_{slug}" if seq_prefix else slug

        md_path = md_dir / f"{fname_core}.md"
        if md_path.exists() and meta["dedup_decision"] == "new":
            md_path = md_dir / f"{fname_core}_{date.today().isoformat()}.md"
        if meta["dedup_decision"] == "conflict_with":
            md_path = md_dir / f"_conflict_{fname_core}_{date.today().isoformat()}.md"

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
            "update_mode": update_mode,
            "max_similarity": dedup["max_score"],
            "tags": meta["tags"],
            "level": meta.get("level", "internal"),
            "summary": meta.get("summary", ""),
            "key_points": meta.get("key_points") or [],
            "source_url": source_url,
            "source_doc_id": doc_id,
        }, ensure_ascii=False, indent=2))
        return

    media_dir = md_dir / "_media" / slug

    print(f"→ Downloading embedded images → {media_dir}", file=sys.stderr)
    body, n_images = process_content(raw, media_dir, md_dir)
    print(f"  {n_images} image(s) downloaded", file=sys.stderr)

    conflicts: list = []
    if update_mode == "merge":
        # Different source updating an existing topic file: merge instead of
        # overwrite so the target's existing knowledge isn't wiped, and surface
        # any 口径 conflicts rather than silently resolving them.
        print(f"→ Merging into existing {md_path.name} (Claude)...", file=sys.stderr)
        _, existing_body = split_frontmatter(md_path.read_text(encoding="utf-8"))
        merged = merge_docs(meta["title"], existing_body, body)
        conflicts = merged.get("conflicts") or []
        body = merged["body"].rstrip() + "\n"
        today = date.today().isoformat()
        fm = make_frontmatter(
            title=existing_fm.get("title") or meta["title"],
            topic_slug=md_path.stem,
            area=target_dir,
            tags=sorted(set(existing_fm.get("tags") or []) | set(meta.get("tags") or [])),
            level=_max_level(existing_fm.get("level"), meta.get("level")),
            summary=merged.get("summary") or meta.get("summary", ""),
            key_points=merged.get("key_points") or meta.get("key_points") or [],
            source=str(existing_fm.get("source", "")) or source_url,
            uploaded_by=existing_fm.get("uploaded_by") or args.uploaded_by,
            learned_date=str(existing_fm.get("learned_date") or today),
            last_updated=today,
            sources=(existing_fm.get("sources") or [])
            + [{"file": source_url or doc_id, "distilled_at": today}],
            it_actions=existing_fm.get("it_actions"),
        )
        print(f"  merged; {len(conflicts)} 口径 conflict(s) flagged", file=sys.stderr)
    else:
        # "new", "conflict_with", or same-source "update_of" overwrite. Preserve
        # the original creation date when overwriting an existing file.
        fm = build_frontmatter(meta, source_url, doc_id, args.uploaded_by,
                               topic_slug=md_path.stem, area=target_dir,
                               learned_date=str(existing_fm.get("learned_date") or "") or None)
    full = fm + body

    md_path.write_text(full, encoding="utf-8")
    print(f"→ Wrote {md_path}", file=sys.stderr)

    files_to_commit = [md_path]
    if media_dir.exists():
        files_to_commit.extend(p for p in media_dir.rglob("*") if p.is_file())

    commit_msg = (
        f"learn: {meta['title']} (from {args.uploaded_by})\n\n"
        f"source: {source_url}\n"
        f"dedup: {meta['dedup_decision']}"
        + (f" → {meta['dedup_target']} ({update_mode})" if meta.get("dedup_target") else "")
        + (f"\nconflicts flagged: {len(conflicts)}" if conflicts else "")
    )
    git_commit(REPO, files_to_commit, commit_msg)
    print(json.dumps({
        "ok": True,
        "path": str(md_path.relative_to(REPO)),
        "decision": meta["dedup_decision"],
        "dedup_target": meta.get("dedup_target"),
        "update_mode": update_mode,
        "conflicts": conflicts,
        "max_similarity": dedup["max_score"],
        "n_images": n_images,
        "tags": meta["tags"],
        "level": meta.get("level", "internal"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
