"""Build a per-bot subset of a corpus repo.

Usage:
  corpus-build-subsets                                  # build all in cwd repo
  corpus-build-subsets --repo ~/gofo_hr_corpus          # specify repo
  corpus-build-subsets bot_configs/lark-qa-bot.yaml     # build one config

If --repo not given, walks up from cwd looking for knowledge/ + bot_configs/.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("pyyaml not installed — pip install gofo-corpus-tools")

LEVEL_ORDER = {"public": 0, "internal": 1, "confidential": 2}


def find_repo_root(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "knowledge").is_dir() and (candidate / "bot_configs").is_dir():
            return candidate
    sys.exit("not inside a corpus repo. pass --repo PATH explicitly.")


def parse_frontmatter(text: str) -> dict:
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    try:
        meta = yaml.safe_load(text[3:end].strip()) or {}
        return meta if isinstance(meta, dict) else {}
    except yaml.YAMLError:
        return {}


def select_files(knowledge: Path, scope: dict) -> list[Path]:
    paths_glob: list[str] = scope.get("paths") or ["**/*"]
    include_tags: set[str] = set(scope.get("include_tags") or [])
    exclude_tags: set[str] = set(scope.get("exclude_tags") or [])
    max_level: str = scope.get("max_level") or "confidential"
    max_level_n = LEVEL_ORDER.get(max_level, 2)

    candidates: set[Path] = set()
    for pat in paths_glob:
        if pat.endswith("/**"):
            pat = pat + "/*"
        candidates.update(p for p in knowledge.glob(pat) if p.is_file())

    selected: list[Path] = []
    for p in sorted(candidates):
        if p.suffix.lower() == ".md":
            meta = parse_frontmatter(p.read_text(encoding="utf-8"))
            file_tags = set(meta.get("tags") or [])
            file_level = meta.get("level") or "internal"
            if include_tags and not (file_tags & include_tags):
                continue
            if file_tags & exclude_tags:
                continue
            if LEVEL_ORDER.get(file_level, 1) > max_level_n:
                continue
        selected.append(p)
    return selected


def build_for_config(repo: Path, config_path: Path) -> dict:
    knowledge = repo / "knowledge"
    dist = repo / "dist"

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    bot_id = cfg["bot_id"]
    scope = cfg.get("knowledge_scope") or {}

    out_dir = dist / bot_id
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    files = select_files(knowledge, scope)
    md_count = 0
    asset_count = 0
    for src in files:
        rel = src.relative_to(knowledge)
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if src.suffix.lower() == ".md":
            md_count += 1
        else:
            asset_count += 1

    return {
        "bot_id": bot_id,
        "out_dir": str(out_dir.relative_to(repo)),
        "markdown_files": md_count,
        "other_files": asset_count,
        "total_files": len(files),
    }


def main() -> None:
    ap = argparse.ArgumentParser(prog="corpus-build-subsets")
    ap.add_argument("config", nargs="?",
                    help="single bot config path; default: build all in bot_configs/")
    ap.add_argument("--repo", type=Path, default=None,
                    help="Corpus repo root. Default: walk up from cwd.")
    args = ap.parse_args()

    repo = (args.repo.resolve() if args.repo else find_repo_root())
    bot_configs_dir = repo / "bot_configs"
    if not bot_configs_dir.is_dir():
        sys.exit(f"no bot_configs/ in {repo}")

    if args.config:
        cfgs = [Path(args.config).resolve()]
    else:
        cfgs = sorted(bot_configs_dir.glob("*.yaml"))
    if not cfgs:
        sys.exit("no bot configs found")

    (repo / "dist").mkdir(exist_ok=True)
    for c in cfgs:
        s = build_for_config(repo, c)
        print(f"  → {s['bot_id']}: {s['markdown_files']} md + {s['other_files']} asset files → {s['out_dir']}")


if __name__ == "__main__":
    main()
