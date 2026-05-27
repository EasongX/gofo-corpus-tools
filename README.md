# gofo-corpus-tools

GUS 知识库的共享工具——给 `gofo_it_corpus`、（未来）`gofo_hr_corpus`、`gofo_finance_corpus` 等所有 corpus repo 用。每个 corpus repo 只放 markdown 数据 + bot_configs；逻辑 / prompt / 模型 / frontmatter schema 都在这里，改一处全员生效。

## 装

```bash
# 只跑 build_bot_subset（CI 用，轻）
pip install git+https://github.com/EasongX/gofo-corpus-tools.git

# 跑 ingest 也要装（含 sentence-transformers + anthropic，~1GB）
pip install "gofo-corpus-tools[ingest] @ git+https://github.com/EasongX/gofo-corpus-tools.git"
```

或在 corpus repo 旁边 clone + `pip install -e ./gofo-corpus-tools`。

## 命令

装完之后两个 CLI：

```bash
# 入库一篇飞书文档到 cwd 所在的 corpus repo
cd ~/gofo_it_corpus
corpus-ingest "<feishu_url>" --uploaded-by "宋宜烜"

# 或显式指定 repo
corpus-ingest "<feishu_url>" --repo ~/gofo_hr_corpus --uploaded-by "..."

# 选项
corpus-ingest <url> --target-dir ops          # 强制落点
corpus-ingest <url> --level confidential      # 强制等级（否则 Claude 判）
corpus-ingest <url> --dry-run                 # 只看不写

# 打 per-bot 子集
corpus-build-subsets                                       # 所有 bot
corpus-build-subsets bot_configs/lark-qa-bot.yaml          # 单个
corpus-build-subsets --repo ~/gofo_hr_corpus               # 别的 repo
```

不带 `--repo` 时，CLI 从 cwd 往上找包含 `knowledge/` + `bot_configs/` 的目录。

## 对 corpus repo 的 contract

corpus repo 要长这样才能被这个工具用：

```
<repo>/
├── knowledge/                 ← markdown 数据，每篇带 frontmatter
│   ├── <area>/
│   │   ├── *.md
│   │   └── _media/<slug>/    ← 图片
│   └── shared/
├── bot_configs/<bot>.yaml     ← 每个下游 bot 一份
└── data/                       ← 本地 cache（gitignored）
```

`frontmatter` 必须包含：

```yaml
---
title: ...
tags: [...]                # 3-6 个 kebab-case
level: public | internal | confidential
source: ...
source_doc_id: ...         # 飞书 doc_id（自动生成）
learned_date: 2026-MM-DD
uploaded_by: ...
summary: "..."
key_points:
  - ...
---
```

`bot_configs/<bot>.yaml` 长这样：

```yaml
bot_id: <id>
name: <名称>
feishu_app_id: cli_xxxx
knowledge_scope:
  paths:                # glob 相对 knowledge/，dir/** = 该目录递归
    - "ops/**"
    - "shared/**"
  include_tags: []      # 空 = 不限
  exclude_tags: [confidential]
  max_level: internal   # public | internal | confidential
```

## 起一个新 corpus repo

```bash
gh repo create gofo_<dept>_corpus --private
git clone git@github.com:EasongX/gofo_<dept>_corpus.git
cd gofo_<dept>_corpus
mkdir -p knowledge/{shared,<area>} bot_configs data .github/workflows
# 写好 README、bot_configs/<bot>.yaml、.github/workflows/build-subsets.yml
git add . && git commit -m "init" && git push
```

CI workflow 例 (`.github/workflows/build-subsets.yml`)：

```yaml
on:
  push:
    branches: [main]
  workflow_dispatch: {}
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install git+https://github.com/EasongX/gofo-corpus-tools.git
      - run: corpus-build-subsets
      - uses: actions/upload-artifact@v4
        with: { name: bot-subsets, path: dist/ }
```

## 已知限制

见各 corpus repo 的 MAINTAINER.md "已知限制 / 风险" 章节（统一维护在那里，不在这个 tools repo 里复制一份）。
