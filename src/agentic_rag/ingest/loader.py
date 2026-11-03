"""从种子数据加载非结构化文档。"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import settings
from ..schema import Document


def load_policy_documents(seed_dir: Path | None = None) -> list[Document]:
    """加载政策类文档（退货/售后/配送/会员）。"""
    seed_dir = seed_dir or settings.seed_dir
    data = json.loads((seed_dir / "policies.json").read_text(encoding="utf-8"))
    return [Document(doc_id=d["id"], title=d["title"], content=d["content"]) for d in data]
