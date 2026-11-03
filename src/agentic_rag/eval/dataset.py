"""评测数据集加载。"""
from __future__ import annotations

import json

from ..config import settings


def load_retrieval_dataset() -> list[dict]:
    """检索评测集：query → 相关文档 id。"""
    return json.loads(
        (settings.eval_dir / "retrieval_dataset.json").read_text(encoding="utf-8")
    )


def load_qa_dataset() -> list[dict]:
    """QA 评测集：question + 参考答案（供 RAGAS 使用）。"""
    return json.loads((settings.eval_dir / "qa_dataset.json").read_text(encoding="utf-8"))
