"""文档与检索相关的通用数据结构。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Document:
    """一份原始文档（如一篇政策）。"""

    doc_id: str
    title: str
    content: str


@dataclass
class Chunk:
    """文档切分后的一段文本。"""

    chunk_id: str
    doc_id: str
    title: str
    text: str


@dataclass
class RetrievalResult:
    """一条检索结果，带得分与来源。"""

    chunk: Chunk
    score: float
    source: str = "hybrid"  # vector / bm25 / hybrid
