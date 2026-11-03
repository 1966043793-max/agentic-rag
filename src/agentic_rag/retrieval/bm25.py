"""BM25 关键词检索。

中文文本没有天然空格分词，这里用「字符二元组」做轻量分词，
生产环境可替换为 jieba / IK 分词等，接口不变。
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from ..schema import Chunk

_CJK = r"[一-鿿]+"


def tokenize(text: str) -> list[str]:
    """中文按字符二元组，英文/数字按词切分。"""
    tokens: list[str] = []
    for run in re.findall(_CJK, text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens += [run[i : i + 2] for i in range(len(run) - 1)]
    tokens += [w.lower() for w in re.findall(r"[a-zA-Z0-9]+", text)]
    return tokens


class BM25Index:
    """基于 rank-bm25 的关键词索引。"""

    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self._chunks = chunks or []
        self._tokens: list[list[str]] = []
        self._index: BM25Okapi | None = None

    def build(self) -> None:
        self._tokens = [tokenize(c.text) for c in self._chunks]
        self._index = BM25Okapi(self._tokens)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """返回 (chunk_id, bm25 分数) 列表，按分数降序。"""
        if self._index is None:
            self.build()
        scores = self._index.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self._chunks[i].chunk_id, float(scores[i])) for i in ranked]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunks": self._chunks, "tokens": self._tokens, "index": self._index}, f)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj._chunks = data["chunks"]
        obj._tokens = data["tokens"]
        obj._index = data["index"]
        return obj
