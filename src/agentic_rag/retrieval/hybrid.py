"""混合检索：向量 + BM25 加权融合，可选 LLM 重排。"""
from __future__ import annotations

import json

import chromadb

from ..config import settings
from ..llm import embedder
from ..schema import Chunk, RetrievalResult
from .bm25 import BM25Index
from .reranker import rerank as rerank_docs


class VectorStore:
    """Chroma 向量存储的薄封装。"""

    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.collection = self.client.get_or_create_collection(
            "policies", metadata={"hnsw:space": "cosine"}
        )

    def search(self, query_embedding: list[float], top_k: int) -> list[tuple[str, float]]:
        """返回 (chunk_id, 相似度) 列表。cosine 空间下 distance = 1 - similarity。"""
        res = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        ids = res.get("ids", [[]])[0]
        distances = res.get("distances", [[]])[0]
        return [(cid, 1.0 - dist) for cid, dist in zip(ids, distances)]


def _load_chunks() -> list[Chunk]:
    path = settings.data_dir / "chunks.json"
    if not path.exists():
        raise RuntimeError("未找到 chunks.json，请先运行：python -m agentic_rag.cli ingest")
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk(**item) for item in data]


class HybridRetriever:
    """向量 + BM25 融合检索器。"""

    def __init__(self, alpha: float | None = None) -> None:
        self.alpha = settings.hybrid_alpha if alpha is None else alpha
        self.chunks = _load_chunks()
        self._by_id = {c.chunk_id: c for c in self.chunks}
        self.vector = VectorStore()
        self.bm25 = BM25Index.load(settings.bm25_path)

    def _fuse(
        self, vec: list[tuple[str, float]], bm: list[tuple[str, float]]
    ) -> list[RetrievalResult]:
        """对两类分数做归一化加权融合。"""
        vmax = max((s for _, s in vec), default=1.0) or 1.0
        bmax = max((s for _, s in bm), default=1.0) or 1.0

        scores: dict[str, float] = {}
        for cid, s in vec:
            scores[cid] = scores.get(cid, 0.0) + (1 - self.alpha) * (s / vmax)
        for cid, s in bm:
            scores[cid] = scores.get(cid, 0.0) + self.alpha * (s / bmax)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [
            RetrievalResult(chunk=self._by_id[cid], score=s, source="hybrid")
            for cid, s in ranked
        ]

    def search(
        self, query: str, top_k: int | None = None, rerank: bool | None = None
    ) -> list[RetrievalResult]:
        top_k = top_k or settings.top_k
        do_rerank = settings.rerank if rerank is None else rerank
        fetch_k = max(settings.bm25_top_k, top_k)

        vec = self.vector.search(embedder.embed_query(query), top_k=fetch_k)
        bm = self.bm25.search(query, top_k=fetch_k)
        results = self._fuse(vec, bm)

        if do_rerank and settings.llm_ready:
            results = rerank_docs(query, results, top_n=max(settings.rerank_top_n, top_k))
        return results[:top_k]


# 模块级单例
retriever = HybridRetriever()
