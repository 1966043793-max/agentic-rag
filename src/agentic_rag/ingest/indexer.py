"""建索引：文档 → 分块 → 向量索引(Chroma) + 关键词索引(BM25) + 块登记表。"""
from __future__ import annotations

import json
from dataclasses import asdict

import chromadb

from ..config import settings
from ..llm import embedder
from ..retrieval.bm25 import BM25Index
from ..schema import Chunk
from .chunker import split_chunks
from .loader import load_policy_documents


def run_indexing() -> list[Chunk]:
    """执行一次完整索引构建，返回所有 chunk。"""
    docs = load_policy_documents()
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(split_chunks(doc))

    # 1) 块登记表（检索层的权威 chunk 列表）
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "chunks.json").write_text(
        json.dumps([asdict(c) for c in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 2) 向量索引
    if not settings.embedding_ready:
        raise RuntimeError(
            "未配置 embedding，请先在 .env 中设置 EMBEDDING_API_KEY / EMBEDDING_BASE_URL / EMBEDDING_MODEL"
        )
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    try:
        client.delete_collection("policies")
    except Exception:
        pass
    coll = client.get_or_create_collection("policies", metadata={"hnsw:space": "cosine"})
    vectors = embedder.embed([c.text for c in chunks])
    coll.add(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[{"doc_id": c.doc_id, "title": c.title} for c in chunks],
        embeddings=vectors,
    )

    # 3) BM25 关键词索引
    bm25 = BM25Index(chunks)
    bm25.build()
    bm25.save(settings.bm25_path)

    return chunks
