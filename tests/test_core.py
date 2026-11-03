"""无需外部 API 的单元测试。运行：pytest"""
from __future__ import annotations


def test_tokenize_chinese_bigram():
    from agentic_rag.retrieval.bm25 import tokenize

    tokens = tokenize("退货政策")
    assert "退货" in tokens
    assert "货政" in tokens


def test_split_chunks_keeps_doc_id():
    from agentic_rag.ingest.chunker import split_chunks
    from agentic_rag.schema import Document

    doc = Document(doc_id="D1", title="t", content="段落一。\n\n段落二。\n\n段落三。")
    chunks = split_chunks(doc, max_chars=100)
    assert len(chunks) >= 1
    assert all(c.doc_id == "D1" for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_data_store_refund_rate():
    from agentic_rag.data_store import DataStore

    store = DataStore()
    rows = store.refund_rate_by_product()
    assert isinstance(rows, list)
    assert rows, "应至少有一个售出商品"
    assert all("refund_rate" in r for r in rows)
