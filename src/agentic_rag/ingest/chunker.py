"""文档分块：按段落聚合 + 超长内容按句切分。"""
from __future__ import annotations

import re

from ..schema import Chunk, Document


def _split_long(text: str, max_chars: int) -> list[str]:
    """把超长文本按句子边界切成不超过 max_chars 的片段。"""
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[。！？；])", text)
    parts: list[str] = []
    buf = ""
    for s in sentences:
        if buf and len(buf) + len(s) > max_chars:
            parts.append(buf)
            buf = s
        else:
            buf += s
    if buf:
        parts.append(buf)
    return parts


def split_chunks(doc: Document, max_chars: int = 500) -> list[Chunk]:
    """先把段落聚合成接近 max_chars 的块，再处理超长段落。"""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc.content) if p.strip()]

    grouped: list[str] = []
    buf = ""
    for para in paragraphs:
        if not buf:
            buf = para
        elif len(buf) + len(para) + 1 <= max_chars:
            buf += "\n" + para
        else:
            grouped.append(buf)
            buf = para
    if buf:
        grouped.append(buf)

    chunks: list[Chunk] = []
    for group in grouped:
        for sub in _split_long(group, max_chars):
            chunk_id = f"{doc.doc_id}#{len(chunks)}"
            chunks.append(Chunk(chunk_id=chunk_id, doc_id=doc.doc_id, title=doc.title, text=sub))
    return chunks
