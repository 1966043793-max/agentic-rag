"""LLM 重排：对候选片段做相关性打分，取 Top-N。

用 LLM 而非 cross-encoder，好处是无需额外模型、走同一 OpenAI 兼容接口。
生产环境可替换为 bge-reranker 等专用重排模型。
"""
from __future__ import annotations

from ..config import settings
from ..llm import llm
from ..schema import RetrievalResult

_SYSTEM = (
    "你是检索相关性评估器。请判断每个候选片段与用户问题的相关程度，"
    "输出 0-10 的整数分数（0=完全不相关，10=高度相关）。只输出 JSON，不要解释。"
)


def rerank(
    query: str, results: list[RetrievalResult], top_n: int | None = None
) -> list[RetrievalResult]:
    top_n = top_n or settings.rerank_top_n
    if len(results) <= top_n:
        return results

    candidates = "\n\n".join(
        f"[{i}] 标题：{r.chunk.title}\n内容：{r.chunk.text[:300]}" for i, r in enumerate(results)
    )
    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": f"问题：{query}\n\n候选片段：\n{candidates}\n\n请输出 JSON：{{\"scores\": {{\"0\": 分数, \"1\": 分数, ...}}}}",
        },
    ]

    try:
        data = llm.generate_json(messages)
        scores = data.get("scores", {})
    except Exception:
        return results[:top_n]

    scored: list[tuple[float, RetrievalResult]] = []
    for i, r in enumerate(results):
        try:
            s = float(scores.get(str(i), 0.0))
        except (TypeError, ValueError):
            s = 0.0
        scored.append((s, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:top_n]]
