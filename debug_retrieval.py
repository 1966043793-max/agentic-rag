"""排查检索 miss：看某个 query 到底检索出了什么，正确条文排第几、分数多少。

用法：
    python debug_retrieval.py "查询内容" [期望的 doc_id]

示例：
    python debug_retrieval.py "网购商品几天内可以无理由退货？" crpl-25
"""
import sys

from agentic_rag.retrieval.hybrid import retriever

query = sys.argv[1]
expected = sys.argv[2] if len(sys.argv) > 2 else None

# 故意多拉一些，看正确条文排在第几位
results = retriever.search(query, top_k=20, rerank=False)

print(f"查询：{query}")
print(f"期望条文：{expected or '（未指定）'}")
print("-" * 70)

rank_of_expected = None
for i, r in enumerate(results, 1):
    mark = ""
    if expected and r.chunk.doc_id == expected:
        mark = "  ★ 命中正确条文"
        rank_of_expected = i
    print(f"第{i:2d}名  {r.chunk.doc_id:<10} 分数={r.score:.4f}  {r.chunk.title}{mark}")
    print(f"       {r.chunk.text[:100]}")

print("-" * 70)
if expected:
    if rank_of_expected:
        print(f"✓ 正确条文 {expected} 排在第 {rank_of_expected} 名")
    else:
        print(f"✗ 正确条文 {expected} 不在前 20 名（可能被 BM25/向量漏掉了）")
