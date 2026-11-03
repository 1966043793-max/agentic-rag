"""检索评测：命中率 / MRR / 召回率，无需 LLM，可离线运行。"""
from __future__ import annotations

from ..retrieval.hybrid import retriever
from .dataset import load_retrieval_dataset


def evaluate_retrieval(dataset: list[dict] | None = None, k: int = 3) -> dict:
    """计算 Hit@k、MRR@k、Recall@k。"""
    dataset = dataset or load_retrieval_dataset()

    hit = 0
    mrr = 0.0
    recall_sum = 0.0
    details: list[dict] = []

    for item in dataset:
        query = item["query"]
        relevant = set(item["relevant_docs"])

        results = retriever.search(query, top_k=k, rerank=False)
        retrieved = []
        for r in results:
            if r.chunk.doc_id not in retrieved:
                retrieved.append(r.chunk.doc_id)

        is_hit = any(d in relevant for d in retrieved)

        rr = 0.0
        for i, d in enumerate(retrieved):
            if d in relevant:
                rr = 1.0 / (i + 1)
                break

        recall = len(set(retrieved) & relevant) / len(relevant) if relevant else 1.0

        hit += int(is_hit)
        mrr += rr
        recall_sum += recall
        details.append(
            {
                "query": query,
                "hit": is_hit,
                "rr": round(rr, 4),
                "retrieved": retrieved,
                "relevant": sorted(relevant),
            }
        )

    n = len(dataset)
    return {
        "hit_rate": round(hit / n, 4),
        "mrr": round(mrr / n, 4),
        "recall": round(recall_sum / n, 4),
        "details": details,
    }


def print_report(metrics: dict) -> None:
    print("=" * 60)
    print("检索评测结果")
    print("=" * 60)
    print(f"  Hit@{3}   : {metrics['hit_rate']:.2%}")
    print(f"  MRR@{3}   : {metrics['mrr']:.4f}")
    print(f"  Recall@{3}: {metrics['recall']:.2%}")
    print("-" * 60)
    for d in metrics["details"]:
        mark = "✓" if d["hit"] else "✗"
        print(f"  {mark} {d['query']}")
        print(f"        命中: {d['retrieved'][:3]}  相关: {d['relevant']}")
    print("=" * 60)
