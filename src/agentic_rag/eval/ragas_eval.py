"""RAGAS 评测（可选，LLM-as-judge）：答案忠实性 / 相关性 / 上下文质量。

依赖：pip install ragas langchain-openai
注意：ragas 版本迭代快，本模块做了防御式降级，安装失败或版本不匹配时
由 CLI 捕获并打印提示，不影响主流程。
"""
from __future__ import annotations

from ..config import settings
from ..llm import llm
from ..retrieval.hybrid import retriever
from .dataset import load_qa_dataset


def _judge_llm():
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(
        ChatOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )
    )


def _judge_embeddings():
    from langchain_openai import OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
        )
    )


def _generate_answer(question: str, contexts: list[str]) -> str:
    context = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    resp = llm.generate(
        [
            {"role": "system", "content": "基于给定上下文回答用户问题，并标注来源。"},
            {"role": "user", "content": f"上下文：\n{context}\n\n问题：{question}"},
        ]
    )
    return resp["content"] or ""


def evaluate_ragas() -> dict:
    """跑 RAGAS 评测，返回指标 DataFrame（转 dict）。"""
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    llm_judge = _judge_llm()
    embeddings_judge = _judge_embeddings()

    samples = []
    for item in load_qa_dataset():
        question = item["question"]
        results = retriever.search(question, top_k=settings.top_k, rerank=False)
        contexts = [r.chunk.text for r in results]
        answer = _generate_answer(question, contexts)
        samples.append(
            SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
                reference=item.get("ground_truth", ""),
            )
        )

    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm_judge,
        embeddings=embeddings_judge,
    )
    return result.to_pandas().to_dict(orient="records")
