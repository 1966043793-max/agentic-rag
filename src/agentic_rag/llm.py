"""LLM 与 Embedding 的薄封装，统一走 OpenAI 兼容协议。

这里没有用 LangChain 的模型抽象，而是直接封装 openai SDK，
目的是把「调用 LLM」这件事的底层细节暴露出来，便于理解与讲解。
"""
from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from .config import settings


class ChatModel:
    """OpenAI 兼容的 chat 封装，支持工具调用与 JSON 模式。"""

    def __init__(self, model: str | None = None, temperature: float | None = None) -> None:
        self.client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
        self.model = model or settings.llm_model
        self.temperature = settings.temperature if temperature is None else temperature

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """调用 chat，返回 content / tool_calls / usage 的统一结构。"""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if tools:
            kwargs["tools"] = tools

        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message

        out: dict[str, Any] = {"content": msg.content or "", "tool_calls": None, "usage": None}
        if msg.tool_calls:
            out["tool_calls"] = [
                {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                for tc in msg.tool_calls
            ]
        if resp.usage:
            out["usage"] = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
        return out

    def generate_json(self, messages: list[dict[str, Any]], temperature: float | None = None) -> Any:
        """要求模型返回 JSON 对象（OpenAI 兼容的 json_object 模式）。"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content or "{}")


class EmbeddingModel:
    """OpenAI 兼容的 embedding 封装。"""

    def __init__(self, model: str | None = None) -> None:
        self.client = OpenAI(
            base_url=settings.embedding_base_url, api_key=settings.embedding_api_key
        )
        self.model = model or settings.embedding_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        # 按输入顺序排序（不同服务商返回顺序可能不一致）
        data = sorted(resp.data, key=lambda d: d.index)
        return [d.embedding for d in data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


llm = ChatModel()
embedder = EmbeddingModel()
