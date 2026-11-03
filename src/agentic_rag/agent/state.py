"""LangGraph 的 Agent 状态定义。"""
from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """Agent 在图各节点间传递的状态。

    messages 使用原生 OpenAI 消息格式（role/content/tool_calls），
    不依赖 LangChain 的消息抽象。
    """

    query: str
    user_id: str
    messages: list[dict[str, Any]]
    route: str            # rag / data / analytics / chitchat
    plan: list[str]       # analytics 场景下的分步计划
    memory: dict[str, Any]
    final_answer: str
    iteration: int
    pending_tool_calls: list[dict[str, Any]]
    trace_id: str
