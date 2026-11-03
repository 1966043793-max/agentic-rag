"""用 LangGraph 组装 Agentic RAG 状态图。

流程：
    START → load_memory → route ─(analytics)→ plan → agent ⇄ tools
                                    └(其他)→ agent ⇄ tools
                                                    ↓
                                                finalize → END

Agent 节点内部用原生 OpenAI function-calling 做工具循环：
LLM 决定调用哪个工具 → tools 节点执行 → 回到 agent 继续，直到给出最终答案。
"""
from __future__ import annotations

import json
from typing import Any

from langgraph.graph import END, START, StateGraph

from ..config import settings
from ..llm import llm
from ..observability.tracer import tracer
from .memory import Memory
from .planner import plan_steps
from .router import route_query
from .state import AgentState
from .tools import execute_tool, tools_for

_SYSTEM_PROMPT = (
    "你是电商平台的智能客服与运营分析助手。你可以：\n"
    "1. 检索知识库中的政策文档（退货/售后/配送/会员）回答规则类问题；\n"
    "2. 查询订单、商品、客户、退款等结构化数据；\n"
    "3. 组合多个工具完成多步分析。\n"
    "要求：回答基于工具返回的真实数据，不要编造；引用政策时注明出处标题；"
    "涉及具体订单/客户时要先调用工具查询；用简洁中文回答。"
)


def _system_with_memory(memory: dict[str, Any]) -> str:
    parts = [_SYSTEM_PROMPT]
    prefs = memory.get("preferences") or {}
    facts = memory.get("facts") or []
    if prefs or facts:
        parts.append("\n用户画像（长期记忆）：")
        if prefs:
            parts.append("偏好：" + json.dumps(prefs, ensure_ascii=False))
        if facts:
            parts.append("已知事实：" + "；".join(facts))
    return "\n".join(parts)


def _assistant_msg(content: str, tool_calls: list[dict] | None) -> dict:
    msg: dict[str, Any] = {"role": "assistant", "content": content or None}
    if tool_calls:
        msg["tool_calls"] = [
            {
                "id": t["id"],
                "type": "function",
                "function": {"name": t["name"], "arguments": t["arguments"]},
            }
            for t in tool_calls
        ]
    return msg


def load_memory_node(state: AgentState) -> dict:
    memory = Memory(state.get("user_id", "default")).load()
    system = _system_with_memory(memory)
    return {"memory": memory, "messages": [{"role": "system", "content": system},
                                            {"role": "user", "content": state["query"]}]}


def route_node(state: AgentState) -> dict:
    with tracer.span("router", input=state["query"]) as sp:
        result = route_query(state["query"])
        sp.record(output=result)
    return {"route": result["route"]}


def plan_node(state: AgentState) -> dict:
    with tracer.span("planner", input=state["query"]) as sp:
        steps = plan_steps(state["query"], state.get("memory"))
        sp.record(output={"steps": steps})
    return {"plan": steps}


def agent_node(state: AgentState) -> dict:
    route = state.get("route", "chitchat")
    iteration = state.get("iteration", 0)
    messages = state["messages"]

    force_answer = iteration >= settings.max_iterations
    tools = [] if force_answer else tools_for(route)

    if force_answer:
        messages = messages + [{"role": "system", "content": "已达最大步数，请基于已有信息直接给出最终答案。"}]
    elif state.get("plan"):
        plan_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(state["plan"]))
        messages = messages + [{"role": "system", "content": f"请按以下计划逐步调用工具执行：\n{plan_text}"}]

    with tracer.span("agent", input={"route": route, "iteration": iteration}) as sp:
        resp = llm.generate(messages, tools=tools or None)
        usage = resp.get("usage") or {}
        sp.record(
            output={
                "content": (resp["content"] or "")[:300],
                "tool_calls": [t["name"] for t in (resp.get("tool_calls") or [])],
            },
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    tool_calls = resp.get("tool_calls")
    assistant = _assistant_msg(resp["content"], tool_calls)
    if tool_calls:
        return {
            "messages": messages + [assistant],
            "pending_tool_calls": tool_calls,
            "iteration": iteration + 1,
        }
    return {"messages": messages + [assistant], "final_answer": resp["content"],
            "iteration": iteration + 1}


def tools_node(state: AgentState) -> dict:
    calls = state.get("pending_tool_calls", [])
    tool_msgs: list[dict] = []
    for call in calls:
        name = call["name"]
        try:
            args = json.loads(call["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {}
        with tracer.span("tool", input={"name": name, "args": args}) as sp:
            result = execute_tool(name, args)
            sp.record(output=result[:800])
        tool_msgs.append({"role": "tool", "tool_call_id": call["id"], "content": result})
    return {"messages": state["messages"] + tool_msgs, "pending_tool_calls": []}


def _synthesize(state: AgentState) -> str:
    """从消息中抽取信息，生成最终答案（用于超步数/异常兜底）。"""
    parts: list[str] = []
    for m in state["messages"]:
        role = m.get("role")
        if role == "tool":
            parts.append("【工具返回】" + (m.get("content") or ""))
        elif role == "user":
            parts.append("【用户问题】" + (m.get("content") or ""))
    context = "\n".join(parts)
    resp = llm.generate(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"请基于以下信息给出最终答案：\n\n{context}"},
        ]
    )
    return resp["content"] or "抱歉，我暂时无法回答这个问题。"


def finalize_node(state: AgentState) -> dict:
    answer = state.get("final_answer", "")
    if not answer:
        answer = _synthesize(state)

    memory = Memory(state.get("user_id", "default"))
    extracted = memory.extract(state["messages"])
    memory.save(extracted)

    with tracer.span("finalize", input={"answer": answer[:200]}) as sp:
        sp.record(output={"memory_updated": True})
    return {"final_answer": answer, "memory": extracted}


def _after_route(state: AgentState) -> str:
    return "plan" if state.get("route") == "analytics" else "agent"


def _after_agent(state: AgentState) -> str:
    if state.get("final_answer"):
        return "finalize"
    if state.get("iteration", 0) >= settings.max_iterations:
        return "finalize"
    return "tools"


def _after_tools(state: AgentState) -> str:
    if state.get("iteration", 0) >= settings.max_iterations:
        return "finalize"
    return "agent"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("load_memory", load_memory_node)
    g.add_node("route", route_node)
    g.add_node("plan", plan_node)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "load_memory")
    g.add_edge("load_memory", "route")
    g.add_conditional_edges("route", _after_route, {"plan": "plan", "agent": "agent"})
    g.add_edge("plan", "agent")
    g.add_conditional_edges("agent", _after_agent, {"tools": "tools", "finalize": "finalize"})
    g.add_conditional_edges("tools", _after_tools, {"agent": "agent", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()


_app = build_graph()


def run(query: str, user_id: str = "default") -> dict[str, Any]:
    """执行一次对话，返回 {answer, route, plan, trace_id}。"""
    if not settings.llm_ready:
        raise RuntimeError(
            "未配置 LLM，请先在 .env 中设置 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL"
        )
    trace_id = tracer.start_trace(query)
    initial: AgentState = {
        "query": query,
        "user_id": user_id,
        "messages": [],
        "route": "",
        "plan": [],
        "memory": {},
        "final_answer": "",
        "iteration": 0,
        "pending_tool_calls": [],
        "trace_id": trace_id,
    }
    result = _app.invoke(initial)
    return {
        "answer": result.get("final_answer", ""),
        "route": result.get("route", ""),
        "plan": result.get("plan", []),
        "trace_id": trace_id,
    }
