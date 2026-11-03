"""FastAPI 服务：REST + SSE 流式接口。"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .agent.graph import run

app = FastAPI(title="Agentic RAG E-commerce", version="0.1.0")


class ChatRequest(BaseModel):
    query: str
    user_id: str = "default"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    """同步对话，返回 {answer, route, plan, trace_id}。"""
    return run(req.query, req.user_id)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """SSE 流式：依次下发 route / answer / trace 事件。"""

    def gen():
        result = run(req.query, req.user_id)
        yield {"event": "route", "data": result["route"]}
        yield {"event": "answer", "data": result["answer"]}
        yield {"event": "trace", "data": result["trace_id"]}

    return EventSourceResponse(gen())
