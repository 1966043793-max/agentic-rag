"""轻量可观测性：基于 SQLite 的 Span 追踪器。

记录 Agent 每一步（路由、规划、LLM 调用、工具调用）的输入/输出、
Token 用量与耗时，支持事后按 trace 还原整条链路。
生产环境可替换为 Langfuse / OpenTelemetry，接口保持一致。
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator

from ..config import settings

_current_trace: ContextVar[str | None] = ContextVar("trace_id", default=None)


class Span:
    def __init__(self, span_id: str, trace_id: str, name: str, input_data: Any) -> None:
        self.span_id = span_id
        self.trace_id = trace_id
        self.name = name
        self.input_json = _safe_json(input_data)
        self.output_json = ""
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def record(
        self,
        output: Any = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        self.output_json = _safe_json(output)
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


def _safe_json(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, default=str)
    except Exception:
        return str(data)


class Tracer:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path or settings.trace_db)
        self._init()

    def _init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS spans(
                    span_id TEXT PRIMARY KEY,
                    trace_id TEXT,
                    name TEXT,
                    input_json TEXT,
                    output_json TEXT,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    latency_ms INTEGER,
                    created_at REAL
                )
                """
            )

    def start_trace(self, query: str) -> str:
        trace_id = uuid.uuid4().hex[:16]
        _current_trace.set(trace_id)
        span_id = uuid.uuid4().hex[:16]
        self._insert(Span(span_id, trace_id, "request", query), 0)
        return trace_id

    @contextmanager
    def span(self, name: str, input: Any = None) -> Iterator[Span]:
        trace_id = _current_trace.get() or self.start_trace(str(input))
        sp = Span(uuid.uuid4().hex[:16], trace_id, name, input)
        started = time.time()
        try:
            yield sp
        finally:
            latency_ms = int((time.time() - started) * 1000)
            self._insert(sp, latency_ms)

    def _insert(self, sp: Span, latency_ms: int) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO spans(span_id, trace_id, name, input_json, output_json,
                                  prompt_tokens, completion_tokens, latency_ms, created_at)
                VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    sp.span_id,
                    sp.trace_id,
                    sp.name,
                    sp.input_json,
                    sp.output_json,
                    sp.prompt_tokens,
                    sp.completion_tokens,
                    latency_ms,
                    time.time(),
                ),
            )

    # ---- 查询 ----
    def list_traces(self, limit: int = 20) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT trace_id,
                       MAX(created_at) - MIN(created_at) AS duration_ms,
                       COUNT(*) AS span_count,
                       SUM(prompt_tokens) AS prompt_tokens,
                       SUM(completion_tokens) AS completion_tokens
                FROM spans
                GROUP BY trace_id
                ORDER BY MIN(created_at) DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_trace(self, trace_id: str) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM spans WHERE trace_id=? ORDER BY created_at", (trace_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def format_trace(self, trace_id: str) -> str:
        """把一条 trace 渲染成易读的树状文本。"""
        spans = self.get_trace(trace_id)
        if not spans:
            return f"未找到 trace: {trace_id}"
        lines = [f"Trace {trace_id}"]
        for s in spans:
            latency = s["latency_ms"] or 0
            tokens = (s["prompt_tokens"] or 0) + (s["completion_tokens"] or 0)
            name = s["name"]
            detail = _preview(s["output_json"] or "")
            lines.append(f"  ├─ {name}  ({latency}ms, {tokens} tokens)")
            if detail:
                lines.append(f"  │    └─ {detail}")
        return "\n".join(lines)


def _preview(text: str, n: int = 120) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= n else text[:n] + "…"


tracer = Tracer()
