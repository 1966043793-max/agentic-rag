"""命令行入口。

用法：
  python -m agentic_rag.cli ingest             # 建索引
  python -m agentic_rag.cli chat               # 交互问答
  python -m agentic_rag.cli chat --mcp         # 接入外部 MCP 工具
  python -m agentic_rag.cli search "退货政策"   # 直接检索
  python -m agentic_rag.cli eval               # 检索评测
  python -m agentic_rag.cli eval --ragas       # + RAGAS 评测
  python -m agentic_rag.cli traces             # 查看追踪
  python -m agentic_rag.cli serve              # 启动 FastAPI
"""
from __future__ import annotations

import argparse
import sys


def _cmd_ingest(args: argparse.Namespace) -> int:
    from .ingest.indexer import run_indexing

    chunks = run_indexing()
    print(f"索引构建完成，共 {len(chunks)} 个 chunk。")
    return 0


def _do_chat(query: str, user_id: str) -> None:
    from .agent.graph import run

    result = run(query, user_id)
    print(f"\n[{result['route']}] {result['answer']}")
    if result.get("plan"):
        print("\n执行计划：")
        for i, s in enumerate(result["plan"]):
            print(f"  {i + 1}. {s}")
    print(f"(trace_id: {result['trace_id']})\n")


def _cmd_chat(args: argparse.Namespace) -> int:
    if args.mcp:
        from .mcp.client import connect_demo_server

        try:
            connect_demo_server()
            print("已连接外部 MCP 服务（物流跟踪）。\n")
        except Exception as e:  # noqa: BLE001
            print(f"MCP 连接失败：{e}\n")

    if args.query:
        try:
            _do_chat(args.query, args.user)
        except RuntimeError as e:
            print(f"错误：{e}", file=sys.stderr)
            return 1
        return 0

    print("进入交互模式，输入 exit 退出。")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        try:
            _do_chat(q, args.user)
        except RuntimeError as e:
            print(f"错误：{e}", file=sys.stderr)
            return 1
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    from .retrieval.hybrid import retriever

    results = retriever.search(args.query, top_k=args.top_k)
    for r in results:
        print(f"[{r.score:.3f}] ({r.chunk.doc_id}) {r.chunk.title}")
        print(f"  {r.chunk.text[:120]}\n")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .eval.retrieval_eval import evaluate_retrieval, print_report

    print_report(evaluate_retrieval(k=args.k))

    if args.ragas:
        from .eval.ragas_eval import evaluate_ragas

        try:
            rows = evaluate_ragas()
            print("\nRAGAS 评测结果：")
            for row in rows:
                parts = []
                for k, v in row.items():
                    parts.append(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}")
                print("  " + "  ".join(parts))
        except Exception as e:  # noqa: BLE001
            print(f"\nRAGAS 评测跳过：{e}")
    return 0


def _cmd_traces(args: argparse.Namespace) -> int:
    from .observability.tracer import tracer

    traces = tracer.list_traces(limit=args.limit)
    print(f"{'trace_id':<18}{'spans':>6}{'p_tok':>8}{'c_tok':>8}{'dur(ms)':>9}")
    for t in traces:
        print(
            f"{t['trace_id']:<18}{t['span_count']:>6}"
            f"{(t['prompt_tokens'] or 0):>8}{(t['completion_tokens'] or 0):>8}"
            f"{int(t['duration_ms'] or 0):>9}"
        )
    if args.trace_id:
        print("\n" + tracer.format_trace(args.trace_id))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .config import settings

    uvicorn.run("agentic_rag.api:app", host=settings.host, port=settings.port)
    return 0


def _cmd_mcp_server(args: argparse.Namespace) -> int:
    from .mcp.server import mcp

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="streamable-http", host="127.0.0.1", port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentic-rag", description="Agentic RAG 电商智能助手")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="构建索引")
    p.set_defaults(func=_cmd_ingest)

    p = sub.add_parser("chat", help="对话")
    p.add_argument("query", nargs="?", help="单次提问；省略则进入交互模式")
    p.add_argument("--user", default="default", help="用户ID（用于长期记忆）")
    p.add_argument("--mcp", action="store_true", help="接入外部 MCP 服务（物流跟踪演示）")
    p.set_defaults(func=_cmd_chat)

    p = sub.add_parser("search", help="直接检索")
    p.add_argument("query")
    p.add_argument("--top-k", type=int, default=5)
    p.set_defaults(func=_cmd_search)

    p = sub.add_parser("eval", help="评测")
    p.add_argument("--k", type=int, default=3, help="检索 Top-K")
    p.add_argument("--ragas", action="store_true", help="同时跑 RAGAS 评测（需额外依赖）")
    p.set_defaults(func=_cmd_eval)

    p = sub.add_parser("traces", help="查看追踪")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--trace-id", help="查看某条 trace 的详情")
    p.set_defaults(func=_cmd_traces)

    p = sub.add_parser("serve", help="启动 FastAPI 服务")
    p.set_defaults(func=_cmd_serve)

    p = sub.add_parser("mcp-server", help="以 MCP 服务端方式暴露知识库")
    p.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    p.add_argument("--port", type=int, default=9000)
    p.set_defaults(func=_cmd_mcp_server)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
