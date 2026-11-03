"""把电商知识库暴露成 MCP 服务，供 Claude / Cursor 等 MCP 客户端调用。

运行：
  python -m agentic_rag.mcp.server                    # stdio（Claude Desktop / Cursor）
  python -m agentic_rag.mcp.server --transport streamable-http --port 9000
"""
from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from ..data_store import store
from ..retrieval.hybrid import retriever

mcp = FastMCP("agentic-rag-ecommerce")


def _json(data) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


@mcp.tool()
def search_knowledge(query: str) -> str:
    """在电商政策知识库中检索（退货/售后/配送/会员政策），返回相关片段及出处。"""
    results = retriever.search(query)
    items = [
        {"title": r.chunk.title, "doc_id": r.chunk.doc_id, "text": r.chunk.text[:300]}
        for r in results
    ]
    return _json({"results": items})


@mcp.tool()
def query_orders(customer_id: str = "", status: str = "") -> str:
    """按客户和/或订单状态查询订单。"""
    orders = store.list_orders(customer_id=customer_id or None, status=status or None)
    return _json({"orders": [o.model_dump() for o in orders], "count": len(orders)})


@mcp.tool()
def get_product(product_id: str) -> str:
    """根据商品ID查询商品信息。"""
    p = store.get_product(product_id)
    return _json({"product": p.model_dump() if p else None})


@mcp.tool()
def refund_rate_by_product() -> str:
    """统计各商品的退款率排名。"""
    return _json({"refund_rate": store.refund_rate_by_product()})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="streamable-http", host="127.0.0.1", port=args.port)
