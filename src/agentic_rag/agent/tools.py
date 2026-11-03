"""工具集：OpenAI function-calling 格式的工具定义与执行分发。

工具分两类：
1. 内置工具 —— 结构化数据查询（DataStore）与 RAG 检索（HybridRetriever）
2. MCP 工具 —— 由外部 MCP 客户端动态注册
"""
from __future__ import annotations

import json
from typing import Any

from ..config import settings
from ..data_store import store

# MCP 客户端注册表（由 mcp.client 注入，避免循环 import）
_registered_mcp: list[Any] = []


def register_mcp_client(client: Any) -> None:
    _registered_mcp.append(client)


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


# ---- 内置工具实现 ----
def _search_knowledge(query: str) -> str:
    from ..retrieval.hybrid import retriever

    results = retriever.search(query, top_k=settings.top_k)
    items = [
        {
            "doc_id": r.chunk.doc_id,
            "title": r.chunk.title,
            "text": r.chunk.text[:300],
            "score": round(r.score, 4),
        }
        for r in results
    ]
    return _json({"results": items})


def _query_orders(customer_id: str | None = None, status: str | None = None) -> str:
    orders = store.list_orders(customer_id=customer_id or None, status=status or None)
    return _json({"orders": [o.model_dump() for o in orders], "count": len(orders)})


def _get_order(order_id: str) -> str:
    o = store.get_order(order_id)
    return _json({"order": o.model_dump() if o else None})


def _get_customer(customer_id: str) -> str:
    c = store.get_customer(customer_id)
    return _json({"customer": c.model_dump() if c else None})


def _query_products(category: str | None = None, keyword: str | None = None) -> str:
    products = store.list_products(category=category or None, keyword=keyword or None)
    return _json({"products": [p.model_dump() for p in products], "count": len(products)})


def _query_refunds(product_id: str | None = None, customer_id: str | None = None) -> str:
    refunds = store.list_refunds(product_id=product_id or None, customer_id=customer_id or None)
    return _json({"refunds": [r.model_dump() for r in refunds], "count": len(refunds)})


def _refund_rate_by_product() -> str:
    return _json({"refund_rate": store.refund_rate_by_product()})


def _sales_by_category() -> str:
    return _json({"sales_by_category": store.sales_by_category()})


# ---- 工具注册表（OpenAI function schema）----
ALL_TOOLS: dict[str, dict[str, Any]] = {
    "search_knowledge": {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "在知识库中检索政策类文档（退货/售后/配送/会员政策），返回相关片段及出处。适合回答规则、流程、权益类问题。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "检索查询词"}},
                "required": ["query"],
            },
        },
    },
    "query_orders": {
        "type": "function",
        "function": {
            "name": "query_orders",
            "description": "按客户和/或订单状态查询订单列表。",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "客户ID，如 C001"},
                    "status": {
                        "type": "string",
                        "enum": ["pending", "paid", "shipped", "delivered", "refunding", "refunded", "cancelled"],
                        "description": "订单状态",
                    },
                },
            },
        },
    },
    "get_order": {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "根据订单号查询单个订单详情。",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "订单号，如 ORD-2024-1001"}},
                "required": ["order_id"],
            },
        },
    },
    "get_customer": {
        "type": "function",
        "function": {
            "name": "get_customer",
            "description": "根据客户ID查询客户画像（等级、标签、偏好）。",
            "parameters": {
                "type": "object",
                "properties": {"customer_id": {"type": "string", "description": "客户ID，如 C001"}},
                "required": ["customer_id"],
            },
        },
    },
    "query_products": {
        "type": "function",
        "function": {
            "name": "query_products",
            "description": "按品类或关键词查询商品。",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "品类，如 手机/家电/服饰/食品/数码/美妆/家居"},
                    "keyword": {"type": "string", "description": "商品名或品牌关键词"},
                },
            },
        },
    },
    "query_refunds": {
        "type": "function",
        "function": {
            "name": "query_refunds",
            "description": "按商品或客户查询退款/退货记录。",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string", "description": "商品ID，如 P001"},
                    "customer_id": {"type": "string", "description": "客户ID，如 C001"},
                },
            },
        },
    },
    "refund_rate_by_product": {
        "type": "function",
        "function": {
            "name": "refund_rate_by_product",
            "description": "统计各商品的退款率排名（退款单数/售出件数）。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    "sales_by_category": {
        "type": "function",
        "function": {
            "name": "sales_by_category",
            "description": "按品类统计销售额。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
}

_ROUTE_TOOLS: dict[str, list[str]] = {
    "rag": ["search_knowledge"],
    "data": ["query_orders", "get_order", "get_customer", "query_products", "query_refunds"],
    "analytics": [
        "search_knowledge",
        "query_orders",
        "get_order",
        "get_customer",
        "query_products",
        "query_refunds",
        "refund_rate_by_product",
        "sales_by_category",
    ],
    "chitchat": [],
}

_EXECUTORS: dict[str, Any] = {
    "search_knowledge": _search_knowledge,
    "query_orders": _query_orders,
    "get_order": _get_order,
    "get_customer": _get_customer,
    "query_products": _query_products,
    "query_refunds": _query_refunds,
    "refund_rate_by_product": _refund_rate_by_product,
    "sales_by_category": _sales_by_category,
}


def tools_for(route: str) -> list[dict[str, Any]]:
    """返回某路由下 Agent 可用的工具 schema（含已注册的 MCP 工具）。"""
    names = _ROUTE_TOOLS.get(route, [])
    schemas = [ALL_TOOLS[n] for n in names if n in ALL_TOOLS]
    if route != "chitchat":
        for client in _registered_mcp:
            schemas.extend(client.tool_schemas)
    return schemas


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """执行工具，内置优先，其次 MCP 工具。异常统一转成错误文本。"""
    try:
        for client in _registered_mcp:
            if name in client.tool_names:
                return client.call(name, arguments)
        fn = _EXECUTORS.get(name)
        if fn is None:
            return _json({"error": f"未知工具: {name}"})
        return fn(**arguments)
    except Exception as e:  # noqa: BLE001
        return _json({"error": f"工具 {name} 执行失败: {e}"})
