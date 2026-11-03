"""任务规划：把复杂分析问题拆解成可执行步骤。"""
from __future__ import annotations

from ..config import settings
from ..llm import llm

_PLAN_SYSTEM = (
    "你是任务规划器。把用户的复杂分析问题拆解成 2-5 个可执行步骤，"
    "每个步骤说明要用哪个工具、查什么。\n"
    "可用工具：search_knowledge(检索政策文档)、query_orders/get_order(订单)、"
    "get_customer(客户画像)、query_products(商品)、query_refunds(退款)、"
    "refund_rate_by_product(各商品退款率排名)、sales_by_category(品类销售额)。\n"
    '输出 JSON：{"steps": ["步骤1", "步骤2", ...]}'
)


def plan_steps(query: str, memory: dict | None = None) -> list[str]:
    """返回步骤列表；未配置 LLM 或失败时返回空列表（Agent 自行判断）。"""
    if not settings.llm_ready:
        return []
    try:
        data = llm.generate_json(
            [{"role": "system", "content": _PLAN_SYSTEM}, {"role": "user", "content": query}]
        )
        steps = data.get("steps", [])
        if isinstance(steps, list) and all(isinstance(s, str) for s in steps):
            return steps
    except Exception:
        pass
    return []
