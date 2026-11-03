"""查询路由：判断 query 应走哪条处理路径（rag / data / analytics / chitchat）。

LLM 判定为主，关键词启发式兜底，保证未配置 LLM 时也能得到合理路由。
"""
from __future__ import annotations

import re

from ..config import settings
from ..llm import llm

_VALID_ROUTES = {"rag", "data", "analytics", "chitchat"}

_ROUTE_SYSTEM = (
    "你是电商智能助手的路由器。把用户问题归入以下四类之一：\n"
    "- rag：询问政策、规则、流程、权益等文档内容（退货/售后/配送/会员政策）\n"
    "- data：查询单个结构化数据（订单、商品、客户、退款）\n"
    "- analytics：需要多步分析或聚合统计（排名、趋势、对比、归因、占比）\n"
    "- chitchat：闲聊、问候或与业务无关\n"
    '输出 JSON：{"route": "rag|data|analytics|chitchat", "reason": "一句话原因"}'
)


def heuristic_route(query: str) -> str:
    """关键词启发式路由，作为 LLM 失败时的兜底。"""
    if re.search(r"政策|退货|售后|保修|配送|运费|会员|积分|权益|规则|流程|无理由", query):
        return "rag"
    if re.search(r"排名|最高|最低|对比|趋势|统计|退款率|占比|哪些|分析|多少|原因", query):
        return "analytics"
    if re.search(r"订单|商品|客户|退款|价格|库存|下单|购买|买了", query):
        return "data"
    return "chitchat"


def route_query(query: str) -> dict:
    """返回 {"route": ..., "reason": ...}。"""
    if settings.llm_ready:
        try:
            data = llm.generate_json(
                [{"role": "system", "content": _ROUTE_SYSTEM}, {"role": "user", "content": query}]
            )
            route = data.get("route")
            if route in _VALID_ROUTES:
                return {"route": route, "reason": data.get("reason", "")}
        except Exception:
            pass
    route = heuristic_route(query)
    return {"route": route, "reason": "heuristic-fallback"}
