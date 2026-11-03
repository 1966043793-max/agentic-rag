"""外部演示 MCP 服务：模拟「物流跟踪 + 库存」服务。

Agent 通过 MCP 客户端调用这里的 track_shipment / check_inventory，
演示「Agentic RAG 调用外部 MCP 工具」的能力。
运行：python -m agentic_rag.mcp.demo_server
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("logistics")

_LOGISTICS = {
    "ORD-2024-1001": {"status": "已签收", "carrier": "顺丰速运", "waybill": "SF1234567890", "detail": "2024-01-07 14:32 已签收"},
    "ORD-2024-1003": {"status": "运输中", "carrier": "京东物流", "waybill": "JD9876543210", "detail": "2024-03-04 09:15 到达上海分拨中心"},
    "ORD-2024-1008": {"status": "已揽收", "carrier": "中通快递", "waybill": "ZT5566778899", "detail": "2024-06-20 18:00 商家已发货"},
    "ORD-2024-1013": {"status": "运输中", "carrier": "顺丰速运", "waybill": "SF1122334455", "detail": "2024-09-06 11:20 运输中，预计明日送达"},
    "ORD-2024-1014": {"status": "待发货", "carrier": "待分配", "waybill": "-", "detail": "仓库备货中"},
}

_INVENTORY = {
    "P001": 120, "P002": 45, "P003": 78, "P004": 30, "P005": 200,
    "P006": 350, "P007": 500, "P008": 800, "P009": 150, "P010": 90,
    "P011": 300, "P012": 180,
}


@mcp.tool()
def track_shipment(order_id: str) -> str:
    """根据订单号查询物流轨迹（承运商、运单号、最新状态）。"""
    info = _LOGISTICS.get(order_id)
    if not info:
        return f"未找到订单 {order_id} 的物流信息。"
    return (
        f"订单 {order_id}：承运商 {info['carrier']}，运单号 {info['waybill']}，"
        f"状态「{info['status']}」，最新轨迹：{info['detail']}"
    )


@mcp.tool()
def check_inventory(product_id: str) -> str:
    """查询商品的实时库存。"""
    stock = _INVENTORY.get(product_id)
    if stock is None:
        return f"未找到商品 {product_id} 的库存信息。"
    return f"商品 {product_id} 当前库存：{stock} 件。"


if __name__ == "__main__":
    mcp.run(transport="stdio")
