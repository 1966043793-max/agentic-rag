"""内存数据仓库：加载种子数据，提供结构化查询与聚合。

演示用内存数据集模拟数据库。生产环境可替换为真实 DB（PostgreSQL 等），
只需保持本类的查询接口不变即可。
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import settings
from .models import Customer, Order, Product, Refund


class DataStore:
    """商品 / 订单 / 客户 / 退货四类结构化数据的内存仓库。"""

    def __init__(self, seed_dir: Path | None = None) -> None:
        seed_dir = seed_dir or settings.seed_dir
        self.products: list[Product] = self._load(seed_dir / "products.json", Product)
        self.orders: list[Order] = self._load(seed_dir / "orders.json", Order)
        self.customers: list[Customer] = self._load(seed_dir / "customers.json", Customer)
        self.refunds: list[Refund] = self._load(seed_dir / "refunds.json", Refund)

    @staticmethod
    def _load(path: Path, model):
        return [model(**item) for item in json.loads(path.read_text(encoding="utf-8"))]

    # ---- 单点查询 ----
    def get_product(self, product_id: str) -> Product | None:
        return next((p for p in self.products if p.product_id == product_id), None)

    def get_order(self, order_id: str) -> Order | None:
        return next((o for o in self.orders if o.order_id == order_id), None)

    def get_customer(self, customer_id: str) -> Customer | None:
        return next((c for c in self.customers if c.customer_id == customer_id), None)

    # ---- 列表查询 ----
    def list_orders(self, customer_id: str | None = None, status: str | None = None) -> list[Order]:
        result = self.orders
        if customer_id:
            result = [o for o in result if o.customer_id == customer_id]
        if status:
            result = [o for o in result if o.status == status]
        return result

    def list_products(self, category: str | None = None, keyword: str | None = None) -> list[Product]:
        result = self.products
        if category:
            result = [p for p in result if p.category == category]
        if keyword:
            result = [p for p in result if keyword in p.name or keyword in p.brand]
        return result

    def list_refunds(
        self, product_id: str | None = None, customer_id: str | None = None
    ) -> list[Refund]:
        result = self.refunds
        if product_id:
            result = [r for r in result if r.product_id == product_id]
        if customer_id:
            result = [r for r in result if r.customer_id == customer_id]
        return result

    # ---- 聚合分析 ----
    def refund_rate_by_product(self) -> list[dict]:
        """统计每个商品的退款率 = 退款单数 / 售出件数。"""
        refund_count: dict[str, int] = {}
        for r in self.refunds:
            refund_count[r.product_id] = refund_count.get(r.product_id, 0) + 1

        sold_count: dict[str, int] = {}
        for o in self.orders:
            if o.status == "cancelled":
                continue
            for item in o.items:
                sold_count[item.product_id] = sold_count.get(item.product_id, 0) + item.quantity

        rows = []
        for pid, sold in sold_count.items():
            rows.append(
                {
                    "product_id": pid,
                    "product_name": (self.get_product(pid).name if self.get_product(pid) else pid),
                    "sold": sold,
                    "refunds": refund_count.get(pid, 0),
                    "refund_rate": round(refund_count.get(pid, 0) / sold, 4),
                }
            )
        return sorted(rows, key=lambda x: x["refund_rate"], reverse=True)

    def sales_by_category(self) -> list[dict]:
        """按品类统计销售额。"""
        totals: dict[str, float] = {}
        for o in self.orders:
            if o.status == "cancelled":
                continue
            for item in o.items:
                p = self.get_product(item.product_id)
                cat = p.category if p else "未知"
                totals[cat] = totals.get(cat, 0.0) + item.unit_price * item.quantity
        return [{"category": k, "sales": round(v, 2)} for k, v in
                sorted(totals.items(), key=lambda x: x[1], reverse=True)]


# 模块级单例，供工具层直接使用
store = DataStore()
