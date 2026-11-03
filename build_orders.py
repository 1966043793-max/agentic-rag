"""一次性工具：把天猫订单 CSV 转换成项目 orders.json 的格式。

用法：python build_orders.py
说明：天猫订单是扁平单表（无商品明细、无客户ID），转换时：
  - order_id / total_amount / shipping_address / created_at  直接映射
  - status         根据「退款金额 / 付款时间」推导
  - customer_id    合成（按订单号取模，模拟客户重复下单）
  - items          置空（订单级数据没有商品明细，这是数据源的局限）
"""
import csv
import json
from pathlib import Path

SRC = Path(r"C:\Users\ALIENWARE\Downloads\tmall_order_report.csv")
OUT = Path("data/seed/orders.json")


def read_csv(path: Path) -> list[dict]:
    for enc in ("utf-8-sig", "gbk", "utf-8"):
        try:
            with open(path, encoding=enc, newline="") as f:
                reader = csv.reader(f)
                header = [h.strip() for h in next(reader)]
                return [dict(zip(header, row)) for row in reader]
        except (UnicodeDecodeError, StopIteration):
            continue
    raise RuntimeError("无法识别 CSV 编码")


def derive_status(refund: float, paid: str) -> str:
    if refund > 0:
        return "refunded"      # 有退款金额 → 已退款
    if not paid.strip():
        return "cancelled"     # 无付款时间 → 未付款（取消）
    return "paid"              # 已付款


rows = read_csv(SRC)
orders = []
for r in rows:
    oid = int(r["订单编号"])
    orders.append({
        "order_id": f"ORD-2020-{oid}",
        "customer_id": f"C{(oid % 500) + 1:03d}",
        "items": [],
        "total_amount": float(r["总金额"]),
        "status": derive_status(float(r["退款金额"]), r["订单付款时间"]),
        "created_at": r["订单创建时间"].strip(),
        "shipping_address": r["收货地址"].strip(),
    })

OUT.write_text(json.dumps(orders, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"转换完成：{len(orders)} 条订单 -> {OUT}")

# 顺便统计一下状态分布，方便确认
from collections import Counter
print("状态分布：", dict(Counter(o["status"] for o in orders)))
