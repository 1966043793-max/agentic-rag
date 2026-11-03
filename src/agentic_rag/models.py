"""领域数据模型（Pydantic）。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Product(BaseModel):
    product_id: str
    name: str
    category: str
    brand: str
    price: float
    stock: int
    specs: dict = Field(default_factory=dict)


class OrderItem(BaseModel):
    product_id: str
    product_name: str
    quantity: int
    unit_price: float


class Order(BaseModel):
    order_id: str
    customer_id: str
    items: list[OrderItem]
    total_amount: float
    status: str  # pending/paid/shipped/delivered/refunding/refunded/cancelled
    created_at: str
    shipping_address: str


class Customer(BaseModel):
    customer_id: str
    name: str
    level: str  # normal/silver/gold/diamond
    tags: list[str] = Field(default_factory=list)
    preferences: dict = Field(default_factory=dict)
    joined_at: str


class Refund(BaseModel):
    refund_id: str
    order_id: str
    customer_id: str
    product_id: str
    reason: str
    amount: float
    status: str  # applied/approved/rejected/completed
    created_at: str
