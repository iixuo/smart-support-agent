"""Mock e-commerce backend (SQLite). Seed data is hand-written so eval answers are deterministic.

Simulated today = 2026-11-18 (see SIM_TODAY).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .config import DB_PATH

SCHEMA = """
CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY, name TEXT, level TEXT  -- normal | plus
);
CREATE TABLE orders (
    order_id TEXT PRIMARY KEY, customer_id TEXT, created_at TEXT, status TEXT,
    items TEXT,            -- JSON list of {sku,name,category,unit_price,qty,tags}
    shipping_fee REAL, coupon_discount REAL, paid_amount REAL, payment_method TEXT,
    promised_date TEXT, delivered_at TEXT, carrier TEXT, tracking_no TEXT,
    events TEXT            -- JSON list of {time, desc}
);
CREATE TABLE tickets (
    ticket_id TEXT PRIMARY KEY, customer_id TEXT, order_id TEXT, category TEXT,
    summary TEXT, priority TEXT, status TEXT, created_at TEXT
);
CREATE TABLE handoffs (
    handoff_id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id TEXT, reason TEXT, urgency TEXT
);
"""

CUSTOMERS = [
    ("C001", "林晓", "normal"),
    ("C002", "王磊", "plus"),
    ("C003", "陈静", "normal"),
    ("C004", "赵强", "normal"),
    ("C005", "刘洋", "plus"),
]


def _item(sku, name, category, price, qty=1, tags=()):
    return {"sku": sku, "name": name, "category": category, "unit_price": price, "qty": qty, "tags": list(tags)}


ORDERS = [
    # --- C001 ---
    dict(order_id="SO20261101", customer_id="C001", created_at="2026-11-10", status="delivered",
         items=[_item("EAR-X1", "降噪耳机 X1", "electronics", 399.0)],
         shipping_fee=0, coupon_discount=0, paid_amount=399.0, payment_method="wechat",
         promised_date="2026-11-14", delivered_at="2026-11-14", carrier="顺丰", tracking_no="SF1001",
         events=[("2026-11-11 09:00", "已揽收"), ("2026-11-13 18:20", "到达上海转运中心"),
                 ("2026-11-14 15:02", "已签收，签收人：本人")]),
    dict(order_id="SO20261102", customer_id="C001", created_at="2026-11-08", status="shipped",
         items=[_item("CASE-15", "手机壳", "accessories", 39.0, qty=2)],
         shipping_fee=8.0, coupon_discount=0, paid_amount=86.0, payment_method="alipay",
         promised_date="2026-11-12", delivered_at=None, carrier="中通", tracking_no="ZT2002",
         events=[("2026-11-09 10:00", "已揽收"), ("2026-11-10 22:15", "到达郑州转运中心"),
                 ("2026-11-11 08:30", "转运中心滞留，等待调度")]),
    dict(order_id="SO20261020", customer_id="C001", created_at="2026-10-23", status="delivered",
         items=[_item("FRT-CHERRY", "智利车厘子 2kg", "fresh", 168.0, tags=["fresh"])],
         shipping_fee=0, coupon_discount=0, paid_amount=168.0, payment_method="wechat",
         promised_date="2026-10-25", delivered_at="2026-10-25", carrier="京东冷链", tracking_no="JD3003",
         events=[("2026-10-24 07:00", "已揽收"), ("2026-10-25 11:40", "已签收")]),
    # --- C002 (PLUS) ---
    dict(order_id="SO20261105", customer_id="C002", created_at="2026-11-09", status="delivered",
         items=[_item("KB-M87", "机械键盘 M87", "electronics", 599.0), _item("PAD-XL", "鼠标垫 XL", "accessories", 49.0)],
         shipping_fee=0, coupon_discount=50.0, paid_amount=598.0, payment_method="bank_card",
         promised_date="2026-11-13", delivered_at="2026-11-13", carrier="顺丰", tracking_no="SF4004",
         events=[("2026-11-10 09:30", "已揽收"), ("2026-11-13 10:10", "已签收")]),
    dict(order_id="SO20261110", customer_id="C002", created_at="2026-11-16", status="pending_shipment",
         items=[_item("WATCH-S3", "智能手表 S3", "electronics", 1299.0)],
         shipping_fee=0, coupon_discount=0, paid_amount=1299.0, payment_method="alipay",
         promised_date="2026-11-20", delivered_at=None, carrier=None, tracking_no=None, events=[]),
    # --- C003 ---
    dict(order_id="SO20261015", customer_id="C003", created_at="2026-10-24", status="delivered",
         items=[_item("SPK-B2", "蓝牙音箱 B2", "electronics", 259.0)],
         shipping_fee=0, coupon_discount=0, paid_amount=259.0, payment_method="wechat",
         promised_date="2026-10-28", delivered_at="2026-10-28", carrier="圆通", tracking_no="YT5005",
         events=[("2026-10-25 09:00", "已揽收"), ("2026-10-28 16:45", "已签收")]),
    dict(order_id="SO20261108", customer_id="C003", created_at="2026-11-08", status="delivered",
         items=[_item("CARD-100", "话费充值卡 100 元", "virtual", 100.0, tags=["virtual"])],
         shipping_fee=0, coupon_discount=0, paid_amount=100.0, payment_method="alipay",
         promised_date="2026-11-08", delivered_at="2026-11-08", carrier="电子发货", tracking_no=None,
         events=[("2026-11-08 12:00", "卡密已发送至账户")]),
    # --- C004 ---
    dict(order_id="SO20261103", customer_id="C004", created_at="2026-11-15", status="shipped",
         items=[_item("SHOE-R5", "跑步鞋 R5", "apparel", 459.0)],
         shipping_fee=0, coupon_discount=0, paid_amount=459.0, payment_method="wechat",
         promised_date="2026-11-20", delivered_at=None, carrier="韵达", tracking_no="YD6006",
         events=[("2026-11-16 08:00", "已揽收"), ("2026-11-17 20:00", "到达成都转运中心")]),
    dict(order_id="SO20261001", customer_id="C004", created_at="2026-10-02", status="delivered",
         items=[_item("LAMP-D1", "护眼台灯 D1", "electronics", 89.0)],
         shipping_fee=8.0, coupon_discount=0, paid_amount=97.0, payment_method="bank_card",
         promised_date="2026-10-05", delivered_at="2026-10-05", carrier="中通", tracking_no="ZT7007",
         events=[("2026-10-03 09:00", "已揽收"), ("2026-10-05 14:00", "已签收")]),
    # --- C005 (PLUS) ---
    dict(order_id="SO20261109", customer_id="C005", created_at="2026-11-12", status="delivered",
         items=[_item("CUP-ENG", "定制刻字保温杯", "home", 129.0, tags=["custom"])],
         shipping_fee=0, coupon_discount=0, paid_amount=129.0, payment_method="wechat",
         promised_date="2026-11-15", delivered_at="2026-11-15", carrier="顺丰", tracking_no="SF8008",
         events=[("2026-11-13 09:00", "已揽收"), ("2026-11-15 13:00", "已签收")]),
    dict(order_id="SO20261111", customer_id="C005", created_at="2026-11-08", status="delivered",
         items=[_item("FRY-A6", "空气炸锅 A6", "electronics", 349.0)],
         shipping_fee=0, coupon_discount=0, paid_amount=349.0, payment_method="alipay",
         promised_date="2026-11-11", delivered_at="2026-11-11", carrier="京东", tracking_no="JD9009",
         events=[("2026-11-09 09:00", "已揽收"), ("2026-11-11 10:30", "已签收")]),
]


def init_db(path: Path | str = DB_PATH) -> sqlite3.Connection:
    """(Re)create the database with seed data. Returns an open connection."""
    path = Path(path) if path != ":memory:" else path
    if path != ":memory:" and Path(path).exists():
        Path(path).unlink()
    conn = connect(path)
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO customers VALUES (?,?,?)", CUSTOMERS)
    for o in ORDERS:
        conn.execute(
            "INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (o["order_id"], o["customer_id"], o["created_at"], o["status"], json.dumps(o["items"], ensure_ascii=False),
             o["shipping_fee"], o["coupon_discount"], o["paid_amount"], o["payment_method"], o["promised_date"],
             o["delivered_at"], o["carrier"], o["tracking_no"],
             json.dumps([{"time": t, "desc": d} for t, d in o["events"]], ensure_ascii=False)),
        )
    conn.commit()
    return conn


def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def get_customer(conn, customer_id: str) -> dict | None:
    r = conn.execute("SELECT * FROM customers WHERE customer_id=?", (customer_id,)).fetchone()
    return dict(r) if r else None


def get_order(conn, order_id: str) -> dict | None:
    r = conn.execute("SELECT * FROM orders WHERE order_id=?", (order_id,)).fetchone()
    if not r:
        return None
    d = dict(r)
    d["items"] = json.loads(d["items"])
    d["events"] = json.loads(d["events"])
    return d


def list_orders(conn, customer_id: str) -> list[dict]:
    rows = conn.execute("SELECT order_id FROM orders WHERE customer_id=? ORDER BY created_at DESC",
                        (customer_id,)).fetchall()
    return [get_order(conn, r["order_id"]) for r in rows]
