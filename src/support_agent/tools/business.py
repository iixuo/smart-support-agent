"""Business tools for the e-commerce support agent."""
from __future__ import annotations

import ast
import operator
from datetime import date
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .. import db
from .base import (BusinessRuleError, NotFoundError, PermissionDeniedError, Tool, ToolContext, ToolRegistry)

ORDER_ID = Field(..., pattern=r"^SO\d{8}$", description="订单号，格式 SO + 8 位数字，如 SO20261101")
NON_RETURNABLE_TAGS = {"fresh": "生鲜食品", "custom": "定制类商品", "virtual": "虚拟商品"}
RETURN_SHIPPING_FEE = 10.0
DELAY_THRESHOLD_DAYS = 3
DELAY_COUPON = 20.0


def _owned_order(ctx: ToolContext, order_id: str) -> dict:
    order = db.get_order(ctx.conn, order_id)
    # same error for "not exists" and "not yours" -> no enumeration of other users' orders
    if order is None or order["customer_id"] != ctx.customer_id:
        if order is not None:
            raise PermissionDeniedError(f"订单 {order_id} 不属于当前登录用户",
                                        "告知用户只能查询本人订单，不要透露该订单任何信息")
        raise NotFoundError(f"未找到订单 {order_id}", "请用户核对订单号，或调用 list_my_orders 查看本人订单")
    return order


def _days(a: str, today: date) -> int:
    return (today - date.fromisoformat(a)).days


# ---------------------------------------------------------------- search_policy
class SearchPolicyArgs(BaseModel):
    query: str = Field(..., min_length=2, max_length=100,
                       description="检索语句。用规范的政策关键词改写用户问题，如“七天无理由退货 运费”")
    top_k: int = Field(3, ge=1, le=5, description="返回条数")


def search_policy(ctx: ToolContext, a: SearchPolicyArgs):
    return ctx.retriever.search(a.query, a.top_k)


# ---------------------------------------------------------------- list_my_orders
class OrderStatus(str, Enum):
    pending_shipment = "pending_shipment"
    shipped = "shipped"
    delivered = "delivered"


class ListOrdersArgs(BaseModel):
    status: Optional[OrderStatus] = Field(None, description="按状态过滤，可选")
    limit: int = Field(10, ge=1, le=20)


def list_my_orders(ctx: ToolContext, a: ListOrdersArgs):
    orders = db.list_orders(ctx.conn, ctx.customer_id)
    if a.status:
        orders = [o for o in orders if o["status"] == a.status.value]
    return [{"order_id": o["order_id"], "created_at": o["created_at"], "status": o["status"],
             "items": "、".join(f'{i["name"]}x{i["qty"]}' for i in o["items"]), "paid_amount": o["paid_amount"]}
            for o in orders[: a.limit]]


# ---------------------------------------------------------------- get_order_detail
class OrderIdArgs(BaseModel):
    order_id: str = ORDER_ID


def get_order_detail(ctx: ToolContext, a: OrderIdArgs):
    o = _owned_order(ctx, a.order_id)
    cust = db.get_customer(ctx.conn, ctx.customer_id)
    keys = ["order_id", "created_at", "status", "items", "shipping_fee", "coupon_discount", "paid_amount",
            "payment_method", "promised_date", "delivered_at"]
    out = {k: o[k] for k in keys}
    out["customer_level"] = cust["level"]
    if o["delivered_at"]:
        out["days_since_delivery"] = _days(o["delivered_at"], ctx.today)
    return out


# ---------------------------------------------------------------- track_shipment
def track_shipment(ctx: ToolContext, a: OrderIdArgs):
    o = _owned_order(ctx, a.order_id)
    if o["status"] == "pending_shipment":
        return {"order_id": o["order_id"], "status": "pending_shipment", "message": "订单尚未发货",
                "promised_date": o["promised_date"]}
    days_late = _days(o["promised_date"], ctx.today) if o["status"] == "shipped" else 0
    return {"order_id": o["order_id"], "status": o["status"], "carrier": o["carrier"],
            "tracking_no": o["tracking_no"], "promised_date": o["promised_date"],
            "delivered_at": o["delivered_at"], "events": o["events"],
            "days_past_promised": max(days_late, 0),
            "is_delayed": days_late >= DELAY_THRESHOLD_DAYS}


# ---------------------------------------------------------------- calculate_refund
class RefundReason(str, Enum):
    no_reason_7d = "no_reason_7d"
    quality_issue = "quality_issue"
    delayed_delivery = "delayed_delivery"
    cancel_unshipped = "cancel_unshipped"


class RefundArgs(BaseModel):
    order_id: str = ORDER_ID
    reason: RefundReason = Field(..., description="no_reason_7d=七天无理由; quality_issue=质量问题; "
                                                  "delayed_delivery=物流延误; cancel_unshipped=未发货取消")
    skus: Optional[list[str]] = Field(None, description="只退部分商品时填写商品 sku 列表；不填表示整单")


def calculate_refund(ctx: ToolContext, a: RefundArgs):
    """Deterministic refund engine — the LLM must NOT do refund arithmetic itself."""
    o = _owned_order(ctx, a.order_id)
    level = db.get_customer(ctx.conn, ctx.customer_id)["level"]
    items = o["items"]
    if a.skus:
        unknown = set(a.skus) - {i["sku"] for i in items}
        if unknown:
            raise NotFoundError(f"订单中不存在商品 {sorted(unknown)}", f"可选 sku: {[i['sku'] for i in items]}")
        items = [i for i in items if i["sku"] in a.skus]
    whole = len(items) == len(o["items"])
    res = {"order_id": o["order_id"], "reason": a.reason.value, "customer_level": level,
           "scope": "整单" if whole else "部分商品: " + "、".join(i["name"] for i in items)}

    def ineligible(why: str):
        return {**res, "eligible": False, "refund_amount": 0.0, "explanation": why}

    if a.reason == RefundReason.cancel_unshipped:
        if o["status"] != "pending_shipment":
            return ineligible(f"订单状态为 {o['status']}，只有待发货订单可以直接取消")
        return {**res, "eligible": True, "refund_amount": o["paid_amount"], "explanation": "待发货订单取消，全额退款"}

    if a.reason == RefundReason.delayed_delivery:
        if o["status"] != "shipped":
            return ineligible(f"订单状态为 {o['status']}，物流延误补偿只适用于已发货未签收订单")
        late = _days(o["promised_date"], ctx.today)
        if late < DELAY_THRESHOLD_DAYS:
            return ineligible(f"超过承诺送达日 {max(late, 0)} 天，不足 {DELAY_THRESHOLD_DAYS} 天，不适用延误补偿")
        return {**res, "eligible": True, "refund_amount": o["paid_amount"], "days_past_promised": late,
                "alternative": f"继续等待并获得 {DELAY_COUPON:.0f} 元无门槛优惠券",
                "explanation": f"已超过承诺送达日 {late} 天，可整单全额退款（含运费）或继续等待领券补偿"}

    if o["status"] != "delivered":
        return ineligible(f"订单状态为 {o['status']}，签收后才能申请退货")
    since = _days(o["delivered_at"], ctx.today)
    goods_total = sum(i["unit_price"] * i["qty"] for i in o["items"])
    sel_total = sum(i["unit_price"] * i["qty"] for i in items)
    if whole:
        base = o["paid_amount"]
        coupon_share = o["coupon_discount"]
    else:
        coupon_share = round(o["coupon_discount"] * sel_total / goods_total, 2)
        base = round(sel_total - coupon_share, 2)
    res.update(days_since_delivery=since, items_amount=sel_total, coupon_share=coupon_share)

    if a.reason == RefundReason.quality_issue:
        if any("virtual" in i["tags"] for i in items):
            return ineligible("虚拟商品不支持退货")
        if since > 30:
            return ineligible(f"已签收 {since} 天，超过 30 天质量问题退货期限；电子产品可在 1 年保修期内申请维修")
        return {**res, "eligible": True, "refund_amount": base, "deductions": [],
                "explanation": f"签收 {since} 天，在 30 天质量问题退货期内，全额退还实付金额，运费由商家承担"}

    # no_reason_7d
    for i in items:
        bad = [NON_RETURNABLE_TAGS[t] for t in i["tags"] if t in NON_RETURNABLE_TAGS]
        if bad:
            return ineligible(f"{i['name']} 属于{bad[0]}，不支持七天无理由退货")
    if since > 7:
        return ineligible(f"已签收 {since} 天，超过 7 天无理由退货期限")
    deductions = [] if level == "plus" else [{"item": "退回运费", "amount": RETURN_SHIPPING_FEE}]
    amount = round(base - sum(d["amount"] for d in deductions), 2)
    return {**res, "eligible": True, "refund_amount": amount, "deductions": deductions,
            "explanation": f"签收 {since} 天，可七天无理由退货" + ("；PLUS 会员免退回运费" if level == "plus"
                                                         else f"；扣除退回运费 {RETURN_SHIPPING_FEE:.0f} 元")}


# ---------------------------------------------------------------- create_ticket
class TicketArgs(BaseModel):
    category: Literal["return", "refund", "exchange", "repair", "shipping", "invoice", "complaint", "other"]
    summary: str = Field(..., min_length=5, max_length=300, description="工单摘要：问题、诉求、已核实的信息")
    order_id: Optional[str] = Field(None, pattern=r"^SO\d{8}$")
    priority: Literal["low", "normal", "high"] = "normal"


def create_ticket(ctx: ToolContext, a: TicketArgs):
    if a.order_id:
        _owned_order(ctx, a.order_id)
        dup = ctx.conn.execute("SELECT ticket_id FROM tickets WHERE customer_id=? AND order_id=? AND category=? "
                               "AND status='open'", (ctx.customer_id, a.order_id, a.category)).fetchone()
        if dup:  # idempotency
            return {"ticket_id": dup["ticket_id"], "duplicate": True, "message": "该订单已有同类工单在处理中"}
    n = ctx.conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] + 1
    tid = f"TK{ctx.today:%Y%m%d}{n:03d}"
    ctx.conn.execute("INSERT INTO tickets VALUES (?,?,?,?,?,?,?,?)",
                     (tid, ctx.customer_id, a.order_id, a.category, a.summary, a.priority, "open",
                      ctx.today.isoformat()))
    ctx.conn.commit()
    ctx.side_effects.append({"type": "ticket", "ticket_id": tid, "category": a.category, "order_id": a.order_id})
    return {"ticket_id": tid, "duplicate": False, "status": "open", "sla": "24 小时内客服跟进"}


# ---------------------------------------------------------------- handoff_to_human
class HandoffArgs(BaseModel):
    reason: str = Field(..., min_length=2, max_length=200)
    urgency: Literal["normal", "high"] = "normal"


def handoff_to_human(ctx: ToolContext, a: HandoffArgs):
    cur = ctx.conn.execute("INSERT INTO handoffs (customer_id, reason, urgency) VALUES (?,?,?)",
                           (ctx.customer_id, a.reason, a.urgency))
    ctx.conn.commit()
    ctx.side_effects.append({"type": "handoff", "reason": a.reason, "urgency": a.urgency})
    return {"handoff_id": cur.lastrowid, "queue_position": 1 if a.urgency == "high" else 3,
            "message": "已转接人工客服（服务时间 9:00–22:00）"}


# ---------------------------------------------------------------- calculator
class CalcArgs(BaseModel):
    expression: str = Field(..., max_length=200, description="算术表达式，仅支持 + - * / ( ) 和数字")


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise BusinessRuleError("表达式包含不支持的语法", "只使用数字和 + - * / ( )")


def calculator(ctx: ToolContext, a: CalcArgs):
    try:
        tree = ast.parse(a.expression, mode="eval")
    except SyntaxError:
        raise BusinessRuleError("表达式语法错误", "检查括号与运算符")
    try:
        return {"expression": a.expression, "result": round(_safe_eval(tree.body), 4)}
    except ZeroDivisionError:
        raise BusinessRuleError("除数为 0")


# ---------------------------------------------------------------- registry
def build_registry() -> ToolRegistry:
    return ToolRegistry([
        Tool("search_policy", "检索商城政策知识库（退货、退款、运费、保修、发票、价保、会员、客服规范）。"
                              "回答任何规则类问题前必须先检索，不要凭记忆回答。", SearchPolicyArgs, search_policy),
        Tool("list_my_orders", "列出当前登录用户的订单（用户没给订单号时使用）。", ListOrdersArgs, list_my_orders),
        Tool("get_order_detail", "查询当前用户某个订单的详情：商品、金额、优惠、状态、签收日期、会员等级。",
             OrderIdArgs, get_order_detail),
        Tool("track_shipment", "查询订单物流轨迹，返回承诺送达日、超期天数、是否延误。", OrderIdArgs, track_shipment),
        Tool("calculate_refund", "按平台规则计算退款资格和金额（含优惠分摊、运费扣除、延误补偿）。"
                                 "涉及退款金额时必须调用本工具，不要自己计算。", RefundArgs, calculate_refund),
        Tool("create_ticket", "创建售后工单（退货/退款/换货/维修/物流/发票/投诉）。仅在用户明确要求办理时调用。",
             TicketArgs, create_ticket, writes=True),
        Tool("handoff_to_human", "转接人工客服：用户要求人工、投诉/情绪激烈/法律维权、账户安全问题、超出能力范围。",
             HandoffArgs, handoff_to_human, writes=True),
        Tool("calculator", "精确计算算术表达式。", CalcArgs, calculator),
    ])
