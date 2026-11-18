import json


def run(registry, ctx, name, args):
    return registry.execute(name, args, ctx)


def test_schema_is_openai_function_format(registry):
    schemas = registry.schemas()
    assert {s["function"]["name"] for s in schemas} >= {"search_policy", "calculate_refund", "create_ticket"}
    refund = next(s for s in schemas if s["function"]["name"] == "calculate_refund")["function"]["parameters"]
    assert refund["required"] == ["order_id", "reason"]
    assert "pattern" in refund["properties"]["order_id"]


def test_invalid_arguments_are_reported_not_raised(registry, ctx):
    out = run(registry, ctx(), "get_order_detail", {"order_id": "123"})
    assert out["ok"] is False and out["error"]["type"] == "invalid_arguments"
    out = run(registry, ctx(), "calculate_refund", '{"order_id": "SO20261101", "reason": "because"}')
    assert out["error"]["type"] == "invalid_arguments"
    assert run(registry, ctx(), "get_order_detail", "{not json")["error"]["type"] == "invalid_json"
    assert run(registry, ctx(), "delete_db", {})["error"]["type"] == "unknown_tool"


def test_cannot_read_other_customers_order(registry, ctx):
    out = run(registry, ctx("C001"), "get_order_detail", {"order_id": "SO20261105"})  # belongs to C002
    assert out["error"]["type"] == "permission_denied"
    assert "598" not in json.dumps(out, ensure_ascii=False)


def test_refund_engine_rules(registry, ctx):
    def amount(cid, **a):
        d = run(registry, ctx(cid), "calculate_refund", a)["data"]
        return d["eligible"], d["refund_amount"]
    assert amount("C001", order_id="SO20261101", reason="no_reason_7d") == (True, 389.0)   # -10 shipping
    assert amount("C002", order_id="SO20261105", reason="no_reason_7d", skus=["KB-M87"]) == (True, 552.78)  # coupon share, PLUS
    assert amount("C005", order_id="SO20261111", reason="no_reason_7d") == (True, 349.0)   # day 7 inclusive
    assert amount("C005", order_id="SO20261109", reason="no_reason_7d") == (False, 0.0)    # customised
    assert amount("C001", order_id="SO20261102", reason="delayed_delivery") == (True, 86.0)
    assert amount("C004", order_id="SO20261103", reason="delayed_delivery") == (False, 0.0)
    assert amount("C004", order_id="SO20261001", reason="quality_issue") == (False, 0.0)    # > 30 days


def test_ticket_is_idempotent(registry, ctx):
    c = ctx("C001")
    a = {"category": "return", "order_id": "SO20261101", "summary": "耳机七天无理由退货"}
    first = run(registry, c, "create_ticket", a)["data"]
    second = run(registry, c, "create_ticket", a)["data"]
    assert second["duplicate"] and second["ticket_id"] == first["ticket_id"]
    assert len(c.side_effects) == 1


def test_calculator_is_sandboxed(registry, ctx):
    assert run(registry, ctx(), "calculator", {"expression": "599 - 50*599/648"})["data"]["result"] == 552.7809
    assert run(registry, ctx(), "calculator", {"expression": "__import__('os').system('ls')"})["ok"] is False
