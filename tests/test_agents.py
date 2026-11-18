from support_agent.agents import GraphAgent, ReActAgent
from support_agent.llm import FakeLLM

REFUND_SCRIPT = [
    [("get_order_detail", {"order_id": "SO20261101"}), ("search_policy", {"query": "七天无理由退货 运费"})],
    [("calculate_refund", {"order_id": "SO20261101", "reason": "no_reason_7d"})],
    "可以退，退款 389.00 元。",
]


def make(kind, script, registry, settings):
    cls = ReActAgent if kind == "loop" else GraphAgent
    return cls(FakeLLM(script), registry, settings)


def run(agent, msg, c, thread="t1"):
    if isinstance(agent, GraphAgent):
        return agent.run(msg, c, thread_id=thread)
    return agent.run(msg, c)[0]


import pytest


@pytest.mark.parametrize("kind", ["loop", "graph"])
def test_multi_step_tool_use(kind, registry, ctx, settings):
    res = run(make(kind, list(REFUND_SCRIPT), registry, settings), "耳机能退吗", ctx())
    assert res.stop_reason == "final_answer"
    assert res.tool_names == ["get_order_detail", "search_policy", "calculate_refund"]
    assert res.llm_calls == 3 and "389" in res.answer


@pytest.mark.parametrize("kind", ["loop", "graph"])
def test_self_correction_after_validation_error(kind, registry, ctx, settings):
    script = [[("get_order_detail", {"order_id": "20261101"})],
              [("get_order_detail", {"order_id": "SO20261101"})], "好的"]
    res = run(make(kind, script, registry, settings), "查订单 20261101", ctx())
    assert [c["ok"] for c in res.tool_calls] == [False, True]
    assert res.stop_reason == "final_answer"


@pytest.mark.parametrize("kind", ["loop", "graph"])
def test_max_steps_forces_final_answer(kind, registry, ctx, settings):
    script = [[("search_policy", {"query": f"问题{i}"})] for i in range(10)]
    agent = make(kind, script, registry, settings)
    res = run(agent, "?", ctx())
    assert res.stop_reason == "max_steps"
    assert res.llm_calls == settings.max_steps + 1        # + forced final call
    assert agent.llm.calls[-1]["tool_choice"] == "none"
    assert res.answer


@pytest.mark.parametrize("kind", ["loop", "graph"])
def test_repeated_identical_call_is_blocked(kind, registry, ctx, settings):
    script = [[("track_shipment", {"order_id": "SO20261102"})]] * 5
    res = run(make(kind, script, registry, settings), "?", ctx())
    assert res.stop_reason == "repeated_call"
    assert len(res.tool_calls) == 2


@pytest.mark.parametrize("kind", ["loop", "graph"])
def test_consecutive_errors_stop(kind, registry, ctx, settings):
    script = [[("get_order_detail", {"order_id": f"SO2026110{i}"})] for i in [5, 0, 9, 8]]  # not owned / missing
    res = run(make(kind, script, registry, settings), "?", ctx("C001"))
    assert res.stop_reason == "too_many_errors"


def test_graph_memory_across_turns(registry, ctx, settings):
    agent = make("graph", ["你好，请问有什么可以帮您？", "您刚才问的是耳机"], registry, settings)
    c = ctx()
    agent.run("我想问耳机", c, thread_id="s1")
    agent.run("我刚才问的什么？", c, thread_id="s1")
    second_call_msgs = agent.llm.calls[-1]["messages"]
    assert any(m["content"] == "我想问耳机" for m in second_call_msgs)


def test_loop_memory_across_turns(registry, ctx, settings):
    agent = make("loop", list(REFUND_SCRIPT) + ["是的，389 元"], registry, settings)
    c = ctx()
    _, hist = agent.run("耳机能退吗", c)
    agent.run("确定是389吗", c, history=hist)
    msgs = agent.llm.calls[-1]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
