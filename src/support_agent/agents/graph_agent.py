"""Same agent expressed as a LangGraph state graph, with a checkpointer for per-session memory.

    START -> agent --(tool_calls & guards ok)--> tools -> agent
                 \\--(no tool_calls)-----------> END
                 \\--(guard tripped)-----------> finalize -> END
"""
from __future__ import annotations

import json
import operator
import time
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from ..config import Settings, get_settings
from ..llm import ChatModel, ToolCall
from ..tools.base import ToolContext, ToolRegistry
from .core import Guard, RunResult, execute_tool_calls
from .prompts import FORCE_FINAL, agent_system


class AgentState(TypedDict):
    messages: Annotated[list[dict], operator.add]  # conversation memory, persisted by checkpointer
    turn_start: int                                 # index of this turn's user message
    llm_calls: int
    stop_reason: str
    pending: list[dict]                             # tool calls requested by the last LLM step


class GraphAgent:
    name = "agent_langgraph"

    def __init__(self, llm: ChatModel, registry: ToolRegistry, settings: Settings | None = None):
        self.llm, self.registry, self.s = llm, registry, settings or get_settings()
        self.tools = registry.schemas()
        self.checkpointer = InMemorySaver()
        # per-run, non-serialisable objects (ctx, tracer) live here, keyed by thread id
        self._runtime: dict[str, tuple[ToolContext, RunResult, Guard]] = {}
        self.graph = self._build()

    # ---------------------------------------------------------------- nodes
    def _agent(self, state: AgentState, config):
        ctx, res, guard = self._runtime[config["configurable"]["thread_id"]]
        stop = guard.check_before_llm(state["llm_calls"])
        if stop:
            return {"stop_reason": stop, "pending": []}
        msgs = [{"role": "system", "content": agent_system(ctx.today, ctx.customer_id)}] + state["messages"]
        r = self.llm.chat(msgs, self.tools)
        res.add_usage(r, self.s)
        out = {"messages": [r.as_message()], "llm_calls": state["llm_calls"] + 1,
               "pending": [c.__dict__ for c in r.tool_calls]}
        if r.tool_calls:
            stop = guard.check_calls(r.tool_calls)
            if stop:
                out["messages"].extend({"role": "tool", "tool_call_id": c.id, "content": json.dumps(
                    {"ok": False, "error": {"type": "repeated_call", "message": "重复调用已被拦截"}},
                    ensure_ascii=False)} for c in r.tool_calls)
                out.update(stop_reason=stop, pending=[])
        else:
            res.answer = r.content or ""
            out["stop_reason"] = "final_answer"
        return out

    def _tools(self, state: AgentState, config):
        ctx, res, guard = self._runtime[config["configurable"]["thread_id"]]
        calls = [ToolCall(**c) for c in state["pending"]]
        tool_msgs, raws = execute_tool_calls(calls, self.registry, ctx, res)
        stop = guard.record_results(raws)
        return {"messages": tool_msgs, "pending": [], "stop_reason": stop or ""}

    def _finalize(self, state: AgentState, config):
        ctx, res, _ = self._runtime[config["configurable"]["thread_id"]]
        msgs = ([{"role": "system", "content": agent_system(ctx.today, ctx.customer_id)}] + state["messages"]
                + [{"role": "user", "content": FORCE_FINAL}])
        r = self.llm.chat(msgs, self.tools, tool_choice="none")
        res.add_usage(r, self.s)
        res.answer = r.content or ""
        return {"messages": [{"role": "assistant", "content": res.answer}], "llm_calls": state["llm_calls"] + 1}

    # ---------------------------------------------------------------- routing
    @staticmethod
    def _after_agent(state: AgentState) -> str:
        if state["stop_reason"] == "final_answer":
            return END
        if state["stop_reason"]:
            return "finalize"
        return "tools"

    @staticmethod
    def _after_tools(state: AgentState) -> str:
        return "finalize" if state["stop_reason"] else "agent"

    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("agent", self._agent)
        g.add_node("tools", self._tools)
        g.add_node("finalize", self._finalize)
        g.add_edge(START, "agent")
        g.add_conditional_edges("agent", self._after_agent, {"tools": "tools", "finalize": "finalize", END: END})
        g.add_conditional_edges("tools", self._after_tools, {"agent": "agent", "finalize": "finalize"})
        g.add_edge("finalize", END)
        return g.compile(checkpointer=self.checkpointer)

    # ---------------------------------------------------------------- public
    def run(self, user_msg: str, ctx: ToolContext, thread_id: str) -> RunResult:
        """Multi-turn memory: call again with the same thread_id and the checkpointer restores the history."""
        t0 = time.perf_counter()
        res, n_effects = RunResult(answer=""), len(ctx.side_effects)
        self._runtime[thread_id] = (ctx, res, Guard(self.s))
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 4 * self.s.max_steps + 10}
        prev = self.graph.get_state(config).values.get("messages", [])
        try:
            final = self.graph.invoke({"messages": [{"role": "user", "content": user_msg}], "turn_start": len(prev),
                                       "llm_calls": 0, "stop_reason": "", "pending": []}, config)
        finally:
            self._runtime.pop(thread_id, None)
        res.stop_reason = final["stop_reason"] or "final_answer"
        res.latency_s = time.perf_counter() - t0
        res.side_effects = ctx.side_effects[n_effects:]
        return res

    def mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()
