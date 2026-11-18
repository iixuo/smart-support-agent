"""Hand-written Function Calling loop (no framework) — the core of every tool-using agent.

    while True:
        response = LLM(messages, tools)
        if no tool_calls: return response            # model decided it is done
        for call in tool_calls: messages += run(call) # observe
        guard: max steps / timeout / repeated call / consecutive errors -> forced final answer
"""
from __future__ import annotations

import json
import time

from ..config import Settings, get_settings
from ..llm import ChatModel
from ..tools.base import ToolContext, ToolRegistry
from .core import Guard, RunResult, execute_tool_calls
from .prompts import FORCE_FINAL, agent_system


class ReActAgent:
    name = "agent_loop"

    def __init__(self, llm: ChatModel, registry: ToolRegistry, settings: Settings | None = None):
        self.llm, self.registry, self.s = llm, registry, settings or get_settings()
        self.tools = registry.schemas()

    def run(self, user_msg: str, ctx: ToolContext, history: list[dict] | None = None) -> tuple[RunResult, list[dict]]:
        """Returns (result, updated history). Pass history back in for multi-turn memory."""
        t0 = time.perf_counter()
        msgs = [{"role": "system", "content": agent_system(ctx.today, ctx.customer_id)}]
        msgs += history or []
        msgs.append({"role": "user", "content": user_msg})
        start = len(msgs) - 1
        res, guard = RunResult(answer=""), Guard(self.s)
        n_effects = len(ctx.side_effects)

        while True:
            stop = guard.check_before_llm(res.llm_calls)
            if stop:
                return self._force_final(msgs, res, stop, ctx, t0, start, n_effects)
            r = self.llm.chat(msgs, self.tools)
            res.add_usage(r, self.s)
            msgs.append(r.as_message())
            if not r.tool_calls:
                res.answer = r.content or ""
                return self._finish(msgs, res, ctx, t0, start, n_effects)

            stop = guard.check_calls(r.tool_calls)
            if stop:  # every tool_call still needs a tool message, or the API rejects the next request
                for c in r.tool_calls:
                    msgs.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(
                        {"ok": False, "error": {"type": "repeated_call", "message": "重复调用已被拦截"}},
                        ensure_ascii=False)})
                return self._force_final(msgs, res, stop, ctx, t0, start, n_effects)
            tool_msgs, raws = execute_tool_calls(r.tool_calls, self.registry, ctx, res)
            msgs += tool_msgs
            stop = guard.record_results(raws)
            if stop:
                return self._force_final(msgs, res, stop, ctx, t0, start, n_effects)

    def _force_final(self, msgs, res, reason, ctx, t0, start, n_effects):
        msgs.append({"role": "user", "content": FORCE_FINAL})
        r = self.llm.chat(msgs, self.tools, tool_choice="none")
        res.add_usage(r, self.s)
        msgs.pop()  # don't keep the control instruction in memory
        msgs.append({"role": "assistant", "content": r.content or ""})
        res.answer, res.stop_reason = r.content or "", reason
        return self._finish(msgs, res, ctx, t0, start, n_effects)

    def _finish(self, msgs, res, ctx, t0, start, n_effects):
        res.latency_s = time.perf_counter() - t0
        res.side_effects = ctx.side_effects[n_effects:]
        # memory: keep only user turns + final answers (drop tool chatter to save tokens)
        history = [m for m in msgs[1:start]] + [msgs[start], {"role": "assistant", "content": res.answer}]
        return res, history
