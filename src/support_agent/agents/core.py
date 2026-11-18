"""Pieces shared by the hand-written loop and the LangGraph agent: tracing, guarded tool execution, stop rules."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from ..config import Settings
from ..llm import LLMResponse, ToolCall
from ..tools.base import ToolContext, ToolRegistry

MAX_TOOL_RESULT_CHARS = 4000
MAX_IDENTICAL_CALLS = 2       # same tool + same args more than this -> loop detected
MAX_CONSECUTIVE_ERRORS = 3    # tool errors in a row -> give up gracefully


@dataclass
class RunResult:
    answer: str
    stop_reason: str = "final_answer"  # final_answer | max_steps | timeout | repeated_call | too_many_errors
    llm_calls: int = 0
    tool_calls: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    cost_cny: float = 0.0
    latency_s: float = 0.0
    side_effects: list[dict] = field(default_factory=list)

    def add_usage(self, r: LLMResponse, s: Settings):
        self.llm_calls += 1
        self.prompt_tokens += r.prompt_tokens
        self.completion_tokens += r.completion_tokens
        self.cache_hit_tokens += r.cache_hit_tokens
        self.cost_cny += s.cost_cny(r.prompt_tokens, r.completion_tokens, r.cache_hit_tokens)

    @property
    def tool_names(self) -> list[str]:
        return [c["name"] for c in self.tool_calls]


class Guard:
    """Termination conditions beyond 'model stopped calling tools'."""

    def __init__(self, s: Settings):
        self.max_steps, self.deadline = s.max_steps, time.perf_counter() + s.timeout_s
        self.seen: dict[str, int] = {}
        self.consecutive_errors = 0

    def check_before_llm(self, steps_done: int) -> str | None:
        if steps_done >= self.max_steps:
            return "max_steps"
        if time.perf_counter() > self.deadline:
            return "timeout"
        return None

    def check_calls(self, calls: list[ToolCall]) -> str | None:
        for c in calls:
            key = f"{c.name}:{_canon(c.arguments)}"
            self.seen[key] = self.seen.get(key, 0) + 1
            if self.seen[key] > MAX_IDENTICAL_CALLS:
                return "repeated_call"
        return None

    def record_results(self, results: list[dict]) -> str | None:
        for r in results:
            self.consecutive_errors = 0 if r["ok"] else self.consecutive_errors + 1
        return "too_many_errors" if self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS else None


def _canon(raw: str) -> str:
    try:
        return json.dumps(json.loads(raw), sort_keys=True, ensure_ascii=False)
    except Exception:
        return raw


def execute_tool_calls(calls: list[ToolCall], registry: ToolRegistry, ctx: ToolContext,
                       result: RunResult) -> tuple[list[dict], list[dict]]:
    """Run tool calls, return (tool messages for the LLM, raw results)."""
    msgs, raws = [], []
    for c in calls:
        t0 = time.perf_counter()
        out = registry.execute(c.name, c.arguments, ctx)
        result.tool_calls.append({"name": c.name, "arguments": c.arguments, "ok": out["ok"],
                                  "error": None if out["ok"] else out["error"]["type"],
                                  "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                                  "result": out})
        content = json.dumps(out, ensure_ascii=False, default=str)
        if len(content) > MAX_TOOL_RESULT_CHARS:
            content = content[:MAX_TOOL_RESULT_CHARS] + "...(truncated)"
        msgs.append({"role": "tool", "tool_call_id": c.id, "content": content})
        raws.append(out)
    return msgs, raws
