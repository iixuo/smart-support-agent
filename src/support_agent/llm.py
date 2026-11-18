"""Thin LLM client over the OpenAI-compatible API (DeepSeek), plus a scripted FakeLLM for offline tests."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import Settings, get_settings


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string as produced by the model


@dataclass
class LLMResponse:
    content: str | None
    reasoning: str | None = None  # thinking-mode reasoning_content, must be echoed back within a tool loop
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    latency_s: float = 0.0

    def as_message(self) -> dict:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.reasoning:
            msg["reasoning_content"] = self.reasoning
        if self.tool_calls:
            msg["tool_calls"] = [{"id": c.id, "type": "function",
                                  "function": {"name": c.name, "arguments": c.arguments}} for c in self.tool_calls]
        return msg


class ChatModel(Protocol):
    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             tool_choice: str = "auto") -> LLMResponse: ...


class DeepSeekLLM:
    def __init__(self, settings: Settings | None = None, temperature: float = 0.0):
        from openai import OpenAI
        self.s = settings or get_settings()
        if not self.s.api_key or self.s.api_key.startswith("sk-your"):
            raise RuntimeError("DEEPSEEK_API_KEY missing: copy .env.example to .env and fill it in")
        # SDK handles retry with exponential backoff on 429/5xx/timeouts
        self.client = OpenAI(api_key=self.s.api_key, base_url=self.s.base_url, max_retries=3, timeout=60)
        self.temperature = temperature

    def chat(self, messages, tools=None, tool_choice="auto") -> LLMResponse:
        kwargs: dict[str, Any] = dict(model=self.s.model, messages=messages, temperature=self.temperature)
        if tools:
            kwargs.update(tools=tools, tool_choice=tool_choice)
        if self.s.extra_body:
            kwargs["extra_body"] = json.loads(self.s.extra_body)
        t0 = time.perf_counter()
        r = self.client.chat.completions.create(**kwargs)
        dt = time.perf_counter() - t0
        m = r.choices[0].message
        u = r.usage
        hit = getattr(u, "prompt_cache_hit_tokens", None) or 0
        if not hit and getattr(u, "prompt_tokens_details", None):
            hit = getattr(u.prompt_tokens_details, "cached_tokens", 0) or 0
        return LLMResponse(
            content=m.content, reasoning=getattr(m, "reasoning_content", None),
            tool_calls=[ToolCall(c.id, c.function.name, c.function.arguments or "{}") for c in (m.tool_calls or [])],
            prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens, cache_hit_tokens=hit, latency_s=dt)


class FakeLLM:
    """Replays a script of responses. Each item: str (final answer) or list of (tool_name, args_dict|raw_str)."""

    def __init__(self, script: list):
        self.script = list(script)
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, tool_choice="auto") -> LLMResponse:
        self.calls.append({"messages": [dict(m) for m in messages], "tool_choice": tool_choice})
        if tool_choice == "none" or not self.script:
            return LLMResponse(content="(forced final answer)", prompt_tokens=10, completion_tokens=5)
        step = self.script.pop(0)
        if isinstance(step, str):
            return LLMResponse(content=step, prompt_tokens=10, completion_tokens=5)
        calls = [ToolCall(f"call_{len(self.calls)}_{i}", name, a if isinstance(a, str) else json.dumps(a))
                 for i, (name, a) in enumerate(step)]
        return LLMResponse(content=None, tool_calls=calls, prompt_tokens=10, completion_tokens=5)
