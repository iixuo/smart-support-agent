"""Baselines without tool calling — one retrieval + one LLM call per turn.

- rag:          policy retrieval only (a typical FAQ / knowledge-base bot)
- rag_context:  policy retrieval + the user's full order & shipping records stuffed into the prompt
                (a strong baseline: same facts as the agent, but no tools, no actions, no refund engine)
"""
from __future__ import annotations

import json
import time

from .. import db
from ..config import Settings, get_settings
from ..llm import ChatModel
from ..tools.base import ToolContext
from .core import RunResult
from .prompts import RAG_SYSTEM


class RAGBaseline:
    def __init__(self, llm: ChatModel, with_orders: bool = False, top_k: int = 4, settings: Settings | None = None):
        self.llm, self.with_orders, self.top_k, self.s = llm, with_orders, top_k, settings or get_settings()
        self.name = "rag_context" if with_orders else "rag"

    def run(self, user_msg: str, ctx: ToolContext, history: list[dict] | None = None) -> tuple[RunResult, list[dict]]:
        t0 = time.perf_counter()
        history = history or []
        query = " ".join([m["content"] for m in history if m["role"] == "user"][-1:] + [user_msg])
        docs = ctx.retriever.search(query, self.top_k)
        context = "\n\n".join(f"[{d['source']}]\n{d['content']}" for d in docs)
        orders = ""
        if self.with_orders:
            data = db.list_orders(ctx.conn, ctx.customer_id)
            level = db.get_customer(ctx.conn, ctx.customer_id)["level"]
            orders = (f"\n【当前用户信息】会员等级: {level}\n【当前用户全部订单（含物流）】\n"
                      + json.dumps(data, ensure_ascii=False, default=str))
        system = RAG_SYSTEM.format(today=ctx.today.isoformat(), customer_id=ctx.customer_id,
                                   context=context, orders=orders)
        msgs = [{"role": "system", "content": system}, *history, {"role": "user", "content": user_msg}]
        r = self.llm.chat(msgs)
        res = RunResult(answer=r.content or "")
        res.add_usage(r, self.s)
        res.latency_s = time.perf_counter() - t0
        return res, [*history, {"role": "user", "content": user_msg}, {"role": "assistant", "content": res.answer}]
