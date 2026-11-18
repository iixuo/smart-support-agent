"""Streamlit demo: chat with the support agent and watch its tool calls.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works without pip install -e

import json
import uuid

import streamlit as st

from support_agent import db
from support_agent.agents import GraphAgent, RAGBaseline, ReActAgent
from support_agent.config import get_settings
from support_agent.llm import DeepSeekLLM
from support_agent.retriever import get_retriever
from support_agent.tools import ToolContext, build_registry

st.set_page_config(page_title="Smart Support Agent", page_icon="🛒", layout="wide")
settings = get_settings()

EXAMPLES = {
    "C001": ["我上周买的耳机不想要了，能退多少钱？", "我的手机壳到现在还没到，能怎么处理？", "好的，帮我申请退货吧"],
    "C002": ["键盘那单我只想退键盘，鼠标垫留着，能退多少？", "智能手表发货了吗？"],
    "C003": ["音箱坏了没声音，能退吗？", "帮我查一下订单 SO20261101"],
    "C004": ["台灯坏了，帮我申请保修维修", "你们太坑了，我要投诉！"],
    "C005": ["空气炸锅不想要了，还能七天无理由吗？"],
}


@st.cache_resource
def resources():
    return get_retriever(), build_registry(), DeepSeekLLM(settings)


try:
    retriever, registry, llm = resources()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    st.header("⚙️ 设置")
    customer = st.selectbox("当前登录用户", [c[0] for c in db.CUSTOMERS],
                            format_func=lambda c: f"{c} · {dict((x[0], x[1] + '（' + x[2] + '）') for x in db.CUSTOMERS)[c]}")
    mode = st.radio("实现", ["agent_langgraph", "agent_loop", "rag_context", "rag"],
                    captions=["LangGraph 状态图 + 记忆", "手写 Function Calling 循环", "RAG + 订单数据塞进 Prompt", "纯 RAG"])
    if st.button("🔄 新会话", use_container_width=True) or "sid" not in st.session_state \
            or st.session_state.get("key") != (customer, mode):
        st.session_state.update(sid=str(uuid.uuid4()), key=(customer, mode), chat=[], history=[],
                                conn=db.init_db(":memory:"))
        st.session_state.agent = {"agent_langgraph": lambda: GraphAgent(llm, registry, settings),
                                  "agent_loop": lambda: ReActAgent(llm, registry, settings),
                                  "rag_context": lambda: RAGBaseline(llm, True, settings=settings),
                                  "rag": lambda: RAGBaseline(llm, False, settings=settings)}[mode]()
    st.caption(f"模型 `{settings.model}` · 模拟日期 {settings.today} · 检索 {retriever.mode}")
    st.subheader("该用户的订单")
    for o in db.list_orders(st.session_state.conn, customer):
        st.caption(f"`{o['order_id']}` {o['items'][0]['name']} · {o['status']} · ¥{o['paid_amount']}")
    tickets = [dict(r) for r in st.session_state.conn.execute("SELECT ticket_id, order_id, category FROM tickets")]
    if tickets:
        st.subheader("已创建工单")
        st.json(tickets)

st.title("🛒 星河优选 · 智能客服 Agent")
st.caption("虚构商城 Demo：Function Calling 工具调用 + 混合检索 RAG + 售后业务工具")

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("trace"):
            with st.expander(m["trace_title"]):
                for i, c in enumerate(m["trace"], 1):
                    st.markdown(f"**{i}. `{c['name']}`** {'✅' if c['ok'] else '❌ ' + str(c['error'])}")
                    st.code(json.dumps(json.loads(c["arguments"]), ensure_ascii=False), language="json")
                    st.json(c["result"], expanded=False)

cols = st.columns(len(EXAMPLES[customer]))
clicked = next((q for col, q in zip(cols, EXAMPLES[customer]) if col.button(q, use_container_width=True)), None)
prompt = st.chat_input("输入你的问题…") or clicked

if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    ctx = ToolContext(st.session_state.conn, customer, settings.today, retriever)
    agent = st.session_state.agent
    with st.spinner("思考中…"):
        if isinstance(agent, GraphAgent):
            res = agent.run(prompt, ctx, thread_id=st.session_state.sid)
        else:
            res, st.session_state.history = agent.run(prompt, ctx, st.session_state.history)
    title = (f"🔧 {len(res.tool_calls)} 次工具调用 · {res.llm_calls} 次 LLM 调用 · {res.latency_s:.1f}s · "
             f"{res.prompt_tokens + res.completion_tokens} tokens · ¥{res.cost_cny:.4f} · stop={res.stop_reason}")
    st.session_state.chat.append({"role": "assistant", "content": res.answer, "trace": res.tool_calls,
                                  "trace_title": title})
    st.rerun()
