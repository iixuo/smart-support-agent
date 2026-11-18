"""Run the eval set against RAG baselines and the agents; write per-case runs + summary tables.

    python eval/run_eval.py                          # all systems, 1 repeat
    python eval/run_eval.py --systems agent_loop --limit 5
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works without pip install -e

import argparse
import json
import re
import statistics
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from support_agent import db
from support_agent.agents import GraphAgent, RAGBaseline, ReActAgent
from support_agent.config import get_settings
from support_agent.llm import DeepSeekLLM
from support_agent.retriever import get_retriever
from support_agent.tools import ToolContext, build_registry

HERE = Path(__file__).parent
RESULTS = HERE / "results"
SYSTEMS = ["rag", "rag_context", "agent_loop", "agent_langgraph"]
CATEGORY_ORDER = ["policy", "order", "multi_step", "action", "handoff", "safety", "multi_turn"]


class LockedRetriever:
    """ONNX embedder + BM25 shared across threads."""

    def __init__(self, r):
        self.r, self.lock = r, threading.Lock()

    def search(self, q, k=3):
        with self.lock:
            return self.r.search(q, k)


def normalize(text: str) -> str:
    t = re.sub(r"\s+", "", text or "")
    t = re.sub(r"(\d),(?=\d{3})", r"\1", t)
    t = re.sub(r"(\d)[-–—~～至到](\d)", r"\1-\2", t)
    return t.replace("：", ":").replace("¥", "").replace("￥", "")


def check_case(case: dict, answer: str, tool_names: list[str], conn, is_agent: bool) -> list[str]:
    """Return list of failure reasons (empty = pass)."""
    ck, ans, fails = case["checks"], normalize(answer), []
    for group in ck.get("any", []):
        if not any(normalize(alt) in ans for alt in group):
            fails.append(f"missing one of {group[:4]}")
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", ans)]
    for a in ck.get("amounts", []):
        if not any(abs(n - a) < 0.011 for n in nums):
            fails.append(f"missing amount {a}")
    for s in ck.get("none", []):
        if normalize(s) in ans:
            fails.append(f"leaked/forbidden '{s}'")
    if is_agent:
        for t in ck.get("tools", []):
            if t not in tool_names:
                fails.append(f"did not call {t}")
    for t in ck.get("no_tools", []):
        if t in tool_names:
            fails.append(f"should not call {t}")
    tickets = [dict(r) for r in conn.execute("SELECT * FROM tickets")]
    if ck.get("ticket") is False and tickets:
        fails.append("created a ticket it should not")
    elif isinstance(ck.get("ticket"), dict):
        want = ck["ticket"]
        ok = [t for t in tickets if t["order_id"] == want["order_id"]
              and (want["categories"] is None or t["category"] in want["categories"])]
        if not ok:
            fails.append(f"no matching ticket (got {[(t['order_id'], t['category']) for t in tickets]})")
    if ck.get("handoff") and not conn.execute("SELECT COUNT(*) FROM handoffs").fetchone()[0]:
        fails.append("no handoff to human")
    return fails


def make_system(name: str, llm, registry, settings):
    return {"rag": lambda: RAGBaseline(llm, with_orders=False, settings=settings),
            "rag_context": lambda: RAGBaseline(llm, with_orders=True, settings=settings),
            "agent_loop": lambda: ReActAgent(llm, registry, settings),
            "agent_langgraph": lambda: GraphAgent(llm, registry, settings)}[name]()


def run_case(system_name: str, case: dict, llm, registry, retriever, settings, rep: int) -> dict:
    conn = db.init_db(":memory:")
    ctx = ToolContext(conn, case["customer_id"], settings.today, retriever)
    system = make_system(system_name, llm, registry, settings)
    agg = dict(llm_calls=0, prompt_tokens=0, completion_tokens=0, cache_hit_tokens=0, cost_cny=0.0, latency_s=0.0)
    tool_calls, stop_reasons, answers, history, error = [], [], [], [], None
    try:
        for turn in case["turns"]:
            if isinstance(system, GraphAgent):
                res = system.run(turn, ctx, thread_id=f"{case['id']}-{rep}")
            else:
                res, history = system.run(turn, ctx, history)
            for k in agg:
                agg[k] += getattr(res, k)
            tool_calls += res.tool_calls
            stop_reasons.append(res.stop_reason)
            answers.append(res.answer)
    except Exception as e:  # API failure etc. — count as failed, keep going
        error = f"{type(e).__name__}: {e}"
    is_agent = system_name.startswith("agent")
    names = [c["name"] for c in tool_calls]
    fails = [f"exception: {error}"] if error else check_case(case, answers[-1] if answers else "", names, conn, is_agent)
    req = case["checks"].get("tools", [])
    return {"system": system_name, "case_id": case["id"], "category": case["category"], "repeat": rep,
            "success": not fails, "fail_reasons": fails, "answers": answers, **agg,
            "tool_calls": [{k: c[k] for k in ("name", "arguments", "ok", "error")} for c in tool_calls],
            "n_tool_calls": len(tool_calls), "tool_errors": sum(not c["ok"] for c in tool_calls),
            "required_tools_hit": sum(t in names for t in req), "required_tools": len(req),
            "stop_reasons": stop_reasons}


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def summarize(rows: list[dict]) -> dict:
    out = {}
    for sysname in SYSTEMS:
        rs = [r for r in rows if r["system"] == sysname]
        if not rs:
            continue
        lat = [r["latency_s"] for r in rs]
        by_cat = defaultdict(list)
        for r in rs:
            by_cat[r["category"]].append(r["success"])
        req = sum(r["required_tools"] for r in rs)
        out[sysname] = {
            "n": len(rs),
            "success_rate": sum(r["success"] for r in rs) / len(rs),
            "by_category": {c: sum(v) / len(v) for c, v in sorted(by_cat.items(), key=lambda x: CATEGORY_ORDER.index(x[0]))},
            "avg_llm_calls": statistics.mean(r["llm_calls"] for r in rs),
            "avg_tool_calls": statistics.mean(r["n_tool_calls"] for r in rs),
            "tool_error_rate": (sum(r["tool_errors"] for r in rs) / max(1, sum(r["n_tool_calls"] for r in rs))),
            "required_tool_recall": (sum(r["required_tools_hit"] for r in rs) / req) if req else None,
            "latency_mean_s": statistics.mean(lat), "latency_p50_s": pct(lat, 50), "latency_p95_s": pct(lat, 95),
            "avg_prompt_tokens": statistics.mean(r["prompt_tokens"] for r in rs),
            "avg_completion_tokens": statistics.mean(r["completion_tokens"] for r in rs),
            "cache_hit_ratio": sum(r["cache_hit_tokens"] for r in rs) / max(1, sum(r["prompt_tokens"] for r in rs)),
            "avg_cost_cny": statistics.mean(r["cost_cny"] for r in rs),
            "total_cost_cny": sum(r["cost_cny"] for r in rs),
            "stop_reasons": dict(Counter(s for r in rs for s in r["stop_reasons"])),
            "exceptions": sum(any(f.startswith("exception") for f in r["fail_reasons"]) for r in rs),
        }
    return out


def to_markdown(summary: dict, meta: dict) -> str:
    cats = CATEGORY_ORDER
    names = list(summary)
    L = [f"# Eval results\n", f"- model: `{meta['model']}` · cases: {meta['n_cases']} · repeats: {meta['repeats']} · "
         f"retriever: {meta['retriever']} · date: {meta['date']}\n",
         "## Overall\n",
         "| System | Success | Avg LLM calls | Avg tool calls | Tool recall | Latency P50 / P95 (s) | Avg tokens (in/out) | Cost / case (CNY) |",
         "|---|---|---|---|---|---|---|---|"]
    for n in names:
        s = summary[n]
        rec = f"{s['required_tool_recall']:.0%}" if s["required_tool_recall"] is not None and n.startswith("agent") else "–"
        L.append(f"| {n} | **{s['success_rate']:.1%}** | {s['avg_llm_calls']:.2f} | {s['avg_tool_calls']:.2f} | {rec} | "
                 f"{s['latency_p50_s']:.2f} / {s['latency_p95_s']:.2f} | {s['avg_prompt_tokens']:.0f} / "
                 f"{s['avg_completion_tokens']:.0f} | {s['avg_cost_cny']:.4f} |")
    L += ["\n## Success rate by category\n", "| System | " + " | ".join(cats) + " |", "|---" * (len(cats) + 1) + "|"]
    for n in names:
        bc = summary[n]["by_category"]
        L.append(f"| {n} | " + " | ".join(f"{bc[c]:.0%}" if c in bc else "–" for c in cats) + " |")
    L += ["\n## Agent robustness\n", "| System | Tool error rate | Stop reasons | Exceptions |", "|---|---|---|---|"]
    for n in names:
        s = summary[n]
        L.append(f"| {n} | {s['tool_error_rate']:.1%} | {s['stop_reasons']} | {s['exceptions']} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=SYSTEMS, choices=SYSTEMS)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="only first N cases (smoke test)")
    ap.add_argument("--only", nargs="*", help="case ids")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="", help="suffix for output files")
    args = ap.parse_args()

    settings = get_settings()
    cases = [json.loads(l) for l in (HERE / "cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.only:
        cases = [c for c in cases if c["id"] in args.only]
    if args.limit:
        cases = cases[: args.limit]
    llm, registry = DeepSeekLLM(settings), build_registry()
    base_retriever = get_retriever()
    retriever = LockedRetriever(base_retriever)

    jobs = [(s, c, r) for r in range(args.repeats) for s in args.systems for c in cases]
    print(f"running {len(jobs)} jobs ({len(cases)} cases x {len(args.systems)} systems x {args.repeats})", flush=True)
    t0, rows = time.time(), []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run_case, s, c, llm, registry, retriever, settings, r) for s, c, r in jobs]
        for i, f in enumerate(futs, 1):
            row = f.result()
            rows.append(row)
            mark = "✓" if row["success"] else "✗ " + "; ".join(row["fail_reasons"])[:120]
            print(f"[{i}/{len(jobs)}] {row['system']:<16} {row['case_id']} {mark}", flush=True)

    RESULTS.mkdir(exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    with open(RESULTS / f"runs{tag}.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = summarize(rows)
    meta = {"model": settings.model, "n_cases": len(cases), "repeats": args.repeats, "retriever": base_retriever.mode,
            "date": time.strftime("%Y-%m-%d"), "wall_time_s": round(time.time() - t0, 1),
            "pricing_cny_per_1M": {"in_hit": settings.price_in_hit, "in_miss": settings.price_in_miss,
                                   "out": settings.price_out}}
    (RESULTS / f"summary{tag}.json").write_text(json.dumps({"meta": meta, "summary": summary}, ensure_ascii=False,
                                                           indent=2), encoding="utf-8")
    md = to_markdown(summary, meta)
    (RESULTS / f"summary{tag}.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    sys.exit(main())
