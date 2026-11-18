# Eval results

- model: `deepseek-flash` · cases: 62 · repeats: 3 · retriever: hybrid(bm25+bge-small-zh, RRF) · date: 2026-11-18

## Overall

| System | Success | Avg LLM calls | Avg tool calls | Tool recall | Latency P50 / P95 (s) | Avg tokens (in/out) | Cost / case (CNY) |
|---|---|---|---|---|---|---|---|
| rag | **45.7%** | 1.08 | 0.00 | – | 2.23 / 6.02 | 390 / 268 | 0.0027 |
| rag_context | **76.3%** | 1.08 | 0.00 | – | 2.13 / 5.57 | 976 / 266 | 0.0028 |
| agent_loop | **100.0%** | 2.97 | 2.68 | 100% | 4.07 / 8.50 | 6470 / 470 | 0.0057 |
| agent_langgraph | **99.5%** | 2.96 | 2.60 | 100% | 3.81 / 7.61 | 6511 / 449 | 0.0055 |

## Success rate by category

| System | policy | order | multi_step | action | handoff | safety | multi_turn |
|---|---|---|---|---|---|---|---|
| rag | 100% | 10% | 44% | 0% | 0% | 72% | 40% |
| rag_context | 100% | 100% | 88% | 0% | 0% | 89% | 80% |
| agent_loop | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| agent_langgraph | 100% | 100% | 100% | 95% | 100% | 100% | 100% |

## Agent robustness

| System | Tool error rate | Stop reasons | Exceptions |
|---|---|---|---|
| rag | 0.0% | {'final_answer': 201} | 0 |
| rag_context | 0.0% | {'final_answer': 201} | 0 |
| agent_loop | 2.4% | {'final_answer': 201} | 0 |
| agent_langgraph | 2.5% | {'final_answer': 201} | 0 |
