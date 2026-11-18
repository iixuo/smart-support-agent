"""Hybrid retriever over policy docs: BM25 (jieba) + dense (bge-small-zh via fastembed), fused by RRF.

Docs are split by `##` section; each chunk keeps its doc title for citation.
Dense retrieval is optional — falls back to BM25 only if fastembed / the model is unavailable.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from .config import POLICY_DIR

jieba.setLogLevel(logging.WARNING)
log = logging.getLogger(__name__)

EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："  # bge-zh query prefix
_PUNCT = re.compile(r"[\s，。、；：！？（）()“”\"'《》【】\-–—/、,.;:!?]+")


@dataclass
class Chunk:
    chunk_id: str
    doc: str
    section: str
    text: str


def load_chunks(policy_dir: Path = POLICY_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(policy_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        title = raw.splitlines()[0].lstrip("# ").strip()
        for i, part in enumerate(re.split(r"^## ", raw, flags=re.M)[1:]):
            section, _, body = part.partition("\n")
            chunks.append(Chunk(f"{path.stem}#{i}", title, section.strip(), body.strip()))
    return chunks


def _tok(text: str) -> list[str]:
    return [t for t in jieba.lcut(_PUNCT.sub(" ", text.lower())) if t.strip()]


class HybridRetriever:
    def __init__(self, chunks: list[Chunk] | None = None, use_dense: bool = True, rrf_k: int = 60):
        self.chunks = chunks or load_chunks()
        self.rrf_k = rrf_k
        self.bm25 = BM25Okapi([_tok(f"{c.section} {c.text}") for c in self.chunks])
        self.embedder = None
        self.emb = None
        if use_dense:
            try:
                from fastembed import TextEmbedding
                self.embedder = TextEmbedding(EMBED_MODEL)
                vecs = np.array(list(self.embedder.embed([f"{c.section}：{c.text}" for c in self.chunks])))
                self.emb = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
            except Exception as e:  # pragma: no cover - depends on network/model cache
                log.warning("dense retrieval disabled (%s); using BM25 only", e)
                self.embedder = None

    @property
    def mode(self) -> str:
        return "hybrid(bm25+bge-small-zh, RRF)" if self.embedder else "bm25"

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        bm25_rank = np.argsort(-self.bm25.get_scores(_tok(query)))
        rankings = [bm25_rank]
        if self.embedder is not None:
            q = np.array(list(self.embedder.embed([QUERY_INSTRUCTION + query])))[0]
            rankings.append(np.argsort(-(self.emb @ (q / np.linalg.norm(q)))))
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, idx in enumerate(ranking):
                fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (self.rrf_k + rank + 1)
        best = sorted(fused, key=fused.get, reverse=True)[:top_k]
        return [{"source": f"{self.chunks[i].doc} / {self.chunks[i].section}", "content": self.chunks[i].text,
                 "score": round(fused[i], 4)} for i in best]


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever()
