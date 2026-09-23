"""FastAPI service (spec §12.2).

Two modes:
- real: loads the corpus/KG from data/, talks to the configured LLM (Ollama by default) and uses
  the hybrid index when `data/index/qdrant` exists (models load lazily on first request).
- fake: `LLM__BACKEND=fake` or no corpus on disk. A two-chunk demo corpus, a lexical retriever
  and a fake LLM that cites its first source, so the container runs with no GPU, data or keys.
"""

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import polars as pl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fixgraph.answer.grounded import AnswerOutput, AnswerSentence, answer_question
from fixgraph.core.config import Settings, load_settings
from fixgraph.kg.store import KG, read_kg
from fixgraph.llm.base import LLMClient, LLMRequest
from fixgraph.llm.fake import FakeLLMClient
from fixgraph.retrieval.base import NoRetrieval, RetrievalResult, Retriever
from fixgraph.retrieval.bm25 import tokenize

logger = logging.getLogger(__name__)

DEMO_CHUNKS: dict[str, str] = {
    "demo:0:0": (
        "If your Apple Watch won't pair with your iPhone\n"
        "Make sure that Bluetooth is turned on on your iPhone. Unpair Apple Watch in the "
        "Apple Watch app, then pair it again."
    ),
    "demo:1:0": (
        "If your AirPods won't charge\n"
        "Put your AirPods in the charging case and connect the case to power for 15 minutes."
    ),
}

_SOURCE_ID_RE = re.compile(r"^\[([^\]\n]+)\]$", re.MULTILINE)


class LexicalRetriever:
    """Term-overlap retriever over an in-memory chunk dict (fake mode and a no-index fallback)."""

    def __init__(self, chunk_text: dict[str, str], name: str = "lexical") -> None:
        self.name = name
        self.chunk_text = chunk_text
        self._tokens = {cid: set(tokenize(t)) for cid, t in chunk_text.items()}

    def retrieve(self, question: str, k: int) -> RetrievalResult:
        q = set(tokenize(question))
        scored = sorted(
            ((len(q & toks) / (len(q) or 1), cid) for cid, toks in self._tokens.items()),
            key=lambda x: (-x[0], x[1]),
        )
        top = [(cid, s) for s, cid in scored if s > 0][:k]
        return RetrievalResult(chunk_ids=[c for c, _ in top], scores=[s for _, s in top])


def fake_answer_responder(request: LLMRequest) -> str:
    """Valid AnswerOutput citing the first source in the prompt; abstains without sources."""
    prompt = request.messages[-1].content
    ids = _SOURCE_ID_RE.findall(prompt)
    if not ids:
        out = AnswerOutput(abstain=True, abstain_reason="no sources (fake mode)")
    else:
        out = AnswerOutput(
            abstain=False,
            sentences=[
                AnswerSentence(text="Follow the steps in the cited article.", citations=[ids[0]])
            ],
        )
    return out.model_dump_json()


@dataclass
class ServiceState:
    llm: LLMClient
    retrievers: dict[str, Retriever]
    chunk_text: dict[str, str]
    kg: KG | None = None
    answer_model: str = "qwen3:4b"
    fake: bool = False
    qdrant_path: Path | None = None
    llm_health_url: str | None = None
    gap_file: Path | None = None
    lazy: dict[str, Callable[[], Retriever]] = field(
        default_factory=lambda: dict[str, Callable[[], Retriever]]()
    )

    def retriever(self, system: str) -> Retriever:
        if system not in self.retrievers and system in self.lazy:
            self.retrievers[system] = self.lazy.pop(system)()
        if system not in self.retrievers:
            raise HTTPException(400, f"unknown system {system!r}; have {self.systems()}")
        return self.retrievers[system]

    def systems(self) -> list[str]:
        return sorted(set(self.retrievers) | set(self.lazy))


def fake_state() -> ServiceState:
    return ServiceState(
        llm=FakeLLMClient(responder=fake_answer_responder),
        retrievers={"S0": NoRetrieval(), "S1": LexicalRetriever(DEMO_CHUNKS, "S1")},
        chunk_text=dict(DEMO_CHUNKS),
        answer_model="fake",
        fake=True,
    )


def _hybrid_factory(settings: Settings, chunk_text: dict[str, str]) -> Callable[[], Retriever]:
    def build() -> Retriever:
        from fixgraph.embeddings import SentenceTransformerEmbedder
        from fixgraph.retrieval.hybrid import HybridRetriever
        from fixgraph.retrieval.index import load_bm25, open_client
        from fixgraph.retrieval.rerank import CrossEncoderReranker

        index_dir = settings.paths.index
        return HybridRetriever(
            open_client(index_dir),
            SentenceTransformerEmbedder(),
            load_bm25(index_dir),
            CrossEncoderReranker(),
            chunk_text,
        )

    return build


def default_state(settings: Settings | None = None) -> ServiceState:
    settings = settings or load_settings()
    paths = settings.paths
    if settings.llm.backend == "fake" or not paths.chunks.exists():
        logger.info("starting in fake mode")
        return fake_state()

    from fixgraph.ingest.store import read_chunks
    from fixgraph.llm.factory import build_llm_client

    chunk_text = {c.chunk_id: c.text for c in read_chunks(paths.chunks)}
    kg = read_kg(paths.kg) if (paths.kg / "nodes.parquet").exists() else None
    qdrant = paths.index / "qdrant"
    state = ServiceState(
        llm=build_llm_client(settings),
        retrievers={"S0": NoRetrieval(), "lexical": LexicalRetriever(chunk_text)},
        chunk_text=chunk_text,
        kg=kg,
        answer_model=settings.llm.answer_model,
        qdrant_path=qdrant,
        llm_health_url=(
            settings.llm.base_url.rstrip("/").removesuffix("/v1") + "/api/tags"
            if settings.llm.backend == "ollama"
            else None
        ),
        gap_file=paths.results / "gnn" / "gap_candidates.jsonl",
    )
    if qdrant.is_dir():
        state.lazy["S1"] = _hybrid_factory(settings, chunk_text)
    return state


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------


class QueryIn(BaseModel):
    question: str = Field(min_length=1)
    system: str = "S1"
    k: int = Field(default=8, ge=1, le=50)


class RetrieveOut(BaseModel):
    system: str
    chunk_ids: list[str]
    scores: list[float]
    subgraph: dict[str, Any] | None
    timings: dict[str, float]


class AnswerOut(BaseModel):
    system: str
    answer: str
    sentences: list[AnswerSentence]
    citations: list[str]
    abstained: bool
    chunk_ids: list[str]
    subgraph: dict[str, Any] | None
    timings: dict[str, float]


class SubgraphIn(BaseModel):
    node_ids: list[str] = Field(min_length=1, max_length=200)


def _node_row(kg: KG, node_id: str) -> dict[str, Any] | None:
    rows = kg.nodes.filter(pl.col("node_id") == node_id).to_dicts()
    if not rows:
        return None
    row = rows[0]
    return {
        "node_id": row["node_id"],
        "label": row["label"],
        "text": row["canonical_text"],
        "props": json.loads(row["props_json"]),
    }


def _edges_touching(kg: KG, node_ids: list[str]) -> list[dict[str, Any]]:
    edges = kg.edges.filter(pl.col("src").is_in(node_ids) | pl.col("dst").is_in(node_ids))
    return edges.select(
        "src", "rel", "dst", "origin", "extraction_confidence", "source_chunk_ids"
    ).to_dicts()


def create_app(state: ServiceState | None = None) -> FastAPI:
    app = FastAPI(title="FixGraph", version="0.1.0")
    st = state or default_state()

    @app.get("/health")
    def health() -> dict[str, Any]:
        qdrant_ok = st.qdrant_path is not None and st.qdrant_path.is_dir()
        llm_ok = st.fake
        if not st.fake and st.llm_health_url:
            try:
                llm_ok = httpx.get(st.llm_health_url, timeout=2.0).status_code == 200
            except httpx.HTTPError:
                llm_ok = False
        return {
            "status": "ok" if (st.fake or llm_ok) else "degraded",
            "mode": "fake" if st.fake else "real",
            "kg_loaded": st.kg is not None,
            "qdrant_readable": qdrant_ok,
            "llm_reachable": llm_ok,
            "chunks": len(st.chunk_text),
            "systems": st.systems(),
        }

    @app.post("/retrieve")
    def retrieve(q: QueryIn) -> RetrieveOut:
        res = st.retriever(q.system).retrieve(q.question, q.k)
        return RetrieveOut(
            system=q.system,
            chunk_ids=res.chunk_ids,
            scores=res.scores,
            subgraph=res.subgraph.model_dump() if res.subgraph else None,
            timings=res.timings,
        )

    @app.post("/answer")
    def answer(q: QueryIn) -> AnswerOut:
        res = st.retriever(q.system).retrieve(q.question, q.k)
        ans = answer_question(
            st.llm,
            q.question,
            res.chunk_ids,
            st.chunk_text,
            st.answer_model,
            closed_book=q.system == "S0",
        )
        return AnswerOut(
            system=q.system,
            answer=ans.text,
            sentences=ans.sentences,
            citations=ans.cited_chunk_ids,
            abstained=ans.abstained,
            chunk_ids=ans.context_chunk_ids,
            subgraph=res.subgraph.model_dump() if res.subgraph else None,
            timings={**res.timings, "answer_s": ans.latency_s},
        )

    @app.get("/graph/entity/{node_id}")
    def entity(node_id: str) -> dict[str, Any]:
        if st.kg is None:
            raise HTTPException(404, "no knowledge graph loaded")
        node = _node_row(st.kg, node_id)
        if node is None:
            raise HTTPException(404, f"unknown node {node_id!r}")
        return {"node": node, "edges": _edges_touching(st.kg, [node_id])}

    @app.post("/graph/subgraph")
    def subgraph(body: SubgraphIn) -> dict[str, Any]:
        if st.kg is None:
            raise HTTPException(404, "no knowledge graph loaded")
        nodes = [n for nid in body.node_ids if (n := _node_row(st.kg, nid)) is not None]
        ids = [n["node_id"] for n in nodes]
        edges = [e for e in _edges_touching(st.kg, ids) if e["src"] in ids and e["dst"] in ids]
        return {"nodes": nodes, "edges": edges}

    @app.get("/links/suggestions")
    def suggestions(symptom_id: str) -> list[dict[str, Any]]:
        """GNN-predicted fixes. Always flagged `predicted`: routing hints, never evidence."""
        if st.gap_file is None or not st.gap_file.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in st.gap_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("symptom_id") == symptom_id:
                out.append({**row, "origin": "predicted"})
        return out

    return app
