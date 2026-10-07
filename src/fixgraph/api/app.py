"""FastAPI service (spec §12.2).

Two modes:
- real: loads the corpus/KG from data/, talks to the configured LLM (Ollama by default) and uses
  the hybrid index when `data/index/qdrant` exists (models load lazily on first request).
- fake: `LLM__BACKEND=fake` or no corpus on disk. A two-chunk demo corpus, a lexical retriever
  and a fake LLM that cites its first source, so the container runs with no GPU, data or keys.

Endpoints: /health (with LLM-cache hit rate), /retrieve, /answer, /graph/entity/{id},
/graph/subgraph, /links/suggestions (GNN gap candidates, always flagged `predicted`).
Used by: `fixgraph serve` (cli.py) via `create_app`.
Uses: answer.grounded, retrieval (hybrid, index, rerank, bm25), kg.store, ingest.store,
llm.factory, core.config.
"""

# Imports: FastAPI + pydantic for the HTTP layer, polars for graph lookups, and the answer,
# retrieval, KG and LLM pieces the endpoints glue together.
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
from fixgraph.llm.cache import CachedLLMClient
from fixgraph.llm.fake import FakeLLMClient
from fixgraph.retrieval.base import NoRetrieval, RetrievalResult, Retriever
from fixgraph.retrieval.bm25 import tokenize

logger = logging.getLogger(__name__)

# Two tiny built-in chunks used in fake mode, so the API runs with no data, GPU or model.
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

# Matches "[chunk_id]" header lines in the prompt; the fake model uses it to find a source to cite.
_SOURCE_ID_RE = re.compile(r"^\[([^\]\n]+)\]$", re.MULTILINE)


# Simple word-overlap retriever: used as S1 in fake mode and as a "lexical" fallback system.
class LexicalRetriever:
    """Term-overlap retriever over an in-memory chunk dict (fake mode and a no-index fallback)."""

    # Pre-tokenize every chunk once into a set of words so each query is just set overlaps.
    def __init__(self, chunk_text: dict[str, str], name: str = "lexical") -> None:
        self.name = name
        self.chunk_text = chunk_text
        self._tokens = {cid: set(tokenize(t)) for cid, t in chunk_text.items()}

    # Score each chunk by the share of question words it contains, best first (ties by chunk id),
    # and return the top k chunks with a non-zero score.
    def retrieve(self, question: str, k: int) -> RetrievalResult:
        q = set(tokenize(question))
        scored = sorted(
            ((len(q & toks) / (len(q) or 1), cid) for cid, toks in self._tokens.items()),
            key=lambda x: (-x[0], x[1]),
        )
        top = [(cid, s) for s, cid in scored if s > 0][:k]
        return RetrievalResult(chunk_ids=[c for c, _ in top], scores=[s for _, s in top])


# Fake LLM reply for /answer in fake mode: cite the first source id it sees, or abstain if none.
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


# Everything the endpoints need, built once at startup: the LLM client, retrievers by system name,
# chunk text, the optional graph, plus loaders for heavy retrievers that are built on first use.
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

    # Return the retriever for a system name. Heavy ones (S1) are built on first request: loader is
    # popped from `lazy`, called, and the result is kept in `retrievers` for later requests.
    def retriever(self, system: str) -> Retriever:
        if system not in self.retrievers and system in self.lazy:
            self.retrievers[system] = self.lazy.pop(system)()
        # Unknown system name -> HTTP 400 that lists the systems that do exist.
        if system not in self.retrievers:
            raise HTTPException(400, f"unknown system {system!r}; have {self.systems()}")
        return self.retrievers[system]

    # All system names the service can serve, whether already built or still lazy.
    def systems(self) -> list[str]:
        return sorted(set(self.retrievers) | set(self.lazy))


# State for fake mode: fake LLM, no-retrieval S0, and the lexical retriever over the demo chunks.
def fake_state() -> ServiceState:
    return ServiceState(
        llm=FakeLLMClient(responder=fake_answer_responder),
        retrievers={"S0": NoRetrieval(), "S1": LexicalRetriever(DEMO_CHUNKS, "S1")},
        chunk_text=dict(DEMO_CHUNKS),
        answer_model="fake",
        fake=True,
    )


# Return a function that builds the S1 hybrid retriever; called later so startup stays fast.
def _hybrid_factory(settings: Settings, chunk_text: dict[str, str]) -> Callable[[], Retriever]:
    # Load Qdrant index, embedder, BM25 stats and cross-encoder(s), then wire up HybridRetriever.
    def build() -> Retriever:
        # Imported here so heavy ML libraries load only when S1 is actually used.
        from fixgraph.embeddings import SentenceTransformerEmbedder
        from fixgraph.retrieval.hybrid import HybridRetriever
        from fixgraph.retrieval.index import load_bm25, open_client
        from fixgraph.retrieval.rerank import CrossEncoderReranker

        index_dir = settings.paths.index
        rs = settings.retrieval
        second = rs.second_reranker
        return HybridRetriever(
            open_client(index_dir),
            SentenceTransformerEmbedder(),
            load_bm25(index_dir),
            CrossEncoderReranker.named(rs.reranker),
            chunk_text,
            rerank_top=rs.rerank_top,
            second=CrossEncoderReranker.named(second) if second else None,
            w2=rs.second_weight,
            alpha=rs.first_stage_weight,
        )

    return build


# Build the real service state from settings and data/ on disk (falls back to fake mode).
def default_state(settings: Settings | None = None) -> ServiceState:
    settings = settings or load_settings()
    paths = settings.paths
    # No real LLM or no chunk file on disk -> run the demo/fake setup instead.
    if settings.llm.backend == "fake" or not paths.chunks.exists():
        logger.info("starting in fake mode")
        return fake_state()

    from fixgraph.ingest.store import read_chunks
    from fixgraph.llm.factory import build_llm_client

    # Load chunk text into an in-memory dict (raw text, no title) and the graph if it exists.
    chunk_text = {c.chunk_id: c.text for c in read_chunks(paths.chunks)}
    kg = read_kg(paths.kg) if (paths.kg / "nodes.parquet").exists() else None
    qdrant = paths.index / "qdrant"
    # Real state: cached LLM client, S0 and lexical retrievers, plus the Ollama URL used by /health.
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
    # Only register S1 if the Qdrant index folder exists; it is built on the first S1 request.
    if qdrant.is_dir():
        state.lazy["S1"] = _hybrid_factory(settings, chunk_text)
    return state


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------


# Request body for /retrieve and /answer: a non-empty question, a system name, and k in 1..50.
class QueryIn(BaseModel):
    question: str = Field(min_length=1)
    system: str = "S1"
    k: int = Field(default=8, ge=1, le=50)


# Response of /retrieve: ranked chunk ids, scores, an optional subgraph and stage timings.
class RetrieveOut(BaseModel):
    system: str
    chunk_ids: list[str]
    scores: list[float]
    subgraph: dict[str, Any] | None
    timings: dict[str, float]


# Response of /answer: answer text, cited sentences, citation ids, abstain flag and timings.
class AnswerOut(BaseModel):
    system: str
    answer: str
    sentences: list[AnswerSentence]
    citations: list[str]
    abstained: bool
    chunk_ids: list[str]
    subgraph: dict[str, Any] | None
    timings: dict[str, float]


# Request body for /graph/subgraph: between 1 and 200 node ids.
class SubgraphIn(BaseModel):
    node_ids: list[str] = Field(min_length=1, max_length=200)


# Look up one graph node by id and return it as a plain dict (or None if it does not exist).
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


# All graph edges that start or end at any of the given nodes, with their source chunk ids.
def _edges_touching(kg: KG, node_ids: list[str]) -> list[dict[str, Any]]:
    edges = kg.edges.filter(pl.col("src").is_in(node_ids) | pl.col("dst").is_in(node_ids))
    return edges.select(
        "src", "rel", "dst", "origin", "extraction_confidence", "source_chunk_ids"
    ).to_dicts()


def create_app(state: ServiceState | None = None) -> FastAPI:
    app = FastAPI(title="FixGraph", version="0.1.0")
    st = state or default_state()

    # GET /health: report mode, whether Qdrant and the LLM are reachable, and the cache hit rate.
    @app.get("/health")
    def health() -> dict[str, Any]:
        qdrant_ok = st.qdrant_path is not None and st.qdrant_path.is_dir()
        # In real mode, ping Ollama's /api/tags with a short timeout to see if the server is up.
        llm_ok = st.fake
        if not st.fake and st.llm_health_url:
            try:
                llm_ok = httpx.get(st.llm_health_url, timeout=2.0).status_code == 200
            except httpx.HTTPError:
                llm_ok = False
        # "degraded" means the LLM is unreachable; /retrieve still works in that case.
        return {
            "status": "ok" if (st.fake or llm_ok) else "degraded",
            "mode": "fake" if st.fake else "real",
            "kg_loaded": st.kg is not None,
            "qdrant_readable": qdrant_ok,
            "llm_reachable": llm_ok,
            "chunks": len(st.chunk_text),
            "systems": st.systems(),
            "llm_cache": st.llm.stats() if isinstance(st.llm, CachedLLMClient) else None,
        }

    # POST /retrieve: run only the chosen retriever and return the ranked chunk ids.
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

    # POST /answer: the main path. Retrieve top-k chunks, then generate a cited answer from them.
    @app.post("/answer")
    def answer(q: QueryIn) -> AnswerOut:
        # Step 1: pick the retriever by system name and get the ranked chunk ids.
        res = st.retriever(q.system).retrieve(q.question, q.k)
        # Step 2: grounded answer from those chunks; S0 runs closed book with no sources.
        ans = answer_question(
            st.llm,
            q.question,
            res.chunk_ids,
            st.chunk_text,
            st.answer_model,
            closed_book=q.system == "S0",
        )
        # Step 3: package answer, citations, the chunk ids shown, and retrieval + answer timings.
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

    # GET /graph/entity/{id}: one node from the unverified graph plus every edge touching it.
    @app.get("/graph/entity/{node_id}")
    def entity(node_id: str) -> dict[str, Any]:
        if st.kg is None:
            raise HTTPException(404, "no knowledge graph loaded")
        node = _node_row(st.kg, node_id)
        if node is None:
            raise HTTPException(404, f"unknown node {node_id!r}")
        return {"node": node, "edges": _edges_touching(st.kg, [node_id])}

    # POST /graph/subgraph: the given nodes and only the edges between them.
    @app.post("/graph/subgraph")
    def subgraph(body: SubgraphIn) -> dict[str, Any]:
        if st.kg is None:
            raise HTTPException(404, "no knowledge graph loaded")
        # Keep the ids that exist, then keep edges whose both ends are in that set.
        nodes = [n for nid in body.node_ids if (n := _node_row(st.kg, nid)) is not None]
        ids = [n["node_id"] for n in nodes]
        edges = [e for e in _edges_touching(st.kg, ids) if e["src"] in ids and e["dst"] in ids]
        return {"nodes": nodes, "edges": edges}

    # GET /links/suggestions: GNN-predicted fixes for a symptom, read from the gap candidates file.
    @app.get("/links/suggestions")
    def suggestions(symptom_id: str) -> list[dict[str, Any]]:
        """GNN-predicted fixes. Always flagged `predicted`: routing hints, never evidence."""
        if st.gap_file is None or not st.gap_file.exists():
            return []
        out: list[dict[str, Any]] = []
        # Scan the JSONL file line by line and return rows for this symptom, tagged as "predicted".
        for line in st.gap_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("symptom_id") == symptom_id:
                out.append({**row, "origin": "predicted"})
        return out

    return app
