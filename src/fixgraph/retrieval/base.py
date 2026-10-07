"""Shared retrieval interface (spec §10). Every system (S0-S5) returns a RetrievalResult.

Defines the Retriever protocol, RetrievalResult / Subgraph types and S0 (NoRetrieval,
closed-book). Implemented by retrieval.hybrid and retrieval.graphrag; consumed by bench.run,
bench/cli.py and api/app.py. No fixgraph imports.
"""

# Imports: Protocol for the retriever interface, pydantic for the result types.
from typing import Protocol

from pydantic import BaseModel, Field


# One graph edge (source, relation, target) shown alongside graph-based retrieval results.
class SubgraphEdge(BaseModel):
    src: str
    rel: str
    dst: str
    origin: str = "extracted"  # predicted edges are routing-only, never evidence


# The slice of the knowledge graph a graph retriever used: nodes, edges, paths and seed nodes.
class Subgraph(BaseModel):
    nodes: list[str] = Field(default_factory=lambda: list[str]())
    edges: list[SubgraphEdge] = Field(default_factory=lambda: list[SubgraphEdge]())
    paths: list[list[str]] = Field(default_factory=lambda: list[list[str]]())  # serialized hops
    seeds: list[str] = Field(default_factory=lambda: list[str]())


# What every retriever returns: ranked chunk ids, their scores, an optional subgraph and timings.
class RetrievalResult(BaseModel):
    chunk_ids: list[str] = Field(default_factory=lambda: list[str]())  # ranked
    scores: list[float] = Field(default_factory=lambda: list[float]())
    subgraph: Subgraph | None = None
    timings: dict[str, float] = Field(default_factory=lambda: dict[str, float]())
    llm_calls: int = 0


# Common interface for all systems S0-S4: question and k in, RetrievalResult out.
class Retriever(Protocol):
    name: str

    def retrieve(self, question: str, k: int) -> RetrievalResult: ...


# S0 baseline: retrieves nothing, so the answer model must answer from its own knowledge.
class NoRetrieval:
    """S0: closed-book."""

    name = "S0"

    # Always returns an empty result; used as the closed-book comparison point.
    def retrieve(self, question: str, k: int) -> RetrievalResult:
        return RetrievalResult()
