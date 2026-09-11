"""Shared retrieval interface (spec §10). Every system (S0-S5) returns a RetrievalResult.

Defines the Retriever protocol, RetrievalResult / Subgraph types and S0 (NoRetrieval,
closed-book). Implemented by retrieval.hybrid and retrieval.graphrag; consumed by bench.run,
bench/cli.py and api/app.py. No fixgraph imports.
"""

from typing import Protocol

from pydantic import BaseModel, Field


class SubgraphEdge(BaseModel):
    src: str
    rel: str
    dst: str
    origin: str = "extracted"  # predicted edges are routing-only, never evidence


class Subgraph(BaseModel):
    nodes: list[str] = Field(default_factory=lambda: list[str]())
    edges: list[SubgraphEdge] = Field(default_factory=lambda: list[SubgraphEdge]())
    paths: list[list[str]] = Field(default_factory=lambda: list[list[str]]())  # serialized hops
    seeds: list[str] = Field(default_factory=lambda: list[str]())


class RetrievalResult(BaseModel):
    chunk_ids: list[str] = Field(default_factory=lambda: list[str]())  # ranked
    scores: list[float] = Field(default_factory=lambda: list[float]())
    subgraph: Subgraph | None = None
    timings: dict[str, float] = Field(default_factory=lambda: dict[str, float]())
    llm_calls: int = 0


class Retriever(Protocol):
    name: str

    def retrieve(self, question: str, k: int) -> RetrievalResult: ...


class NoRetrieval:
    """S0: closed-book."""

    name = "S0"

    def retrieve(self, question: str, k: int) -> RetrievalResult:
        return RetrievalResult()
