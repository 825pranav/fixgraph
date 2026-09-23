"""S1: hybrid RAG = Qdrant dense + BM25 sparse, reciprocal-rank fusion, cross-encoder rerank."""

import time

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from fixgraph.embeddings import Embedder
from fixgraph.retrieval.base import RetrievalResult
from fixgraph.retrieval.bm25 import BM25Encoder
from fixgraph.retrieval.index import CHUNKS
from fixgraph.retrieval.rerank import Reranker


class HybridRetriever:
    name = "S1"

    def __init__(
        self,
        client: QdrantClient,
        embedder: Embedder,
        bm25: BM25Encoder,
        reranker: Reranker | None,
        chunk_docs: dict[str, str],
        candidates: int = 50,
        rerank_top: int = 30,
    ) -> None:
        self.client, self.embedder, self.bm25, self.reranker = client, embedder, bm25, reranker
        self.chunk_docs = chunk_docs
        self.candidates, self.rerank_top = candidates, rerank_top

    def candidates_for(self, question: str, limit: int) -> list[tuple[str, float]]:
        """Fused (chunk_id, rrf_score) candidates, best first."""
        dense = self.embedder.encode([question], query=True)[0].tolist()
        ids, vals = self.bm25.encode_query(question)
        prefetch = [qm.Prefetch(query=dense, using="dense", limit=limit)]
        if ids:
            prefetch.append(
                qm.Prefetch(
                    query=qm.SparseVector(indices=ids, values=vals), using="bm25", limit=limit
                )
            )
        res = self.client.query_points(
            CHUNKS,
            prefetch=prefetch,
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [(str((p.payload or {})["chunk_id"]), float(p.score)) for p in res.points]

    def rerank(
        self, question: str, cands: list[tuple[str, float]], k: int
    ) -> list[tuple[str, float]]:
        if self.reranker is None or not cands:
            return cands[:k]
        head = cands[: self.rerank_top]
        scores = self.reranker.score(question, [self.chunk_docs[c] for c, _ in head])
        ranked = sorted(zip([c for c, _ in head], scores, strict=True), key=lambda x: -x[1])
        return ranked[:k]

    def retrieve(self, question: str, k: int) -> RetrievalResult:
        t0 = time.perf_counter()
        cands = self.candidates_for(question, self.candidates)
        t1 = time.perf_counter()
        ranked = self.rerank(question, cands, k)
        t2 = time.perf_counter()
        return RetrievalResult(
            chunk_ids=[c for c, _ in ranked],
            scores=[s for _, s in ranked],
            timings={"search_s": t1 - t0, "rerank_s": t2 - t1, "total_s": t2 - t0},
        )
