"""S1: hybrid RAG = Qdrant dense + BM25 sparse, reciprocal-rank fusion, cross-encoder rerank.

Used by: bench/cli.py (S1, and as the fallback inside S2/S3), retrieval.graphrag, api/app.py.
Uses: retrieval.index (chunk collection), retrieval.bm25, retrieval.rerank, embeddings.
"""

import math
import time
from collections.abc import Sequence

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from fixgraph.embeddings import Embedder
from fixgraph.retrieval.base import RetrievalResult
from fixgraph.retrieval.bm25 import BM25Encoder
from fixgraph.retrieval.index import CHUNKS
from fixgraph.retrieval.rerank import Reranker


def zscore(xs: Sequence[float]) -> list[float]:
    """Standardise scores within one pool (constant pools map to 0)."""
    if not xs:
        return []
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))
    return [(x - m) / sd if sd > 0 else 0.0 for x in xs]


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
        second: Reranker | None = None,
        w2: float = 0.0,
        alpha: float = 0.0,
    ) -> None:
        """`second` / `w2`: optional second reranker blended in with weight w2; `alpha`: weight
        of the first-stage RRF score. Blends use z-normalised scores within the pool (D39)."""
        self.client, self.embedder, self.bm25, self.reranker = client, embedder, bm25, reranker
        self.chunk_docs = chunk_docs
        self.candidates, self.rerank_top = max(candidates, rerank_top), rerank_top
        self.second, self.w2, self.alpha = second, w2, alpha

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
        texts = [self.chunk_docs[c] for c, _ in head]
        scores = self.reranker.score(question, texts)
        if self.second is not None or self.alpha:
            scores = zscore(scores)
            if self.second is not None:
                z2 = zscore(self.second.score(question, texts))
                scores = [(1 - self.w2) * a + self.w2 * b for a, b in zip(scores, z2, strict=True)]
            if self.alpha:
                zr = zscore([s for _, s in head])
                scores = [a + self.alpha * b for a, b in zip(scores, zr, strict=True)]
        order = sorted(range(len(head)), key=lambda i: (-scores[i], i))
        return [(head[i][0], scores[i]) for i in order[:k]]

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
