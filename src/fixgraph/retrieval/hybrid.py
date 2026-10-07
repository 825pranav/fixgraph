"""S1: hybrid RAG = Qdrant dense + BM25 sparse, reciprocal-rank fusion, cross-encoder rerank.

Used by: bench/cli.py (S1, and as the fallback inside S2/S3), retrieval.graphrag, api/app.py.
Uses: retrieval.index (chunk collection), retrieval.bm25, retrieval.rerank, embeddings.
"""

# Imports: Qdrant client and query models, plus the embedder, BM25 encoder and reranker it combines.
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


# Turns a list of scores into z-scores, so scores from different models can be blended fairly.
def zscore(xs: Sequence[float]) -> list[float]:
    """Standardise scores within one pool (constant pools map to 0)."""
    if not xs:
        return []
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))
    return [(x - m) / sd if sd > 0 else 0.0 for x in xs]


# S1 retriever used by the API and the harness: dense + BM25 search fused by RRF in Qdrant,
# then a cross-encoder reranks the short list and the best k chunks are returned.
class HybridRetriever:
    name = "S1"

    # Stores the Qdrant client, models and chunk text lookup, plus pool sizes and blend weights.
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
        # Make sure the candidate pool is at least as large as the list the reranker will read.
        self.candidates, self.rerank_top = max(candidates, rerank_top), rerank_top
        self.second, self.w2, self.alpha = second, w2, alpha

    # Question in, up to `limit` (chunk_id, rrf_score) pairs out, best first.
    def candidates_for(self, question: str, limit: int) -> list[tuple[str, float]]:
        """Fused (chunk_id, rrf_score) candidates, best first."""
        # Encode the question twice: a dense vector for meaning and BM25 term ids for keywords.
        dense = self.embedder.encode([question], query=True)[0].tolist()
        ids, vals = self.bm25.encode_query(question)
        # Set up both sub-searches; BM25 is skipped if the question has no usable terms.
        prefetch = [qm.Prefetch(query=dense, using="dense", limit=limit)]
        if ids:
            prefetch.append(
                qm.Prefetch(
                    query=qm.SparseVector(indices=ids, values=vals), using="bm25", limit=limit
                )
            )
        # One Qdrant call runs both searches and merges them with reciprocal rank fusion (RRF).
        res = self.client.query_points(
            CHUNKS,
            prefetch=prefetch,
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        # Read the chunk id from each point's payload and keep the fused score.
        return [(str((p.payload or {})["chunk_id"]), float(p.score)) for p in res.points]

    # Reorders the fused candidates with the cross-encoder and returns the top k (chunk_id, score).
    def rerank(
        self, question: str, cands: list[tuple[str, float]], k: int
    ) -> list[tuple[str, float]]:
        # No reranker (or no candidates): just keep the fused order.
        if self.reranker is None or not cands:
            return cands[:k]
        # Only the first rerank_top candidates are read by the slow cross-encoder.
        head = cands[: self.rerank_top]
        texts = [self.chunk_docs[c] for c, _ in head]
        scores = self.reranker.score(question, texts)
        # Optional blend: z-normalize, then mix in a second reranker and/or the RRF score.
        if self.second is not None or self.alpha:
            scores = zscore(scores)
            if self.second is not None:
                z2 = zscore(self.second.score(question, texts))
                scores = [(1 - self.w2) * a + self.w2 * b for a, b in zip(scores, z2, strict=True)]
            if self.alpha:
                zr = zscore([s for _, s in head])
                scores = [a + self.alpha * b for a, b in zip(scores, zr, strict=True)]
        # Sort by score (ties keep the original order) and keep the best k.
        order = sorted(range(len(head)), key=lambda i: (-scores[i], i))
        return [(head[i][0], scores[i]) for i in order[:k]]

    # Main entry point: question in, RetrievalResult with top-k chunk ids, scores and timings out.
    def retrieve(self, question: str, k: int) -> RetrievalResult:
        # Time search and rerank separately so the API and reports show where time goes.
        t0 = time.perf_counter()
        cands = self.candidates_for(question, self.candidates)
        t1 = time.perf_counter()
        ranked = self.rerank(question, cands, k)
        t2 = time.perf_counter()
        # Package the ranked ids and scores with the timings.
        return RetrievalResult(
            chunk_ids=[c for c, _ in ranked],
            scores=[s for _, s in ranked],
            timings={"search_s": t1 - t0, "rerank_s": t2 - t1, "total_s": t2 - t0},
        )
