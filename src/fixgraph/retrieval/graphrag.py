"""GraphRAG retrievers (spec §10).

S2 (HippoRAG-style): link question entities to seed nodes -> Personalized PageRank over the KG
-> score each chunk by the PPR mass of the nodes it supports -> cross-encoder rerank.

S3 (typed paths): from the seeds, search schema-valid paths (<= 3 hops) that end in a Symptom,
Cause or Fix; chunks that provide the edges of the best paths are the evidence (reranked), and
the paths are serialized as navigation hints for the answer model.

Both fall back to the hybrid (S1) candidates when no seed can be linked, and report it.

S2L is S2 over the graph plus the document link layer (retrieval.graph.add_document_layer).
S4 fuses S1's candidates with S2L's PPR chunk ranking by reciprocal rank (FusionRetriever).
Predicted (GNN) edges only change routing; their provenance never becomes evidence.

Used by: bench/cli.py (builds PPRRetriever / PathRetriever for `bench run`).
Uses: retrieval.linking (seeds), retrieval.graph (GraphIndex), retrieval.ppr,
retrieval.hybrid (candidates, fallback, reranker), retrieval.base.
"""

# Imports: shared result types, the graph index, the hybrid retriever (rerank, fallback),
# the entity linker and PageRank.
import time
from collections import defaultdict

import numpy as np

from fixgraph.retrieval.base import RetrievalResult, Subgraph, SubgraphEdge
from fixgraph.retrieval.graph import DEFAULT_REL_WEIGHTS, GraphIndex
from fixgraph.retrieval.hybrid import HybridRetriever
from fixgraph.retrieval.linking import EntityLinker, Mentions, Seed
from fixgraph.retrieval.ppr import personalized_pagerank


# Shared base for the graph retrievers (S2, S3): seed linking, final rerank and S1 fallback.
class _GraphRetrieverBase:
    name = "S?"

    # Keeps the graph, linker, hybrid retriever and the mentions saved earlier per question.
    def __init__(
        self,
        graph: GraphIndex,
        linker: EntityLinker,
        hybrid: HybridRetriever,
        mentions: dict[str, Mentions],
        candidates: int = 30,
    ) -> None:
        self.graph, self.linker, self.hybrid = graph, linker, hybrid
        self.mentions = mentions  # precomputed per question (LLM stage runs separately)
        self.candidates = candidates
        self.last_seeds: list[Seed] = []

    # Question in, {node row: weight} seeds out; uses the saved mentions and remembers the seeds.
    def seeds(self, question: str) -> dict[int, float]:
        found = self.linker.link(question, self.mentions.get(question, Mentions()))
        self.last_seeds = found
        return {
            self.graph.index[s.node_id]: s.weight for s in found if s.node_id in self.graph.index
        }

    # Takes graph-scored chunks, keeps the top candidates, reranks with the cross-encoder, and
    # returns a RetrievalResult with timings and a flag saying whether fallback was used.
    def _finish(
        self,
        question: str,
        scored: dict[str, float],
        k: int,
        t0: float,
        subgraph: Subgraph,
        fallback: bool,
    ) -> RetrievalResult:
        cands = sorted(scored.items(), key=lambda x: -x[1])[: self.candidates]
        t1 = time.perf_counter()
        ranked = self.hybrid.rerank(question, cands, k)
        t2 = time.perf_counter()
        return RetrievalResult(
            chunk_ids=[c for c, _ in ranked],
            scores=[s for _, s in ranked],
            subgraph=subgraph,
            timings={
                "graph_s": t1 - t0,
                "rerank_s": t2 - t1,
                "total_s": t2 - t0,
                "fallback": float(fallback),
            },
        )

    # No usable seeds or paths: use the hybrid (S1) fused candidates instead, flagged as fallback.
    def _fallback(self, question: str, k: int, t0: float) -> RetrievalResult:
        cands = self.hybrid.candidates_for(question, self.hybrid.candidates)
        return self._finish(question, dict(cands), k, t0, Subgraph(), fallback=True)


# S2: ranks chunks by Personalized PageRank mass spreading out from the question's seed nodes.
class PPRRetriever(_GraphRetrieverBase):
    name = "S2"

    # Same setup as the base class plus the PageRank damping factor.
    def __init__(self, *args: object, damping: float = 0.5, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.damping = damping

    # Question in, (chunk -> score, subgraph) out; returns None when no seed node is found.
    def chunk_scores(self, question: str) -> tuple[dict[str, float], Subgraph] | None:
        """PPR mass per chunk (sum over the nodes each chunk supports), or None without seeds."""
        seeds = self.seeds(question)
        if not seeds:
            return None
        # Run PageRank, then give each chunk the summed score of every node it supports.
        scores = personalized_pagerank(self.graph.adj, seeds, damping=self.damping)
        chunk_scores: dict[str, float] = defaultdict(float)
        for i in np.nonzero(scores > 1e-9)[0]:
            for cid in self.graph.node_chunks.get(int(i), ()):
                chunk_scores[cid] += float(scores[i])
        # Record the 15 highest-scoring nodes and the seeds for the subgraph shown in results.
        top_nodes = [int(i) for i in np.argsort(-scores)[:15]]
        subgraph = Subgraph(
            nodes=[self.graph.node_ids[i] for i in top_nodes],
            seeds=[self.graph.node_ids[i] for i in seeds],
        )
        return (dict(chunk_scores), subgraph) if chunk_scores else None

    # Graph-score the chunks, fall back to S1 if there are no seeds, then rerank to the top k.
    def retrieve(self, question: str, k: int) -> RetrievalResult:
        t0 = time.perf_counter()
        found = self.chunk_scores(question)
        if found is None:
            return self._fallback(question, k, t0)
        return self._finish(question, found[0], k, t0, found[1], fallback=False)


# S4: merges the hybrid ranking and the PageRank chunk ranking by reciprocal rank, then reranks.
class FusionRetriever:
    """S4 (DECISIONS.md D34): reciprocal-rank fusion of the hybrid candidates (S1) and the PPR
    chunk ranking of a graph retriever, then the same cross-encoder. Fixed a priori: k = 60,
    equal weights, top `candidates` fused chunks reranked. Without seeds it equals S1."""

    name = "S4"

    # Holds the hybrid and PageRank retrievers plus the fusion constant and pool size.
    def __init__(
        self, hybrid: HybridRetriever, ppr: PPRRetriever, rrf_k: int = 60, candidates: int = 30
    ) -> None:
        self.hybrid, self.ppr, self.rrf_k, self.candidates = hybrid, ppr, rrf_k, candidates

    # Exposes the PageRank retriever's seeds so the harness can log them.
    @property
    def last_seeds(self) -> list[Seed]:
        return self.ppr.last_seeds

    # Question in, top-k RetrievalResult out.
    def retrieve(self, question: str, k: int) -> RetrievalResult:
        t0 = time.perf_counter()
        # Get both rankings: hybrid fused candidates and (if seeds exist) PageRank chunk scores.
        dense = self.hybrid.candidates_for(question, self.hybrid.candidates)
        found = self.ppr.chunk_scores(question)
        fused: dict[str, float] = defaultdict(float)
        # Add 1 / (rrf_k + rank) from the hybrid list for each chunk.
        for rank, (cid, _) in enumerate(dense):
            fused[cid] += 1.0 / (self.rrf_k + rank + 1)
        subgraph = None
        # Add the same rank score from the graph list, so a chunk found by either can rank high.
        if found is not None:
            graph_ranked = sorted(found[0].items(), key=lambda x: -x[1])[: self.hybrid.candidates]
            for rank, (cid, _) in enumerate(graph_ranked):
                fused[cid] += 1.0 / (self.rrf_k + rank + 1)
            subgraph = found[1]
        # Keep the best fused candidates and rerank them with the shared cross-encoder.
        cands = sorted(fused.items(), key=lambda x: -x[1])[: self.candidates]
        t1 = time.perf_counter()
        ranked = self.hybrid.rerank(question, cands, k)
        t2 = time.perf_counter()
        # Return the reranked chunks; fallback is marked when the graph side had no seeds.
        return RetrievalResult(
            chunk_ids=[c for c, _ in ranked],
            scores=[s for _, s in ranked],
            subgraph=subgraph,
            timings={
                "fuse_s": t1 - t0,
                "rerank_s": t2 - t1,
                "total_s": t2 - t0,
                "fallback": float(found is None),
            },
        )


# Path search -------------------------------------------------------------------------------

# Node types a path may end at, and relation types path search must not walk through.
ANSWER_LABELS = frozenset({"Symptom", "Cause", "Fix"})
SKIP_RELS = frozenset({"IN_FAMILY"})  # family hubs connect everything; not troubleshooting hops


# S3: finds short typed paths from the seed nodes to Symptom/Cause/Fix nodes and uses the
# chunks behind those edges as evidence.
class PathRetriever(_GraphRetrieverBase):
    name = "S3"

    # Same setup as the base class plus beam-search limits: hops, beam width, paths kept.
    def __init__(
        self,
        *args: object,
        max_hops: int = 3,
        beam: int = 300,
        top_paths: int = 20,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.max_hops, self.beam, self.top_paths = max_hops, beam, top_paths

    # Returns neighbours of a node across edges in both directions, skipping IN_FAMILY hops.
    def _neighbors(self, node: int) -> list[tuple[int, int]]:
        """(edge index, other node) for every incident edge, both directions."""
        g = self.graph
        out = [(e, g.edges[e][2]) for e in g.out_edges.get(node, [])]
        inn = [(e, g.edges[e][0]) for e in g.in_edges.get(node, [])]
        return [(e, o) for e, o in out + inn if g.edges[e][1] not in SKIP_RELS]

    # Beam search from the seeds (up to max_hops); returns the best paths ending on answer types.
    def search_paths(self, seeds: dict[int, float]) -> list[tuple[float, list[int], list[int]]]:
        """Beam search: (score, node path, edge path) for paths ending at answer-type nodes."""
        # Start one path at each seed, scored by the seed's weight.
        g = self.graph
        frontier: list[tuple[float, list[int], list[int]]] = [
            (w, [s], []) for s, w in seeds.items()
        ]
        done: list[tuple[float, list[int], list[int]]] = []
        # Each hop: extend every path by one edge, never revisiting a node on the same path.
        for _ in range(self.max_hops):
            nxt: list[tuple[float, list[int], list[int]]] = []
            for score, nodes, edges in frontier:
                for e, other in self._neighbors(nodes[-1]):
                    if other in nodes:
                        continue
                    # Each hop scales the score by confidence, relation weight and a 0.8 penalty.
                    _s, rel, _d, conf, _chunks, origin = g.edges[e]
                    w = DEFAULT_REL_WEIGHTS.get("PREDICTED" if origin == "predicted" else rel, 0.5)
                    new_score = score * max(conf, 0.1) * w * 0.8
                    if other in seeds:  # paths that connect several question entities win
                        new_score *= 1.0 + seeds[other]
                    # Keep the path; if it ends on an answer-type node, also save it as finished.
                    path = (new_score, [*nodes, other], [*edges, e])
                    nxt.append(path)
                    if g.labels[other] in ANSWER_LABELS:
                        done.append(path)
            # Keep only the best `beam` paths before the next hop.
            frontier = sorted(nxt, key=lambda p: -p[0])[: self.beam]
        return sorted(done, key=lambda p: -p[0])[: self.top_paths]

    # Turns a path into readable text like "A -[CAUSED_BY]-> B", showing each edge's direction.
    def serialize(self, nodes: list[int], edges: list[int]) -> str:
        g = self.graph
        parts = [g.texts[nodes[0]]]
        for n, e in zip(nodes[1:], edges, strict=True):
            src, rel, _dst, *_ = g.edges[e]
            arrow = f"-[{rel}]->" if src == nodes[nodes.index(n) - 1] else f"<-[{rel}]-"
            parts.append(f"{arrow} {g.texts[n]}")
        return " ".join(parts)

    # Question in, top-k RetrievalResult out, with the found paths attached as a subgraph.
    def retrieve(self, question: str, k: int) -> RetrievalResult:
        # Link seeds and search paths; fall back to S1 when nothing is found.
        t0 = time.perf_counter()
        seeds = self.seeds(question)
        paths = self.search_paths(seeds) if seeds else []
        if not paths:
            return self._fallback(question, k, t0)
        # Score each chunk by the best path its edges support; collect edges for the subgraph.
        g = self.graph
        chunk_scores: dict[str, float] = defaultdict(float)
        sub_edges: list[SubgraphEdge] = []
        for score, _nodes, edges in paths:
            for e in edges:
                s, rel, d, _conf, chunks, origin = g.edges[e]
                sub_edges.append(
                    SubgraphEdge(src=g.node_ids[s], rel=rel, dst=g.node_ids[d], origin=origin)
                )
                if origin != "extracted":
                    continue  # predicted edges route, never provide evidence
                for cid in chunks:
                    chunk_scores[cid] = max(chunk_scores[cid], score)
        # Build the subgraph: path nodes, their edges, the top 8 paths as text, and the seeds.
        subgraph = Subgraph(
            nodes=sorted({n for _, nodes, _ in paths for n in map(g.node_ids.__getitem__, nodes)}),
            edges=sub_edges,
            paths=[[self.serialize(nodes, edges)] for _, nodes, edges in paths[:8]],
            seeds=[g.node_ids[i] for i in seeds],
        )
        # If only predicted edges were found there is no evidence, so fall back; otherwise rerank.
        if not chunk_scores:
            return self._fallback(question, k, t0)
        return self._finish(question, chunk_scores, k, t0, subgraph, fallback=False)
