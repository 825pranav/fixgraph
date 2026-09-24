"""GraphRAG retrievers (spec §10).

S2 (HippoRAG-style): link question entities to seed nodes -> Personalized PageRank over the KG
-> score each chunk by the PPR mass of the nodes it supports -> cross-encoder rerank.

S3 (typed paths): from the seeds, search schema-valid paths (<= 3 hops) that end in a Symptom,
Cause or Fix; chunks that provide the edges of the best paths are the evidence (reranked), and
the paths are serialized as navigation hints for the answer model.

Both fall back to the hybrid (S1) candidates when no seed can be linked, and report it.
Predicted (GNN) edges only change routing; their provenance never becomes evidence.

Used by: bench/cli.py (builds PPRRetriever / PathRetriever for `bench run`).
Uses: retrieval.linking (seeds), retrieval.graph (GraphIndex), retrieval.ppr,
retrieval.hybrid (candidates, fallback, reranker), retrieval.base.
"""

import time
from collections import defaultdict

import numpy as np

from fixgraph.retrieval.base import RetrievalResult, Subgraph, SubgraphEdge
from fixgraph.retrieval.graph import DEFAULT_REL_WEIGHTS, GraphIndex
from fixgraph.retrieval.hybrid import HybridRetriever
from fixgraph.retrieval.linking import EntityLinker, Mentions, Seed
from fixgraph.retrieval.ppr import personalized_pagerank


class _GraphRetrieverBase:
    name = "S?"

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

    def seeds(self, question: str) -> dict[int, float]:
        found = self.linker.link(question, self.mentions.get(question, Mentions()))
        self.last_seeds = found
        return {
            self.graph.index[s.node_id]: s.weight for s in found if s.node_id in self.graph.index
        }

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

    def _fallback(self, question: str, k: int, t0: float) -> RetrievalResult:
        cands = self.hybrid.candidates_for(question, self.hybrid.candidates)
        return self._finish(question, dict(cands), k, t0, Subgraph(), fallback=True)


class PPRRetriever(_GraphRetrieverBase):
    name = "S2"

    def __init__(self, *args: object, damping: float = 0.5, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.damping = damping

    def retrieve(self, question: str, k: int) -> RetrievalResult:
        t0 = time.perf_counter()
        seeds = self.seeds(question)
        if not seeds:
            return self._fallback(question, k, t0)
        scores = personalized_pagerank(self.graph.adj, seeds, damping=self.damping)
        chunk_scores: dict[str, float] = defaultdict(float)
        for i in np.nonzero(scores > 1e-9)[0]:
            for cid in self.graph.node_chunks.get(int(i), ()):
                chunk_scores[cid] += float(scores[i])
        top_nodes = [int(i) for i in np.argsort(-scores)[:15]]
        subgraph = Subgraph(
            nodes=[self.graph.node_ids[i] for i in top_nodes],
            seeds=[self.graph.node_ids[i] for i in seeds],
        )
        if not chunk_scores:
            return self._fallback(question, k, t0)
        return self._finish(question, chunk_scores, k, t0, subgraph, fallback=False)


# Path search -------------------------------------------------------------------------------

ANSWER_LABELS = frozenset({"Symptom", "Cause", "Fix"})
SKIP_RELS = frozenset({"IN_FAMILY"})  # family hubs connect everything; not troubleshooting hops


class PathRetriever(_GraphRetrieverBase):
    name = "S3"

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

    def _neighbors(self, node: int) -> list[tuple[int, int]]:
        """(edge index, other node) for every incident edge, both directions."""
        g = self.graph
        out = [(e, g.edges[e][2]) for e in g.out_edges.get(node, [])]
        inn = [(e, g.edges[e][0]) for e in g.in_edges.get(node, [])]
        return [(e, o) for e, o in out + inn if g.edges[e][1] not in SKIP_RELS]

    def search_paths(self, seeds: dict[int, float]) -> list[tuple[float, list[int], list[int]]]:
        """Beam search: (score, node path, edge path) for paths ending at answer-type nodes."""
        g = self.graph
        frontier: list[tuple[float, list[int], list[int]]] = [
            (w, [s], []) for s, w in seeds.items()
        ]
        done: list[tuple[float, list[int], list[int]]] = []
        for _ in range(self.max_hops):
            nxt: list[tuple[float, list[int], list[int]]] = []
            for score, nodes, edges in frontier:
                for e, other in self._neighbors(nodes[-1]):
                    if other in nodes:
                        continue
                    _s, rel, _d, conf, _chunks, origin = g.edges[e]
                    w = DEFAULT_REL_WEIGHTS.get("PREDICTED" if origin == "predicted" else rel, 0.5)
                    new_score = score * max(conf, 0.1) * w * 0.8
                    if other in seeds:  # paths that connect several question entities win
                        new_score *= 1.0 + seeds[other]
                    path = (new_score, [*nodes, other], [*edges, e])
                    nxt.append(path)
                    if g.labels[other] in ANSWER_LABELS:
                        done.append(path)
            frontier = sorted(nxt, key=lambda p: -p[0])[: self.beam]
        return sorted(done, key=lambda p: -p[0])[: self.top_paths]

    def serialize(self, nodes: list[int], edges: list[int]) -> str:
        g = self.graph
        parts = [g.texts[nodes[0]]]
        for n, e in zip(nodes[1:], edges, strict=True):
            src, rel, _dst, *_ = g.edges[e]
            arrow = f"-[{rel}]->" if src == nodes[nodes.index(n) - 1] else f"<-[{rel}]-"
            parts.append(f"{arrow} {g.texts[n]}")
        return " ".join(parts)

    def retrieve(self, question: str, k: int) -> RetrievalResult:
        t0 = time.perf_counter()
        seeds = self.seeds(question)
        paths = self.search_paths(seeds) if seeds else []
        if not paths:
            return self._fallback(question, k, t0)
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
        subgraph = Subgraph(
            nodes=sorted({n for _, nodes, _ in paths for n in map(g.node_ids.__getitem__, nodes)}),
            edges=sub_edges,
            paths=[[self.serialize(nodes, edges)] for _, nodes, edges in paths[:8]],
            seeds=[g.node_ids[i] for i in seeds],
        )
        if not chunk_scores:
            return self._fallback(question, k, t0)
        return self._finish(question, chunk_scores, k, t0, subgraph, fallback=False)
