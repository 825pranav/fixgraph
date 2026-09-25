"""Miss analysis for the hybrid retriever (DECISIONS.md D37): the ceiling for graph assistance.

For every question and every gold chunk S1 does not return in its top k, check whether the
missing chunk can be reached from S1's retrieved chunks ("hits") by each expansion route, and
how many candidate chunks that route opens (a route that reaches everything by opening the
whole corpus is useless under a fixed 8-chunk budget):

- same_article   the missing chunk is in the same article as a hit
- link1 / link2  1 or 2 Apple hyperlink hops between articles (either direction)
- kg0            the missing chunk mentions an entity a hit mentions (shared node)
- kg1 / kg2      1 or 2 verified extracted-edge hops between the chunks' entities
                 (ontology IN_FAMILY edges excluded: family hubs connect everything)

Used by: `fixgraph bench miss-analysis` (bench/cli.py).
Uses: retrieval.graph (GraphIndex), bench.schema.article_of.
"""

from collections import defaultdict
from statistics import median
from typing import Any

from fixgraph.bench.schema import Question, article_of
from fixgraph.retrieval.graph import GraphIndex

ROUTES = ("same_article", "link1", "link2", "kg0", "kg1", "kg2")


class Expander:
    """Chunk-level neighbourhoods by route, precomputed from the graph index and the links."""

    def __init__(self, graph: GraphIndex, links: list[tuple[str, str]], chunk_ids: list[str]):
        self.chunks_of: dict[str, set[str]] = defaultdict(set)
        for c in chunk_ids:
            self.chunks_of[article_of(c)].add(c)
        self.art_nbrs: dict[str, set[str]] = defaultdict(set)
        for src, dst in links:
            if src != dst:
                self.art_nbrs[src].add(dst)
                self.art_nbrs[dst].add(src)
        self.nodes_of = graph.chunk_nodes
        self.chunks_of_node = graph.node_chunks
        self.node_nbrs: dict[int, set[int]] = defaultdict(set)
        for s, rel, d, _conf, _chunks, origin in graph.edges:
            if origin == "extracted" and rel != "IN_FAMILY":
                self.node_nbrs[s].add(d)
                self.node_nbrs[d].add(s)

    def _articles(self, arts: set[str], hops: int) -> set[str]:
        seen, frontier = set(arts), set(arts)
        for _ in range(hops):
            frontier = {n for a in frontier for n in self.art_nbrs.get(a, ())} - seen
            seen |= frontier
        return seen

    def _nodes(self, nodes: set[int], hops: int) -> set[int]:
        seen, frontier = set(nodes), set(nodes)
        for _ in range(hops):
            frontier = {n for x in frontier for n in self.node_nbrs.get(x, ())} - seen
            seen |= frontier
        return seen

    def pool(self, hits: list[str], route: str) -> set[str]:
        """Candidate chunks the route opens from the hits (hits themselves excluded)."""
        arts = {article_of(c) for c in hits}
        if route == "same_article":
            out = {c for a in arts for c in self.chunks_of[a]}
        elif route in ("link1", "link2"):
            reach = self._articles(arts, 1 if route == "link1" else 2)
            out = {c for a in reach for c in self.chunks_of[a]}
        else:
            hops = {"kg0": 0, "kg1": 1, "kg2": 2}[route]
            seed = {n for c in hits for n in self.nodes_of.get(c, ())}
            out = {c for n in self._nodes(seed, hops) for c in self.chunks_of_node.get(n, ())}
        return out - set(hits)


def miss_analysis(
    questions: list[Question],
    ranked: dict[str, list[str]],
    expander: Expander,
    k: int = 8,
    hit_depths: tuple[int, ...] = (3, 8),
) -> dict[str, Any]:
    """`ranked[qid]` = S1's chunk ids, best first. For each hit depth d (expand from S1's top
    d), the share of missed gold chunks each route reaches and the median pool size."""
    misses = []
    for q in questions:
        got = ranked.get(q.qid, [])[:k]
        for m in q.gold_chunk_ids:
            if m not in got:
                misses.append((q, m))
    n_gold = sum(len(q.gold_chunk_ids) for q in questions)
    by_depth: dict[str, Any] = {}
    for d in hit_depths:
        rows: dict[str, dict[str, Any]] = {}
        per_miss: list[dict[str, Any]] = []
        for route in ROUTES:
            reached = 0
            pools = []
            for q, m in misses:
                pool = expander.pool(ranked.get(q.qid, [])[:d], route)
                pools.append(len(pool))
                reached += m in pool
            rows[route] = {
                "reached": reached,
                "share": round(reached / len(misses), 3) if misses else None,
                "median_pool_chunks": median(pools) if pools else 0,
            }
        for q, m in misses:
            hits = ranked.get(q.qid, [])[:d]
            per_miss.append(
                {
                    "qid": q.qid,
                    "qtype": q.qtype,
                    "missing": m,
                    **{r: m in expander.pool(hits, r) for r in ROUTES},
                }
            )
        any_link = sum(x["same_article"] or x["link1"] for x in per_miss)
        rows["same_article_or_link1"] = {
            "reached": any_link,
            "share": round(any_link / len(misses), 3) if misses else None,
        }
        by_depth[f"expand_from_top{d}"] = {"routes": rows, "misses": per_miss}
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"questions": 0, "missed_chunks": 0})
    for q in questions:
        by_type[q.qtype]["questions"] += 1
    for q, _ in misses:
        by_type[q.qtype]["missed_chunks"] += 1
    return {
        "k": k,
        "n_questions": len(questions),
        "n_gold_chunks": n_gold,
        "n_missed_chunks": len(misses),
        "questions_with_a_miss": len({q.qid for q, _ in misses}),
        "by_type": dict(by_type),
        **by_depth,
    }
