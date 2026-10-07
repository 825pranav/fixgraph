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

# Imports: the graph index gives chunk-to-entity links; article_of maps chunks to articles.
from collections import defaultdict
from statistics import median
from typing import Any

from fixgraph.bench.schema import Question, article_of
from fixgraph.retrieval.graph import GraphIndex

# Expansion routes we test, from cheap (same article) to wide (two knowledge-graph hops).
ROUTES = ("same_article", "link1", "link2", "kg0", "kg1", "kg2")


# Precomputes neighbour lookups so each "can this route reach the miss?" check is fast.
class Expander:
    """Chunk-level neighbourhoods by route, precomputed from the graph index and the links."""

    # Build article/hyperlink/entity lookup tables once from the graph and the link list.
    def __init__(self, graph: GraphIndex, links: list[tuple[str, str]], chunk_ids: list[str]):
        # Group chunk ids by their article.
        self.chunks_of: dict[str, set[str]] = defaultdict(set)
        for c in chunk_ids:
            self.chunks_of[article_of(c)].add(c)
        # Undirected article neighbours from the Apple hyperlinks.
        self.art_nbrs: dict[str, set[str]] = defaultdict(set)
        for src, dst in links:
            if src != dst:
                self.art_nbrs[src].add(dst)
                self.art_nbrs[dst].add(src)
        # Chunk-to-entity and entity-to-chunk maps come straight from the graph index.
        self.nodes_of = graph.chunk_nodes
        self.chunks_of_node = graph.node_chunks
        self.node_nbrs: dict[int, set[int]] = defaultdict(set)
        # Entity neighbours using only extracted edges; family hub edges would connect everything.
        for s, rel, d, _conf, _chunks, origin in graph.edges:
            if origin == "extracted" and rel != "IN_FAMILY":
                self.node_nbrs[s].add(d)
                self.node_nbrs[d].add(s)

    # Breadth-first walk over article links: all articles within `hops` links of the start set.
    def _articles(self, arts: set[str], hops: int) -> set[str]:
        seen, frontier = set(arts), set(arts)
        for _ in range(hops):
            frontier = {n for a in frontier for n in self.art_nbrs.get(a, ())} - seen
            seen |= frontier
        return seen

    # Same breadth-first walk, but over entity nodes in the knowledge graph.
    def _nodes(self, nodes: set[int], hops: int) -> set[int]:
        seen, frontier = set(nodes), set(nodes)
        for _ in range(hops):
            frontier = {n for x in frontier for n in self.node_nbrs.get(x, ())} - seen
            seen |= frontier
        return seen

    # Given S1's hit chunks and a route name, return every new chunk that route could add.
    def pool(self, hits: list[str], route: str) -> set[str]:
        """Candidate chunks the route opens from the hits (hits themselves excluded)."""
        arts = {article_of(c) for c in hits}
        # Route: other chunks from the same articles as the hits.
        if route == "same_article":
            out = {c for a in arts for c in self.chunks_of[a]}
        # Route: chunks from articles one or two hyperlinks away.
        elif route in ("link1", "link2"):
            reach = self._articles(arts, 1 if route == "link1" else 2)
            out = {c for a in reach for c in self.chunks_of[a]}
        # Route: chunks that mention entities within 0, 1 or 2 graph hops of the hits' entities.
        else:
            hops = {"kg0": 0, "kg1": 1, "kg2": 2}[route]
            seed = {n for c in hits for n in self.nodes_of.get(c, ())}
            out = {c for n in self._nodes(seed, hops) for c in self.chunks_of_node.get(n, ())}
        return out - set(hits)


# Main analysis: for each gold chunk S1 missed, which routes could have reached it and how cheaply.
def miss_analysis(
    questions: list[Question],
    ranked: dict[str, list[str]],
    expander: Expander,
    k: int = 8,
    hit_depths: tuple[int, ...] = (3, 8),
) -> dict[str, Any]:
    """`ranked[qid]` = S1's chunk ids, best first. For each hit depth d (expand from S1's top
    d), the share of missed gold chunks each route reaches and the median pool size."""
    # Step 1: collect every (question, gold chunk) pair that is missing from S1's top k.
    misses = []
    for q in questions:
        got = ranked.get(q.qid, [])[:k]
        for m in q.gold_chunk_ids:
            if m not in got:
                misses.append((q, m))
    n_gold = sum(len(q.gold_chunk_ids) for q in questions)
    # Step 2: for each expansion depth, measure every route against all misses.
    by_depth: dict[str, Any] = {}
    for d in hit_depths:
        rows: dict[str, dict[str, Any]] = {}
        per_miss: list[dict[str, Any]] = []
        # Per route: how many misses it reaches and the median number of chunks it would open.
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
        # Per miss: a row of which routes reach it, for the detailed output.
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
        # Combined route: same article or one hyperlink hop.
        any_link = sum(x["same_article"] or x["link1"] for x in per_miss)
        rows["same_article_or_link1"] = {
            "reached": any_link,
            "share": round(any_link / len(misses), 3) if misses else None,
        }
        by_depth[f"expand_from_top{d}"] = {"routes": rows, "misses": per_miss}
    # Step 3: count questions and misses per question type.
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"questions": 0, "missed_chunks": 0})
    for q in questions:
        by_type[q.qtype]["questions"] += 1
    for q, _ in misses:
        by_type[q.qtype]["missed_chunks"] += 1
    # Return one summary dict that the CLI writes out as JSON.
    return {
        "k": k,
        "n_questions": len(questions),
        "n_gold_chunks": n_gold,
        "n_missed_chunks": len(misses),
        "questions_with_a_miss": len({q.qid for q, _ in misses}),
        "by_type": dict(by_type),
        **by_depth,
    }
