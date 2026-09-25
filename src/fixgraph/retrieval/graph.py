"""In-memory graph index over the parquet KG for retrieval (spec §10.2, §10.3).

- Undirected weighted adjacency (scipy.sparse CSR) over non-Article nodes; edge weight =
  relation-type weight x extraction confidence. Parallel edges between the same pair add up.
- node -> chunks (mentions + edge provenance) and chunk -> nodes maps for PPR scoring.
- Only `origin == "extracted"` edges are evidence; predicted edges (GNN, S4) can be added as
  routing-only edges and are tracked separately so they never become citations.
- `add_document_layer` (S2L, D34) adds article nodes joined to the entities they mention and to
  each other by Apple's hyperlinks; like predicted edges these only route.

Used by: bench/cli.py (`build_graph_index` for S2/S3), retrieval.graphrag.
Uses: kg.store.KG.
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

from fixgraph.kg.store import KG

# Relation weights for PPR, tuned on the dev split (DECISIONS.md). Structural relations that
# connect everything (IN_FAMILY) are down-weighted so mass is not sucked into hub families.
DEFAULT_REL_WEIGHTS: dict[str, float] = {
    "RESOLVED_BY": 1.0,
    "CAUSED_BY": 1.0,
    "ADDRESSES": 1.0,
    "SIGNALS": 1.0,
    "EXHIBITS": 0.8,
    "DEPENDS_ON": 0.8,
    "INVOLVES": 0.6,
    "REQUIRES": 0.6,
    "APPLIES_TO": 0.6,
    "RUNS": 0.4,
    "HAS_COMPONENT": 0.4,
    "IN_FAMILY": 0.2,
    "PREDICTED": 0.5,
}


@dataclass
class GraphIndex:
    node_ids: list[str]
    labels: list[str]
    texts: list[str]
    adj: csr_matrix  # symmetric weighted adjacency
    index: dict[str, int]
    node_chunks: dict[int, set[str]]
    chunk_nodes: dict[str, set[int]]
    edges: list[
        tuple[int, str, int, float, list[str], str]
    ]  # (src, rel, dst, conf, chunks, origin)
    out_edges: dict[int, list[int]] = field(default_factory=lambda: dict[int, list[int]]())
    in_edges: dict[int, list[int]] = field(default_factory=lambda: dict[int, list[int]]())

    @property
    def n(self) -> int:
        return len(self.node_ids)


def build_graph_index(
    kg: KG,
    rel_weights: dict[str, float] | None = None,
    predicted: pl.DataFrame | None = None,
    use_type_weights: bool = True,
) -> GraphIndex:
    """`use_type_weights=False` is the "PPR without edge-type weights" ablation."""
    weights = rel_weights or DEFAULT_REL_WEIGHTS
    nodes = kg.nodes.filter(pl.col("label") != "Article")
    node_ids = nodes["node_id"].to_list()
    index = {n: i for i, n in enumerate(node_ids)}

    edge_frames = [kg.extracted_edges()]
    if predicted is not None and len(predicted):
        edge_frames.append(predicted.select(kg.edges.columns))
    edges: list[tuple[int, str, int, float, list[str], str]] = []
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    out_edges: dict[int, list[int]] = defaultdict(list)
    in_edges: dict[int, list[int]] = defaultdict(list)
    for frame in edge_frames:
        for r in frame.iter_rows(named=True):
            s, d = index.get(r["src"]), index.get(r["dst"])
            if s is None or d is None or s == d:
                continue
            origin = r["origin"]
            rel_w = weights.get("PREDICTED" if origin == "predicted" else r["rel"], 0.5)
            w = (rel_w if use_type_weights else 1.0) * max(float(r["extraction_confidence"]), 0.1)
            k = len(edges)
            edges.append(
                (
                    s,
                    r["rel"],
                    d,
                    float(r["extraction_confidence"]),
                    list(r["source_chunk_ids"]),
                    origin,
                )
            )
            out_edges[s].append(k)
            in_edges[d].append(k)
            rows += [s, d]
            cols += [d, s]
            vals += [w, w]
    n = len(node_ids)
    adj = csr_matrix((np.asarray(vals), (np.asarray(rows), np.asarray(cols))), shape=(n, n))
    adj.sum_duplicates()

    node_chunks: dict[int, set[str]] = defaultdict(set)
    chunk_nodes: dict[str, set[int]] = defaultdict(set)
    for nid, cid in kg.mentions.select("node_id", "chunk_id").iter_rows():
        i = index.get(nid)
        if i is not None:
            node_chunks[i].add(cid)
            chunk_nodes[cid].add(i)
    for s, _rel, d, _c, chunks, origin in edges:
        if origin != "extracted":
            continue
        for cid in chunks:
            for i in (s, d):
                node_chunks[i].add(cid)
                chunk_nodes[cid].add(i)
    return GraphIndex(
        node_ids=node_ids,
        labels=nodes["label"].to_list(),
        texts=nodes["canonical_text"].to_list(),
        adj=adj,
        index=index,
        node_chunks=dict(node_chunks),
        chunk_nodes=dict(chunk_nodes),
        edges=edges,
        out_edges=dict(out_edges),
        in_edges=dict(in_edges),
    )


# Document link layer (DECISIONS.md D34) ------------------------------------------------------

# Fixed a priori, not tuned: an article is a structural hub like a product family (IN_FAMILY
# 0.2); an Apple-authored link between articles is weighted like a dependency (DEPENDS_ON 0.8).
MENTION_WEIGHT = 0.2
LINK_WEIGHT = 0.8


def add_document_layer(
    graph: GraphIndex,
    links: list[tuple[str, str, str]],
    article_of: Callable[[str], str],
    mention_weight: float = MENTION_WEIGHT,
    link_weight: float = LINK_WEIGHT,
) -> GraphIndex:
    """Copy of `graph` with one node per article: every entity node is joined to the articles
    whose chunks mention it, and articles are joined by hyperlinks (src_article, dst_article,
    src_chunk_id). An article node maps to all its chunks, so PPR mass that reaches it (for
    example across a link) scores the linked article's chunks. Document edges have origin
    "document": they route, and are never cited as extracted evidence."""
    articles = sorted({article_of(c) for c in graph.chunk_nodes} | {d for _, d, _ in links})
    base = graph.n
    index = dict(graph.index)
    node_ids = list(graph.node_ids)
    for i, a in enumerate(articles):
        index[f"doc:{a}"] = base + i
        node_ids.append(f"doc:{a}")
    art = {a: base + i for i, a in enumerate(articles)}

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    edges = list(graph.edges)
    node_chunks = {k: set(v) for k, v in graph.node_chunks.items()}
    chunk_nodes = {k: set(v) for k, v in graph.chunk_nodes.items()}
    for cid, nodes in graph.chunk_nodes.items():
        d = art[article_of(cid)]
        node_chunks.setdefault(d, set()).add(cid)
        chunk_nodes[cid].add(d)
        for n in nodes:
            rows += [n, d]
            cols += [d, n]
            vals += [mention_weight, mention_weight]
    for src, dst, cid in links:
        if src not in art or src == dst:
            continue
        s, d = art[src], art[dst]
        edges.append((s, "LINKS_TO", d, 1.0, [cid] if cid else [], "document"))
        rows += [s, d]
        cols += [d, s]
        vals += [link_weight, link_weight]
    n = len(node_ids)
    old = graph.adj.tocoo()
    adj = csr_matrix(
        (
            np.concatenate([old.data, np.asarray(vals, dtype=np.float64)]),
            (
                np.concatenate([old.row, np.asarray(rows, dtype=np.int64)]),
                np.concatenate([old.col, np.asarray(cols, dtype=np.int64)]),
            ),
        ),
        shape=(n, n),
    )
    adj.sum_duplicates()
    return GraphIndex(
        node_ids=node_ids,
        labels=[*graph.labels, *(["Article"] * len(articles))],
        texts=[*graph.texts, *articles],
        adj=adj,
        index=index,
        node_chunks=node_chunks,
        chunk_nodes=chunk_nodes,
        edges=edges,
        out_edges=graph.out_edges,
        in_edges=graph.in_edges,
    )
