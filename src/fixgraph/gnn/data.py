"""Parquet KG -> PyG HeteroData (spec §9.1).

Node types are KG labels (Article excluded); node features are text embeddings of the canonical
text. Edge types are (src_label, rel.lower(), dst_label) from `origin == "extracted"` edges only.
The graph is kept directed here; `message_graph` filters target edges and applies ToUndirected,
so held-out edges can be removed from message passing in both directions.

Used by: gnn/cli.py, gnn.splits (message_graph), gnn.models (TARGET edge type).
Uses: kg.store.KG, embeddings.Embedder.
"""

# Imports: polars to read the KG tables, torch / PyG to build the heterogeneous graph.
from collections import defaultdict
from dataclasses import dataclass

import polars as pl
import torch
import torch_geometric.transforms as T
from torch import Tensor
from torch_geometric.data import HeteroData

from fixgraph.embeddings import Embedder
from fixgraph.kg.store import KG

# The link we predict: Symptom -> resolved_by -> Fix, plus its reverse for undirected messages.
TARGET = ("Symptom", "resolved_by", "Fix")
REV_TARGET = ("Fix", "rev_resolved_by", "Symptom")


# The PyG graph plus lookups to map between KG node ids and PyG (type, index) positions.
@dataclass
class GraphData:
    data: HeteroData  # directed; target edges in data[TARGET].edge_index
    node_ids: dict[str, list[str]]  # node type -> node ids in index order
    texts: dict[str, list[str]]
    node_index: dict[str, tuple[str, int]]  # node id -> (type, local index)
    target_articles: list[frozenset[str]]  # per target edge (column), its provenance articles

    # Shortcut to the known Symptom -> Fix edges, as a [2, N] index tensor.
    @property
    def target_edge_index(self) -> Tensor:
        return self.data[TARGET].edge_index

    # Number of nodes of one type (0 if the type is missing).
    def num_nodes(self, node_type: str) -> int:
        return len(self.node_ids.get(node_type, []))


# Convert the KG into PyG HeteroData: embedded node features and typed edges.
def build_graph_data(kg: KG, embedder: Embedder) -> GraphData:
    # Give every non-Article node a per-type index and remember its text.
    nodes = kg.nodes.filter(pl.col("label") != "Article")
    node_ids: dict[str, list[str]] = defaultdict(list)
    texts: dict[str, list[str]] = defaultdict(list)
    node_index: dict[str, tuple[str, int]] = {}
    for nid, label, text in nodes.select("node_id", "label", "canonical_text").iter_rows():
        node_index[nid] = (label, len(node_ids[label]))
        node_ids[label].append(nid)
        texts[label].append(text)

    # Node features = sentence embeddings of each node's canonical text.
    data = HeteroData()
    for label, ts in texts.items():
        data[label].x = torch.from_numpy(embedder.encode(ts)).float()
    for label in ("Symptom", "Fix"):  # the target types always exist, even if empty
        if label not in texts:
            data[label].x = torch.zeros((0, embedder.dim))

    # Group extracted edges by (src type, rel, dst type); note each target edge's articles.
    pairs: dict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
    target_articles: list[frozenset[str]] = []
    for r in kg.extracted_edges().iter_rows(named=True):
        s, d = node_index.get(r["src"]), node_index.get(r["dst"])
        if s is None or d is None:
            continue
        et = (s[0], str(r["rel"]).lower(), d[0])
        pairs[et].append((s[1], d[1]))
        if et == TARGET:
            target_articles.append(frozenset(c.split(":")[0] for c in r["source_chunk_ids"]))
    # Turn each edge list into a [2, N] tensor; the target type always exists, even if empty.
    pairs.setdefault(TARGET, [])
    for et, ps in pairs.items():
        ei = (
            torch.tensor(ps, dtype=torch.long).t().contiguous()
            if ps
            else torch.zeros((2, 0), dtype=torch.long)
        )
        data[et].edge_index = ei
    return GraphData(
        data=data,
        node_ids=dict(node_ids),
        texts=dict(texts),
        node_index=node_index,
        target_articles=target_articles,
    )


# Graph used for message passing: keep only some target edges, then add reverse edges.
def message_graph(g: GraphData, keep_target: Tensor) -> HeteroData:
    """Copy of the graph keeping only the target edges where `keep_target` is True, made
    undirected (reverse edge types added), for message passing."""
    data = g.data.clone()
    data[TARGET].edge_index = g.target_edge_index[:, keep_target]
    return T.ToUndirected()(data)
