"""Parquet KG -> PyG HeteroData (spec §9.1).

Node types are KG labels (Article excluded); node features are text embeddings of the canonical
text. Edge types are (src_label, rel.lower(), dst_label) from `origin == "extracted"` edges only.
The graph is kept directed here; `message_graph` filters target edges and applies ToUndirected,
so held-out edges can be removed from message passing in both directions.
"""

from collections import defaultdict
from dataclasses import dataclass

import polars as pl
import torch
import torch_geometric.transforms as T
from torch import Tensor
from torch_geometric.data import HeteroData

from fixgraph.embeddings import Embedder
from fixgraph.kg.store import KG

TARGET = ("Symptom", "resolved_by", "Fix")
REV_TARGET = ("Fix", "rev_resolved_by", "Symptom")


@dataclass
class GraphData:
    data: HeteroData  # directed; target edges in data[TARGET].edge_index
    node_ids: dict[str, list[str]]  # node type -> node ids in index order
    texts: dict[str, list[str]]
    node_index: dict[str, tuple[str, int]]  # node id -> (type, local index)
    target_articles: list[frozenset[str]]  # per target edge (column), its provenance articles

    @property
    def target_edge_index(self) -> Tensor:
        return self.data[TARGET].edge_index

    def num_nodes(self, node_type: str) -> int:
        return len(self.node_ids.get(node_type, []))


def build_graph_data(kg: KG, embedder: Embedder) -> GraphData:
    nodes = kg.nodes.filter(pl.col("label") != "Article")
    node_ids: dict[str, list[str]] = defaultdict(list)
    texts: dict[str, list[str]] = defaultdict(list)
    node_index: dict[str, tuple[str, int]] = {}
    for nid, label, text in nodes.select("node_id", "label", "canonical_text").iter_rows():
        node_index[nid] = (label, len(node_ids[label]))
        node_ids[label].append(nid)
        texts[label].append(text)

    data = HeteroData()
    for label, ts in texts.items():
        data[label].x = torch.from_numpy(embedder.encode(ts)).float()
    for label in ("Symptom", "Fix"):  # the target types always exist, even if empty
        if label not in texts:
            data[label].x = torch.zeros((0, embedder.dim))

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


def message_graph(g: GraphData, keep_target: Tensor) -> HeteroData:
    """Copy of the graph keeping only the target edges where `keep_target` is True, made
    undirected (reverse edge types added), for message passing."""
    data = g.data.clone()
    data[TARGET].edge_index = g.target_edge_index[:, keep_target]
    return T.ToUndirected()(data)
