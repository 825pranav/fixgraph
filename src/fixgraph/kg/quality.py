"""Graph quality evaluation (spec §8.3): gold-set P/R/F1 per type and graph statistics.

Used by: `fixgraph kg stats | eval` (kg/cli.py); kg.annotate reuses the GoldChunk format.
Uses: kg.validation (ValidatedExtraction, normalize), kg.store.KG.
"""

# Imports: difflib for fuzzy text matching, scipy sparse graphs for connected-component stats.
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import polars as pl
from pydantic import BaseModel, Field
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from fixgraph.kg.store import KG
from fixgraph.kg.validation import ValidatedExtraction, normalize

# ---------------------------------------------------------------------------
# Gold annotations
# ---------------------------------------------------------------------------


# A hand-labelled entity in the gold set.
class GoldEntity(BaseModel):
    type: str
    text: str


# A hand-labelled relation, stored by text so it can be compared to model output by fuzzy match.
class GoldRelation(BaseModel):
    head_type: str
    head: str
    rel: str
    tail_type: str
    tail: str


# One gold-set chunk with its labelled entities and relations, plus who labelled it and status.
class GoldChunk(BaseModel):
    chunk_id: str
    entities: list[GoldEntity] = Field(default_factory=lambda: list[GoldEntity]())
    relations: list[GoldRelation] = Field(default_factory=lambda: list[GoldRelation]())
    annotator: str = ""
    status: str = "draft"  # draft | reviewed


# Loose text equality for scoring: normalized equal, contained with similar length, or 80% similar.
def text_match(a: str, b: str, threshold: float = 0.8) -> bool:
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return False
    if na == nb or na in nb or nb in na:
        # containment counts only when lengths are comparable ("pair" must not match a sentence)
        return min(len(na), len(nb)) / max(len(na), len(nb)) >= 0.5 or na == nb
    return SequenceMatcher(None, na, nb).ratio() >= threshold


# True/false positive and false negative counts, with precision, recall and F1 derived from them.
class PRF(BaseModel):
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    # One report row with the counts and rounded scores.
    def row(self, name: str) -> dict[str, Any]:
        return {
            "name": name,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
            "f1": round(self.f1, 3),
        }


# Count true positives by pairing each prediction with at most one unused gold item.
def _greedy_match(pred: list[Any], gold: list[Any], same: Any) -> int:
    used: set[int] = set()
    tp = 0
    for p in pred:
        for j, g in enumerate(gold):
            if j not in used and same(p, g):
                used.add(j)
                tp += 1
                break
    return tp


# Score model extractions against the gold set; used by `kg eval`.
def evaluate_extractions(
    predictions: dict[str, ValidatedExtraction], gold: list[GoldChunk]
) -> tuple[dict[str, PRF], dict[str, PRF]]:
    """Per-type entity PRF and per-relation PRF (plus 'ALL'), matched greedily one-to-one."""
    ent: dict[str, PRF] = defaultdict(PRF)
    rel: dict[str, PRF] = defaultdict(PRF)
    # Turn each chunk's validated output into the same text form as the gold labels.
    for g in gold:
        v = predictions.get(g.chunk_id) or ValidatedExtraction()
        pred_ents = [GoldEntity(type=e.type, text=e.text) for e in v.entities]
        pred_rels = [
            GoldRelation(
                head_type=v.entities[r.head].type,
                head=v.entities[r.head].text,
                rel=r.rel,
                tail_type=v.entities[r.tail].type,
                tail=v.entities[r.tail].text,
            )
            for r in v.relations
        ]
        # Entity scores per type, also added into an overall "ALL" row.
        for t in {e.type for e in pred_ents} | {e.type for e in g.entities}:
            p = [e for e in pred_ents if e.type == t]
            gg = [e for e in g.entities if e.type == t]
            tp = _greedy_match(p, gg, lambda a, b: text_match(a.text, b.text))
            for key in (t, "ALL"):
                ent[key].tp += tp
                ent[key].fp += len(p) - tp
                ent[key].fn += len(gg) - tp
        # Relation scores per relation type; both the head and the tail text must match.
        for rt in {r.rel for r in pred_rels} | {r.rel for r in g.relations}:
            p = [r for r in pred_rels if r.rel == rt]
            gg = [r for r in g.relations if r.rel == rt]
            tp = _greedy_match(
                p,
                gg,
                lambda a, b: text_match(a.head, b.head) and text_match(a.tail, b.tail),
            )
            for key in (rt, "ALL"):
                rel[key].tp += tp
                rel[key].fp += len(p) - tp
                rel[key].fn += len(gg) - tp
    return dict(ent), dict(rel)


# ---------------------------------------------------------------------------
# Graph statistics
# ---------------------------------------------------------------------------


# Summary numbers for a graph: counts by label/relation, degrees, fix coverage, connectivity.
def graph_stats(kg: KG) -> dict[str, Any]:
    nodes, edges = kg.nodes, kg.extracted_edges()
    label_counts = dict(Counter(nodes["label"].to_list()).most_common())
    rel_counts = dict(Counter(edges["rel"].to_list()).most_common())

    # Map node ids to positions and compute each node's degree from the extracted edges.
    ids = nodes["node_id"].to_list()
    index = {n: i for i, n in enumerate(ids)}
    src = np.array([index[s] for s in edges["src"].to_list()], dtype=np.int64)
    dst = np.array([index[d] for d in edges["dst"].to_list()], dtype=np.int64)
    degree = (
        np.bincount(np.concatenate([src, dst]), minlength=len(ids)) if len(ids) else np.zeros(0)
    )

    # Degree summary per node label, including the share of isolated nodes.
    labels = nodes["label"].to_list()
    degree_by_label: dict[str, dict[str, float]] = {}
    for lab in label_counts:
        d = degree[[i for i, x in enumerate(labels) if x == lab]]
        degree_by_label[lab] = {
            "mean": round(float(d.mean()), 2),
            "median": float(np.median(d)),
            "max": int(d.max()),
            "isolated_pct": round(100 * float((d == 0).mean()), 1),
        }

    # Share of symptoms that have at least one fix edge.
    symptoms = set(nodes.filter(pl.col("label") == "Symptom")["node_id"].to_list())
    with_fix = set(edges.filter(pl.col("rel") == "RESOLVED_BY")["src"].to_list()) & symptoms
    # Count connected components and the largest one, ignoring Article nodes.
    graph_nodes = [i for i, x in enumerate(labels) if x != "Article"]
    n_comp, comp_labels, largest = 0, np.zeros(0), 0
    if len(ids):
        adj = coo_matrix((np.ones(len(src)), (src, dst)), shape=(len(ids), len(ids)))
        n_comp, comp_labels = connected_components(adj, directed=False)
        sizes = np.bincount(comp_labels[graph_nodes]) if graph_nodes else np.zeros(1)
        largest = int(sizes.max())
        n_comp = int((sizes > 0).sum())
    return {
        "nodes": len(nodes),
        "edges_extracted": len(edges),
        "nodes_by_label": label_counts,
        "edges_by_rel": rel_counts,
        "degree_by_label": degree_by_label,
        "symptoms_with_fix_pct": round(100 * len(with_fix) / len(symptoms), 1) if symptoms else 0.0,
        "components_excluding_articles": n_comp,
        "largest_component_nodes": largest,
        "edges_with_provenance_pct": _provenance_pct(edges),
    }


# Percent of edges that point back to source chunks; the provenance check for the graph.
def _provenance_pct(edges: pl.DataFrame) -> float:
    """Share of edges with source chunks (ontology-derived edges count as sourced)."""
    if not len(edges):
        return 100.0
    ok = sum(
        1
        for chunks, model in zip(
            edges["source_chunk_ids"].to_list(), edges["extraction_model"].to_list(), strict=True
        )
        if chunks or model == "ontology"
    )
    return round(100 * ok / len(edges), 1)
