import polars as pl
import pytest
import torch

from fixgraph.embeddings import FakeEmbedder
from fixgraph.gnn.data import REV_TARGET, TARGET, build_graph_data
from fixgraph.gnn.evaluate import aggregate, auroc, evaluate, filtered_ranks, known_fixes
from fixgraph.gnn.models import adamic_adar_scores, cosine_scores, train_distmult, train_sage
from fixgraph.gnn.splits import article_held_out_split, transductive_split
from fixgraph.kg.store import EDGE_SCHEMA, KG, MENTION_SCHEMA, NODE_SCHEMA


def _kg() -> KG:
    nodes = [("product:iphone", "Product", "iPhone"), ("article:1", "Article", "A")]
    nodes += [(f"symptom:{i}", "Symptom", f"problem number {i} won't work") for i in range(6)]
    nodes += [(f"fix:{i}", "Fix", f"restart step {i}") for i in range(6)]
    edges = []
    for i in range(6):
        # symptom i is resolved by fix i and fix (i+1)%6; provenance from article i
        for f in (i, (i + 1) % 6):
            edges.append((f"symptom:{i}", "RESOLVED_BY", f"fix:{f}", [f"{i}:0:0"]))
        edges.append(("product:iphone", "EXHIBITS", f"symptom:{i}", [f"{i}:0:0"]))
    edges.append(("symptom:0", "RESOLVED_BY", "fix:3", ["0:1:0", "5:0:0"]))  # two articles
    node_df = pl.DataFrame(
        [
            {"node_id": n, "label": lab, "canonical_text": t, "props_json": "{}"}
            for n, lab, t in nodes
        ],
        schema=NODE_SCHEMA,
    )
    edge_rows = [
        {
            "edge_id": str(k),
            "src": s,
            "rel": r,
            "dst": d,
            "source_chunk_ids": c,
            "extraction_model": "m",
            "extraction_confidence": 1.0,
            "extracted_at": "",
            "origin": "extracted",
            "n_support": len(c),
        }
        for k, (s, r, d, c) in enumerate(edges)
    ]
    edge_rows.append({**edge_rows[0], "edge_id": "p", "dst": "fix:5", "origin": "predicted"})
    return KG(
        pl.DataFrame(node_df),
        pl.DataFrame(edge_rows, schema=EDGE_SCHEMA),
        pl.DataFrame([], schema=MENTION_SCHEMA),
    )


def test_heterodata_shapes_and_provenance() -> None:
    g = build_graph_data(_kg(), FakeEmbedder(dim=32))
    d = g.data
    assert set(d.node_types) == {"Product", "Symptom", "Fix"}  # Article excluded
    assert d["Symptom"].x.shape == (6, 32) and d["Fix"].x.shape == (6, 32)
    assert d[TARGET].edge_index.shape == (2, 13)  # predicted edge excluded
    assert d[("Product", "exhibits", "Symptom")].edge_index.shape == (2, 6)
    assert g.target_articles[-1] == frozenset({"0", "5"})
    assert g.node_index["fix:3"] == ("Fix", 3)


def test_article_held_out_edges_absent_from_message_passing() -> None:
    g = build_graph_data(_kg(), FakeEmbedder(dim=32))
    split = article_held_out_split(g, holdout_frac=0.34, seed=1)
    assert split.test_pos.size(1) > 0
    test = {tuple(p) for p in split.test_pos.t().tolist()}
    for data in (split.message, split.eval_message):
        fwd = {tuple(p) for p in data[TARGET].edge_index.t().tolist()}
        rev = {(s, f) for f, s in data[REV_TARGET].edge_index.t().tolist()}
        assert not (test & fwd) and not (test & rev)
    sup = {tuple(p) for p in split.train_pos.t().tolist()}
    assert not (sup & {tuple(p) for p in split.message[TARGET].edge_index.t().tolist()})
    # Test edges come only from held-out articles.
    assert all(
        a <= split.held_out_articles
        for a, is_t in zip(
            g.target_articles,
            [tuple(p) in test for p in g.target_edge_index.t().tolist()],
            strict=True,
        )
        if is_t
    )


def test_filtered_mrr_hand_example() -> None:
    table = torch.tensor([[0.9, 0.5, 0.1], [0.8, 0.6, 0.1]])
    test_pos = torch.tensor([[0, 1], [0, 1]])  # (s0, f0) rank 1, (s1, f1) rank 2
    known = known_fixes(test_pos)
    m = evaluate(lambda s: table[s], test_pos, known)
    assert m["mrr"] == pytest.approx(0.75)
    assert m["hits@1"] == 0.5 and m["hits@3"] == 1.0
    # Filtering: if f0 is also a known fix of s1, it no longer outranks f1.
    known[1].add(0)
    ranks, _, _ = filtered_ranks(lambda s: table[s], test_pos, known)
    assert ranks == [1.0, 1.0]


def test_ties_and_auroc() -> None:
    table = torch.zeros((1, 5))
    ranks, _, _ = filtered_ranks(lambda s: table[s], torch.tensor([[0], [2]]), {0: {2}})
    assert ranks == [3.0]  # 4 ties -> expected rank 1 + 4/2
    assert auroc([1.0, 0.8], [0.1, 0.8]) == pytest.approx(0.875)
    agg = aggregate([{"mrr": 0.5}, {"mrr": 0.7}])
    assert agg["mrr"]["mean"] == pytest.approx(0.6) and agg["mrr"]["std"] == pytest.approx(0.1)


def test_baselines_and_sage_train_on_cpu() -> None:
    g = build_graph_data(_kg(), FakeEmbedder(dim=32))
    split = article_held_out_split(g, holdout_frac=0.34, seed=0)
    syms = torch.tensor([0, 1])
    assert cosine_scores(split.eval_message, syms).shape == (2, 6)
    aa = adamic_adar_scores(split.eval_message, syms)
    assert aa.shape == (2, 6) and float(aa.min()) >= 0.0
    dm = train_distmult(split.eval_message, epochs=5, dim=8)
    assert dm.scores(split.eval_message, syms).shape == (2, 6)
    sage = train_sage(split.message, split.train_pos, epochs=5)
    s = sage.scores(split.eval_message, syms)
    assert s.shape == (2, 6) and torch.isfinite(s).all()
    m = evaluate(
        lambda x: sage.scores(split.eval_message, x),
        split.test_pos,
        known_fixes(g.target_edge_index),
    )
    assert 0.0 < m["mrr"] <= 1.0


def test_transductive_split_runs() -> None:
    g = build_graph_data(_kg(), FakeEmbedder(dim=32))
    train, _val, test = transductive_split(g, seed=0)
    assert TARGET in train.edge_types and REV_TARGET in train.edge_types
    assert test[TARGET].edge_label_index.size(1) > 0
