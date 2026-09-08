import json
from collections import Counter

import numpy as np
import polars as pl
import pytest

from fixgraph.core.models import Article
from fixgraph.core.ontology import Ontology, load_ontology
from fixgraph.embeddings import FakeEmbedder
from fixgraph.kg.build import build_kg
from fixgraph.kg.extraction import LLMExtraction, LLMFix, LLMProblem, LLMProduct, to_graph
from fixgraph.kg.quality import (
    GoldChunk,
    GoldEntity,
    GoldRelation,
    evaluate_extractions,
    graph_stats,
)
from fixgraph.kg.resolve import canonical_rule, cluster_texts, node_id, resolve_clustered
from fixgraph.kg.run_extract import ExtractionRecord
from fixgraph.kg.validation import validate


@pytest.fixture(scope="module")
def onto() -> Ontology:
    return load_ontology()


def test_canonical_rules(onto: Ontology) -> None:
    c = canonical_rule("Product", "your iPhone 16 Pro", onto)
    assert c is not None and (c.text, c.props["family"]) == ("iPhone 16 Pro", "iPhone")
    c = canonical_rule("Product", "your Apple Watch", onto)
    assert c is not None and c.text == "Apple Watch"
    assert canonical_rule("OSVersion", "macOS", onto) is None
    c = canonical_rule("OSVersion", "macOS Tahoe or later", onto)
    assert c is not None and c.text == "macOS 26" and c.props["major"] == 26
    c = canonical_rule("ErrorCode", "Error code 4013", onto)
    assert c is not None and c.text == "error 4013"
    c = canonical_rule("Component", "the Bluetooth radio", onto)
    assert c is not None and c.text == "Bluetooth"
    c = canonical_rule("Feature", "your iCloud Photos library", onto)
    assert c is not None and c.text == "iCloud Photos"
    c = canonical_rule("Component", "The side button", onto)
    assert c is not None and c.text == "side button"


def test_node_ids_stable_and_distinct() -> None:
    assert node_id("Symptom", "won't pair") == node_id("Symptom", "won't pair")
    assert node_id("Symptom", "won't pair") != node_id("Symptom", "wont pair")
    assert node_id("Symptom", "x").startswith("symptom:x:")


def test_cluster_texts_groups_near_duplicates() -> None:
    texts = ["restart your iphone", "restart the iphone", "update ios"]
    emb = FakeEmbedder().encode(texts)
    groups = cluster_texts(texts, [1, 1, 1], emb, threshold=0.5)
    assert sorted(map(sorted, groups)) == [[0, 1], [2]]


def test_cluster_texts_greedy_path_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    import fixgraph.kg.resolve as r

    monkeypatch.setattr(r, "MAX_AGGLOMERATIVE", 1)
    texts = ["restart your iphone", "restart the iphone", "update ios"]
    emb = FakeEmbedder().encode(texts)
    groups = r.cluster_texts(texts, [3, 1, 1], emb, threshold=0.5)
    assert sorted(map(sorted, groups)) == [[0, 1], [2]]


def test_resolve_clustered_medoid_and_adjudication() -> None:
    counts = Counter({"restart your iphone": 5, "restart the iphone": 1, "reboot iphone": 1})
    res = resolve_clustered("Fix", counts, FakeEmbedder(), threshold=0.5)
    assert res.canonical_of["restart the iphone"] == res.canonical_of["restart your iphone"]
    assert res.canonical_of["reboot iphone"] == "reboot iphone"

    calls: list[tuple[str, str]] = []

    def yes(type_: str, a: str, b: str) -> bool:
        calls.append((a, b))
        return True

    res2 = resolve_clustered("Fix", counts, FakeEmbedder(), threshold=0.5, adjudicate=yes, band=0.4)
    assert len(set(res2.canonical_of.values())) == 1
    assert calls and any(m.method == "llm" for m in res2.merges)


CHUNK = (
    "If your Apple Watch won't pair with your iPhone, Bluetooth might be turned off. "
    "Turn on Bluetooth, then update your iPhone to iOS 26.1."
)


def _record(chunk_id: str, fix_text: str, onto: Ontology) -> ExtractionRecord:
    x = LLMExtraction(
        products=[LLMProduct(name="Apple Watch", depends_on=["iPhone"])],
        problems=[
            LLMProblem(
                symptom="won't pair",
                products=["your Apple Watch"],
                causes=["Bluetooth might be turned off"],
                fixes=[
                    LLMFix(action=fix_text),
                    LLMFix(action="update your iPhone", requires=["iOS 26.1"]),
                ],
            )
        ],
    )
    g = to_graph(x, onto)
    return ExtractionRecord(
        chunk_id=chunk_id,
        model="qwen3:4b",
        prompt_version="v2",
        extracted_at=f"2026-09-23T00:00:0{chunk_id[-1]}",
        ok=True,
        raw=x,
        graph=g,
        validated=validate(g, CHUNK),
    )


def _articles() -> list[Article]:
    return [Article(article_id="1", url="u", title="If your Apple Watch won't pair")]


def test_build_kg_provenance_and_merging(onto: Ontology) -> None:
    recs = [
        _record("1:0:0", "Turn on Bluetooth", onto),
        _record("1:1:0", "turn on Bluetooth", onto),
    ]
    kg, report = build_kg(recs, _articles(), onto, FakeEmbedder())
    nodes = {r["node_id"]: r for r in kg.nodes.iter_rows(named=True)}
    labels = Counter(r["label"] for r in nodes.values())
    assert labels["Fix"] == 2  # "Turn on Bluetooth" merged across chunks + "update your iPhone"
    assert labels["ProductFamily"] == 2 and labels["Article"] == 1
    assert labels["OSVersion"] == 1

    resolved = kg.edges.filter(pl.col("rel") == "RESOLVED_BY")
    assert len(resolved) == 2
    for row in resolved.iter_rows(named=True):
        assert row["source_chunk_ids"] == ["1:0:0", "1:1:0"]
        assert row["origin"] == "extracted" and row["extraction_model"] == "qwen3:4b"
        assert 0.0 < row["extraction_confidence"] <= 1.0
        assert row["extracted_at"] == "2026-09-23T00:00:00"
    assert set(kg.edges["rel"].to_list()) >= {"IN_FAMILY", "DEPENDS_ON", "EXHIBITS", "REQUIRES"}
    # Every extracted edge has provenance.
    assert graph_stats(kg)["edges_with_provenance_pct"] == 100.0
    assert any(m.method == "rule" and m.canonical == "Apple Watch" for m in report.merges)
    props = json.loads(next(r for r in nodes.values() if r["label"] == "OSVersion")["props_json"])
    assert props == {"platform": "iOS", "major": 26, "minor": 1, "patch": 0}


def test_build_without_canonicalization_keeps_surface_forms(onto: Ontology) -> None:
    recs = [
        _record("1:0:0", "Turn on Bluetooth", onto),
        _record("1:1:0", "Turn Bluetooth on", onto),
    ]
    kg, _ = build_kg(recs, _articles(), onto, FakeEmbedder(), canonicalize=False)
    fixes = kg.nodes.filter(pl.col("label") == "Fix")["canonical_text"].to_list()
    assert len(fixes) == 3


def test_graph_stats(onto: Ontology) -> None:
    kg, _ = build_kg(
        [_record("1:0:0", "Turn on Bluetooth", onto)], _articles(), onto, FakeEmbedder()
    )
    s = graph_stats(kg)
    assert s["symptoms_with_fix_pct"] == 100.0
    assert s["nodes_by_label"]["Symptom"] == 1
    assert s["largest_component_nodes"] >= 5


def test_evaluate_extractions(onto: Ontology) -> None:
    pred = {"c": _record("c:0:0", "Turn on Bluetooth", onto).validated}
    assert pred["c"] is not None
    gold = [
        GoldChunk(
            chunk_id="c",
            entities=[
                GoldEntity(type="Symptom", text="won't pair"),
                GoldEntity(type="Fix", text="Turn on Bluetooth"),
                GoldEntity(type="Fix", text="Restart both devices"),
            ],
            relations=[
                GoldRelation(
                    head_type="Symptom",
                    head="won't pair",
                    rel="RESOLVED_BY",
                    tail_type="Fix",
                    tail="Turn on Bluetooth",
                ),
                GoldRelation(
                    head_type="Symptom",
                    head="won't pair",
                    rel="RESOLVED_BY",
                    tail_type="Fix",
                    tail="Restart both devices",
                ),
            ],
        )
    ]
    ent, rel = evaluate_extractions({"c": pred["c"]}, gold)
    assert ent["Fix"].tp == 1 and ent["Fix"].fn == 1 and ent["Fix"].fp == 1  # "update your iPhone"
    assert ent["Symptom"].tp == 1
    assert rel["RESOLVED_BY"].tp == 1 and rel["RESOLVED_BY"].fn == 1
    assert 0 < rel["ALL"].precision < 1 and np.isclose(rel["RESOLVED_BY"].recall, 0.5)
