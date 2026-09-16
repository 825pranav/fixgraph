import json

import polars as pl

from fixgraph.kg.store import EDGE_SCHEMA, KG, MENTION_SCHEMA, NODE_SCHEMA
from fixgraph.kg.verify import (
    EdgeLabel,
    EdgeVerdict,
    allocate,
    edge_claims,
    edge_sample,
    evaluate_edges,
    verified_kg,
    verify_chunk,
)
from fixgraph.llm.fake import FakeLLMClient

CHUNK = "If your AirPods won't connect, make sure Bluetooth is on. Restart your iPhone."


def _edge(eid: str, src: str, rel: str, dst: str, chunks: list[str], model: str = "m") -> dict:
    return {
        "edge_id": eid,
        "src": src,
        "rel": rel,
        "dst": dst,
        "source_chunk_ids": chunks,
        "extraction_model": model,
        "extraction_confidence": 1.0,
        "extracted_at": "",
        "origin": "extracted",
        "n_support": len(chunks),
    }


def _kg() -> KG:
    nodes = [
        ("s", "Symptom", "AirPods won't connect"),
        ("f1", "Fix", "Make sure Bluetooth is on"),
        ("f2", "Fix", "Restart your iPhone"),
        ("a", "Product", "AirPods"),
        ("i", "Product", "iPhone"),
        ("fam", "ProductFamily", "AirPods"),
    ]
    edges = [
        _edge("e1", "s", "RESOLVED_BY", "f1", ["c1", "c2"]),
        _edge("e2", "s", "RESOLVED_BY", "f2", ["c1"]),
        _edge("e3", "a", "DEPENDS_ON", "i", ["c1"]),
        _edge("o1", "a", "IN_FAMILY", "fam", ["c1"], model="ontology"),
        _edge("o2", "a", "DEPENDS_ON", "i", [], model="ontology"),
    ]
    mentions = [
        ("s", "c1", "AirPods won't connect"),
        ("f1", "c1", "make sure Bluetooth is on"),
        ("f1", "c2", "turn on Bluetooth"),
    ]
    return KG(
        pl.DataFrame(
            [
                {"node_id": n, "label": lab, "canonical_text": t, "props_json": "{}"}
                for n, lab, t in nodes
            ],
            schema=NODE_SCHEMA,
        ),
        pl.DataFrame(edges, schema=EDGE_SCHEMA),
        pl.DataFrame(
            [{"node_id": n, "chunk_id": c, "surface": s} for n, c, s in mentions],
            schema=MENTION_SCHEMA,
        ),
    )


def test_claims_use_chunk_surface_forms_and_skip_ontology_edges() -> None:
    claims = edge_claims(_kg())
    assert set(claims) == {"c1", "c2"}
    assert {c.edge_id for c in claims["c1"]} == {"e1", "e2", "e3"}  # no o1 / o2
    (c2,) = claims["c2"]
    assert '"turn on Bluetooth"' in c2.statement  # c2's own wording, not the canonical text


def test_verdict_needs_llm_yes_and_a_quote_found_in_the_chunk() -> None:
    claims = edge_claims(_kg())["c1"]
    reply = {
        "verdicts": [
            {"id": 0, "supported": True, "quote": "make sure Bluetooth is on"},
            {"id": 1, "supported": True, "quote": "Unplug the charger and wait."},  # invented
            {"id": 2, "supported": False, "quote": ""},
        ]
    }
    client = FakeLLMClient(scripted=[json.dumps(reply)])
    got = {v.edge_id: v for v in verify_chunk(client, CHUNK, claims, "judge")}
    ids = [c.edge_id for c in claims]
    assert got[ids[0]].supported
    assert got[ids[1]].llm_supported and not got[ids[1]].supported
    assert not got[ids[2]].supported


def test_failed_call_fails_closed() -> None:
    claims = edge_claims(_kg())["c1"]
    client = FakeLLMClient(scripted=["not json", "still not json"])
    got = verify_chunk(client, CHUNK, claims, "judge")
    assert not any(v.supported for v in got) and all(v.error for v in got)


def test_verified_kg_keeps_supported_edges_with_narrowed_provenance() -> None:
    verdicts = [
        EdgeVerdict(edge_id="e1", chunk_id="c1", llm_supported=False, supported=False),
        EdgeVerdict(edge_id="e1", chunk_id="c2", llm_supported=True, supported=True),
        EdgeVerdict(edge_id="e2", chunk_id="c1", llm_supported=True, supported=False),
        EdgeVerdict(edge_id="e3", chunk_id="c1", llm_supported=False, supported=False),
    ]
    edges = verified_kg(_kg(), verdicts).edges
    assert sorted(edges["edge_id"]) == ["e1", "o1"]  # seed DEPENDS_ON (o2) dropped too
    e1 = edges.filter(pl.col("edge_id") == "e1").row(0, named=True)
    assert e1["source_chunk_ids"] == ["c2"] and e1["n_support"] == 1


def test_allocation_floor_and_sample_is_seeded() -> None:
    assert allocate({"A": 900, "B": 100, "C": 3}, n=100, min_per=5) == {"A": 90, "B": 10, "C": 3}
    assert edge_sample(_kg(), n=2, min_per=1, seed=1) == edge_sample(_kg(), n=2, min_per=1, seed=1)


def test_rates_are_weighted_by_stratum_size() -> None:
    from fixgraph.kg.verify import SampledEdge

    sample = [
        SampledEdge(edge_id=e, rel=r, chunk_ids=["c"], statements=["x"])
        for e, r in [("a1", "A"), ("a2", "A"), ("b1", "B"), ("b2", "B")]
    ]
    labels = [
        EdgeLabel(edge_id="a1", supported=True, labeler="t"),
        EdgeLabel(edge_id="a2", supported=True, labeler="t"),
        EdgeLabel(edge_id="b1", supported=False, labeler="t"),
        EdgeLabel(edge_id="b2", supported=True, labeler="t"),
    ]
    kept = [
        EdgeVerdict(edge_id=e, chunk_id="c", llm_supported=True, supported=True)
        for e in ("a1", "a2", "b2")
    ]
    rep = evaluate_edges(sample, labels, kept, {"A": 90, "B": 10}, {"A": 90, "B": 5}, n_boot=50)
    assert rep["hallucinated_rate_before"]["mean"] == 0.05  # 10 * 1/2 of B out of 100
    assert rep["hallucinated_rate_after"]["mean"] == 0.0
    assert rep["verifier_vs_labels"]["precision_keep"] == 1.0
    assert rep["verifier_vs_labels"]["tn"] == 1
