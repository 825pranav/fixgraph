"""FastAPI endpoints in fake mode: health, retrieve, answer citations, graph and suggestions.

Covers api/app.py (with kg.store for the graph endpoints).
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fixgraph.api.app import ServiceState, create_app, fake_state
from fixgraph.kg.store import EDGE_SCHEMA, KG, MENTION_SCHEMA, NODE_SCHEMA


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app(fake_state()))


def test_health_fake_mode(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["mode"] == "fake"
    assert body["llm_reachable"] is True and body["kg_loaded"] is False
    assert "S1" in body["systems"]


def test_answer_citations_are_retrieved_chunks(client: TestClient) -> None:
    r = client.post("/answer", json={"question": "My Apple Watch won't pair", "system": "S1"})
    assert r.status_code == 200
    body = r.json()
    assert not body["abstained"]
    assert body["citations"] and set(body["citations"]) <= set(body["chunk_ids"])
    assert body["chunk_ids"][0] == "demo:0:0"


def test_closed_book_abstains_in_fake_mode(client: TestClient) -> None:
    r = client.post("/answer", json={"question": "Why won't my Watch pair?", "system": "S0"})
    assert r.status_code == 200 and r.json()["abstained"] is True


def test_retrieve_and_unknown_system(client: TestClient) -> None:
    r = client.post("/retrieve", json={"question": "AirPods won't charge", "k": 1})
    assert r.status_code == 200 and r.json()["chunk_ids"] == ["demo:1:0"]
    assert client.post("/retrieve", json={"question": "x", "system": "S9"}).status_code == 400


def _kg() -> KG:
    import polars as pl

    nodes = pl.DataFrame(
        [
            {
                "node_id": "symptom:a",
                "label": "Symptom",
                "canonical_text": "won't pair",
                "props_json": "{}",
            },
            {"node_id": "fix:b", "label": "Fix", "canonical_text": "unpair", "props_json": "{}"},
        ],
        schema=NODE_SCHEMA,
    )
    edges = pl.DataFrame(
        [
            {
                "edge_id": "e1",
                "src": "symptom:a",
                "rel": "RESOLVED_BY",
                "dst": "fix:b",
                "source_chunk_ids": ["demo:0:0"],
                "extraction_model": "m",
                "extraction_confidence": 1.0,
                "extracted_at": "",
                "origin": "extracted",
                "n_support": 1,
            }
        ],
        schema=EDGE_SCHEMA,
    )
    return KG(nodes, edges, pl.DataFrame([], schema=MENTION_SCHEMA))


def test_graph_endpoints(tmp_path: Path) -> None:
    gaps = tmp_path / "gaps.jsonl"
    gaps.write_text(
        json.dumps({"symptom_id": "symptom:a", "fix_id": "fix:c", "score": 0.9}) + "\n",
        encoding="utf-8",
    )
    st: ServiceState = fake_state()
    st.kg = _kg()
    st.gap_file = gaps
    c = TestClient(create_app(st))
    assert c.get("/graph/entity/nope").status_code == 404
    e = c.get("/graph/entity/symptom:a").json()
    assert e["node"]["text"] == "won't pair" and e["edges"][0]["rel"] == "RESOLVED_BY"
    sg = c.post("/graph/subgraph", json={"node_ids": ["symptom:a", "fix:b"]}).json()
    assert len(sg["nodes"]) == 2 and len(sg["edges"]) == 1
    sugg = c.get("/links/suggestions", params={"symptom_id": "symptom:a"}).json()
    assert sugg == [
        {"symptom_id": "symptom:a", "fix_id": "fix:c", "score": 0.9, "origin": "predicted"}
    ]


def test_suggestions_empty_without_file(client: TestClient) -> None:
    r = client.get("/links/suggestions", params={"symptom_id": "symptom:x"})
    assert r.status_code == 200 and r.json() == []
    assert client.get("/graph/entity/x").status_code == 404
