import json
from pathlib import Path

import pytest

from fixgraph.core.models import Chunk
from fixgraph.core.ontology import Ontology, load_ontology
from fixgraph.kg.extraction import (
    ChunkExtraction,
    GraphEntity,
    GraphRelation,
    LLMExtraction,
    LLMFix,
    LLMProblem,
    LLMProduct,
    build_request,
    classify_requirement,
    to_graph,
)
from fixgraph.kg.run_extract import output_path, read_records, run_extraction
from fixgraph.kg.schema import relation_allowed
from fixgraph.kg.validation import partial_ratio, validate
from fixgraph.llm import FakeLLMClient

CHUNK_TEXT = (
    "If your Apple Watch won't pair with your iPhone, Bluetooth might be turned off. "
    "Turn on Bluetooth, then update your iPhone to iOS 26.1."
)


@pytest.fixture(scope="module")
def onto() -> Ontology:
    return load_ontology()


def _llm_output() -> LLMExtraction:
    return LLMExtraction(
        products=[LLMProduct(name="Apple Watch", depends_on=["iPhone"])],
        problems=[
            LLMProblem(
                symptom="won't pair",
                products=["Apple Watch"],
                components=["Bluetooth"],
                causes=["Bluetooth might be turned off"],
                fixes=[
                    LLMFix(
                        action="Turn on Bluetooth",
                        addresses_cause="Bluetooth might be turned off",
                    ),
                    LLMFix(action="update your iPhone", requires=["iOS 26.1"]),
                ],
            )
        ],
    )


def _triples(g: ChunkExtraction) -> set[tuple[str, str, str]]:
    return {(g.entities[r.head].text, r.rel, g.entities[r.tail].text) for r in g.relations}


def test_to_graph_types_every_relation_correctly(onto: Ontology) -> None:
    g = to_graph(_llm_output(), onto)
    assert _triples(g) == {
        ("Apple Watch", "DEPENDS_ON", "iPhone"),
        ("Apple Watch", "EXHIBITS", "won't pair"),
        ("won't pair", "INVOLVES", "Bluetooth"),
        ("won't pair", "CAUSED_BY", "Bluetooth might be turned off"),
        ("won't pair", "RESOLVED_BY", "Turn on Bluetooth"),
        ("won't pair", "RESOLVED_BY", "update your iPhone"),
        ("Turn on Bluetooth", "ADDRESSES", "Bluetooth might be turned off"),
        ("update your iPhone", "REQUIRES", "iOS 26.1"),
    }
    for r in g.relations:
        assert relation_allowed(r.rel, g.entities[r.head].type, g.entities[r.tail].type)
    # "Apple Watch" appears twice in the output but is one entity.
    assert [e.text for e in g.entities].count("Apple Watch") == 1


def test_classify_requirement(onto: Ontology) -> None:
    assert classify_requirement("iOS 26.1", onto) == "OSVersion"
    assert classify_requirement("macOS Tahoe", onto) == "OSVersion"
    assert classify_requirement("iPhone", onto) == "Product"
    assert classify_requirement("iCloud Backup", onto) == "Feature"


def test_validation_keeps_grounded_relations(onto: Ontology) -> None:
    v = validate(to_graph(_llm_output(), onto), CHUNK_TEXT)
    assert len(v.relations) == 8
    assert v.rejects == {}
    assert all(0.9 <= r.conf <= 1.0 for r in v.relations)
    watch = next(e for e in v.entities if e.text == "Apple Watch")
    assert watch.span is not None
    assert CHUNK_TEXT[watch.span.start : watch.span.end] == "Apple Watch"


def test_validation_rejections() -> None:
    raw = ChunkExtraction(
        entities=[
            GraphEntity(type="Symptom", text="won't pair"),
            GraphEntity(type="Fix", text="Turn on Bluetooth"),
            GraphEntity(type="Fix", text="factory reset the Watch"),  # not in text
            GraphEntity(type="Symptom", text="won't pair"),  # duplicate
            GraphEntity(type="Cause", text="   "),  # empty
        ],
        relations=[
            GraphRelation(head=0, rel="RESOLVED_BY", tail=1, evidence="Turn on Bluetooth"),
            GraphRelation(head=1, rel="RESOLVED_BY", tail=0, evidence="Turn on Bluetooth"),
            GraphRelation(head=0, rel="RESOLVED_BY", tail=2, evidence="factory reset the Watch"),
            GraphRelation(head=3, rel="RESOLVED_BY", tail=1, evidence="Turn on Bluetooth"),
            GraphRelation(head=0, rel="RESOLVED_BY", tail=9, evidence="x"),
            GraphRelation(head=0, rel="RESOLVED_BY", tail=3, evidence="won't pair"),
        ],
    )
    v = validate(raw, CHUNK_TEXT)
    assert [(v.entities[r.head].text, r.rel, v.entities[r.tail].text) for r in v.relations] == [
        ("won't pair", "RESOLVED_BY", "Turn on Bluetooth")
    ]
    assert v.rejects == {
        "entity_duplicate": 1,
        "entity_empty": 1,
        "relation_type_violation": 1,
        "relation_evidence_not_found": 1,
        "relation_duplicate": 1,  # head 3 is remapped to the first "won't pair"
        "relation_bad_index": 1,
        "relation_self_loop": 1,
    }


def test_partial_ratio() -> None:
    assert partial_ratio("turn on bluetooth", CHUNK_TEXT) == 1.0
    assert partial_ratio("Turn on the Bluetooth", CHUNK_TEXT) > 0.85
    assert partial_ratio("factory reset", CHUNK_TEXT) < 0.7
    assert partial_ratio("won’t pair", CHUNK_TEXT) == 1.0  # curly apostrophe
    assert partial_ratio("", CHUNK_TEXT) == 0.0


def test_request_carries_schema_and_prompt() -> None:
    chunk = Chunk(
        chunk_id="1:0:0",
        article_id="1",
        section_idx=0,
        chunk_idx=0,
        heading="H",
        text=CHUNK_TEXT,
        char_start=0,
        char_end=len(CHUNK_TEXT),
        n_tokens=30,
    )
    req = build_request(chunk, "Title", "qwen3:4b")
    assert req.json_schema == LLMExtraction.model_json_schema()
    assert req.temperature == 0.0 and req.think is False
    assert req.messages[-1].content.endswith(CHUNK_TEXT)
    # Few-shot assistant turns must themselves be valid outputs.
    for m in req.messages:
        if m.role == "assistant":
            LLMExtraction.model_validate_json(m.content)


def _chunk(i: int) -> Chunk:
    return Chunk(
        chunk_id=f"1:{i}:0",
        article_id="1",
        section_idx=i,
        chunk_idx=0,
        heading="H",
        text=CHUNK_TEXT,
        char_start=0,
        char_end=len(CHUNK_TEXT),
        n_tokens=30,
    )


def test_runner_is_resumable_and_retries_failures(tmp_path: Path, onto: Ontology) -> None:
    good = _llm_output().model_dump_json()
    out = output_path(tmp_path, "m")
    chunks = [_chunk(i) for i in range(3)]
    # Chunk 2 fails twice (invalid JSON + invalid retry); others succeed.
    fake = FakeLLMClient(scripted=[good, good, "nope", "nope"])
    stats = run_extraction(fake, chunks, {"1": "T"}, "m", out, onto, concurrency=1)
    assert stats["processed"] == 3 and stats["ok"] == 2

    fake2 = FakeLLMClient(scripted=[good])
    stats2 = run_extraction(fake2, chunks, {"1": "T"}, "m", out, onto, concurrency=1)
    assert stats2["processed"] == 1 and stats2["already_done"] == 2
    recs = read_records(out)
    assert len(recs) == 3 and all(r.ok for r in recs)
    assert out.name == "m_v2.jsonl"
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 4  # 3 first-run records + 1 retry
    assert all(x["validated"]["relations"] for x in lines if x["ok"])
