from pathlib import Path

import pytest

from fixgraph.bench.generate import PathSample, generate_question, sample_paths, verify_loop
from fixgraph.bench.schema import Question, read_questions, write_questions
from fixgraph.core.models import Article
from fixgraph.core.ontology import load_ontology
from fixgraph.embeddings import FakeEmbedder
from fixgraph.kg.build import build_kg
from fixgraph.kg.extraction import LLMExtraction, LLMFix, LLMProblem, LLMProduct, to_graph
from fixgraph.kg.run_extract import ExtractionRecord
from fixgraph.kg.validation import validate
from fixgraph.llm import FakeLLMClient

TEXT = (
    "If your Apple Watch won't pair with your iPhone, error 4013 may appear. Turn on "
    "Bluetooth, then update your iPhone to iOS 26.1."
)


@pytest.fixture(scope="module")
def kg():  # type: ignore[no-untyped-def]
    onto = load_ontology()
    x = LLMExtraction(
        products=[LLMProduct(name="Apple Watch", depends_on=["iPhone"])],
        problems=[
            LLMProblem(
                symptom="won't pair",
                products=["Apple Watch"],
                error_codes=["error 4013"],
                fixes=[
                    LLMFix(action="Turn on Bluetooth"),
                    LLMFix(action="update your iPhone", requires=["iOS 26.1"]),
                ],
            )
        ],
    )
    g = to_graph(x, onto)
    rec = ExtractionRecord(
        chunk_id="1:0:0",
        model="m",
        prompt_version="v2",
        extracted_at="t",
        ok=True,
        raw=x,
        graph=g,
        validated=validate(g, TEXT),
    )
    built, _ = build_kg([rec], [Article(article_id="1", url="u", title="t")], onto, FakeEmbedder())
    return built


def test_sample_paths_covers_types(kg) -> None:  # type: ignore[no-untyped-def]
    samples = sample_paths(kg, per_type=2)
    types = {s.qtype for s in samples}
    assert {"single_hop", "cross_device", "version_conditional", "error_code"} <= types
    assert all(s.chunk_ids == ["1:0:0"] for s in samples)


def test_generate_question_and_verify(tmp_path: Path) -> None:
    fake = FakeLLMClient(
        scripted=[
            '{"question": "Why won\'t my watch pair?", "answer": "Turn on Bluetooth.", '
            '"key_facts": ["turn on Bluetooth"]}'
        ]
    )
    s = PathSample(
        qtype="single_hop",
        facts=["a RESOLVED_BY b"],
        answer_nodes=["b"],
        seed_nodes=["a"],
        chunk_ids=["1:0:0"],
    )
    q = generate_question(fake, s, {"1:0:0": TEXT}, "m", "g0")
    assert q is not None and q.gold_chunk_ids == ["1:0:0"] and not q.verified
    qs = [q, Question(qid="g1", question="bad", qtype="single_hop")]
    path = tmp_path / "q.jsonl"
    keys = iter(["y", "n"])
    n = verify_loop(
        qs,
        {"1:0:0": TEXT},
        lambda x: write_questions(x, path),
        "dev",
        ask=lambda _: next(keys),
        show=lambda _: None,
    )
    saved = read_questions(path)
    assert n == 1 and [x.qid for x in saved] == ["g0"] and saved[0].verified_by == "dev"
