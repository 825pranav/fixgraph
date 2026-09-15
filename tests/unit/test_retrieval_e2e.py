"""Retrieval + answering + report on a tiny synthetic corpus: real Qdrant local mode,
fake embedder/reranker/LLM (no GPU, no network).

Covers retrieval (index, hybrid, graph, graphrag, linking), answer.grounded, answer.verifier
and bench.run's report.
"""

import json
from pathlib import Path

import pytest

from fixgraph.answer.grounded import AnswerOutput, AnswerSentence, answer_question
from fixgraph.answer.verifier import VerifierOutput, verify_answer
from fixgraph.bench.judge import JudgeOutput
from fixgraph.bench.run import AnswerRow, JudgeRow, RetrievalRow, build_report, render_markdown
from fixgraph.bench.schema import Question
from fixgraph.core.models import Article, Chunk
from fixgraph.core.ontology import load_ontology
from fixgraph.embeddings import FakeEmbedder
from fixgraph.kg.build import build_kg
from fixgraph.kg.extraction import LLMExtraction, LLMFix, LLMProblem, LLMProduct, to_graph
from fixgraph.kg.run_extract import ExtractionRecord
from fixgraph.kg.validation import validate
from fixgraph.llm import FakeLLMClient
from fixgraph.llm.base import LLMRequest
from fixgraph.retrieval.graph import build_graph_index
from fixgraph.retrieval.graphrag import PathRetriever, PPRRetriever
from fixgraph.retrieval.hybrid import HybridRetriever
from fixgraph.retrieval.index import build_chunk_index, build_node_index, load_bm25, open_client
from fixgraph.retrieval.linking import EntityLinker, Mentions
from fixgraph.retrieval.rerank import LexicalReranker

TEXTS = {
    "1:0:0": "If your Apple Watch won't pair with your iPhone, turn on Bluetooth on your iPhone.",
    "2:0:0": "If your AirPods won't charge, clean the charging case and connect it to power.",
    "3:0:0": "Update your iPhone to iOS 26 before you pair a new Apple Watch.",
}


def _chunk(cid: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        article_id=cid.split(":")[0],
        section_idx=0,
        chunk_idx=0,
        heading="h",
        text=TEXTS[cid],
        char_start=0,
        char_end=len(TEXTS[cid]),
        n_tokens=20,
    )


def _record(cid: str, x: LLMExtraction) -> ExtractionRecord:
    onto = load_ontology()
    g = to_graph(x, onto)
    return ExtractionRecord(
        chunk_id=cid,
        model="m",
        prompt_version="v2",
        extracted_at="t",
        ok=True,
        raw=x,
        graph=g,
        validated=validate(g, TEXTS[cid]),
    )


@pytest.fixture(scope="module")
def setup(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    tmp = tmp_path_factory.mktemp("idx")
    onto = load_ontology()
    records = [
        _record(
            "1:0:0",
            LLMExtraction(
                products=[LLMProduct(name="Apple Watch", depends_on=["iPhone"])],
                problems=[
                    LLMProblem(
                        symptom="Apple Watch won't pair",
                        products=["Apple Watch"],
                        fixes=[LLMFix(action="turn on Bluetooth")],
                    )
                ],
            ),
        ),
        _record(
            "2:0:0",
            LLMExtraction(
                problems=[
                    LLMProblem(
                        symptom="AirPods won't charge",
                        products=["AirPods"],
                        fixes=[LLMFix(action="clean the charging case")],
                    )
                ]
            ),
        ),
    ]
    articles = [Article(article_id=a, url="u", title=f"T{a}") for a in ("1", "2", "3")]
    emb = FakeEmbedder()
    kg, _ = build_kg(records, articles, onto, emb)
    client = open_client(tmp)
    chunks = [_chunk(c) for c in TEXTS]
    build_chunk_index(client, chunks, {"1": "", "2": "", "3": ""}, emb, tmp)
    names = [
        (r["node_id"], r["label"], r["canonical_text"])
        for r in kg.nodes.iter_rows(named=True)
        if r["label"] != "Article"
    ]
    build_node_index(client, names, emb)
    docs = {c: TEXTS[c] for c in TEXTS}
    hybrid = HybridRetriever(client, emb, load_bm25(tmp), LexicalReranker(), docs)
    graph = build_graph_index(kg)
    counts = {graph.node_ids[i]: len(c) for i, c in graph.node_chunks.items()}
    linker = EntityLinker(client, emb, counts, threshold=0.3)
    q = "My Apple Watch won't pair"
    mentions = {q: Mentions(products=["Apple Watch"], symptoms=["won't pair"])}
    yield hybrid, graph, linker, mentions, q
    client.close()


def test_hybrid_finds_relevant_chunk(setup) -> None:  # type: ignore[no-untyped-def]
    hybrid, *_ = setup
    res = hybrid.retrieve("AirPods won't charge", 2)
    assert res.chunk_ids[0] == "2:0:0"
    assert res.timings["total_s"] >= 0


def test_ppr_and_paths_route_through_graph(setup) -> None:  # type: ignore[no-untyped-def]
    hybrid, graph, linker, mentions, q = setup
    ppr = PPRRetriever(graph, linker, hybrid, mentions)
    res = ppr.retrieve(q, 2)
    assert res.chunk_ids[0] == "1:0:0"
    assert res.timings["fallback"] == 0.0 and res.subgraph and res.subgraph.seeds
    paths = PathRetriever(graph, linker, hybrid, mentions)
    res3 = paths.retrieve(q, 2)
    assert "1:0:0" in res3.chunk_ids
    assert res3.subgraph and any("RESOLVED_BY" in p[0] for p in res3.subgraph.paths)


def test_graph_retriever_falls_back_without_seeds(setup) -> None:  # type: ignore[no-untyped-def]
    hybrid, graph, linker, _m, _q = setup
    linker_strict = EntityLinker(
        linker.client, linker.embedder, linker.node_chunk_counts, threshold=1.01
    )
    res = PPRRetriever(graph, linker_strict, hybrid, {}).retrieve("AirPods won't charge", 2)
    assert res.timings["fallback"] == 1.0 and res.chunk_ids


def test_answer_strips_foreign_citations_and_verifier_drops_unsupported() -> None:
    def respond(req: LLMRequest) -> str:
        if "Claims:" in req.messages[-1].content:
            return VerifierOutput.model_validate(
                {
                    "verdicts": [
                        {"claim": 0, "verdict": "supported"},
                        {"claim": 1, "verdict": "unsupported"},
                    ]
                }
            ).model_dump_json()
        return AnswerOutput(
            abstain=False,
            sentences=[
                AnswerSentence(text="Turn on Bluetooth.", citations=["1:0:0", "9:9:9"]),
                AnswerSentence(text="Also buy a new watch.", citations=["1:0:0"]),
            ],
        ).model_dump_json()

    llm = FakeLLMClient(responder=respond)
    ans = answer_question(llm, "q", ["1:0:0"], TEXTS, "m")
    assert ans.sentences[0].citations == ["1:0:0"]  # 9:9:9 was never in context
    ver = verify_answer(llm, ans, TEXTS, "judge")
    assert ver.verdicts == ["supported", "unsupported"]
    assert [s.text for s in ver.kept] == ["Turn on Bluetooth."]


def test_report_end_to_end(tmp_path: Path) -> None:
    from fixgraph.answer.grounded import GroundedAnswer
    from fixgraph.answer.verifier import VerifiedAnswer
    from fixgraph.retrieval.base import RetrievalResult

    qs = [
        Question(
            qid=f"q{i}",
            question=f"q{i}",
            qtype="single_hop",
            gold_answer="a",
            key_facts=["a"],
            # q0-q2 cite one article, q3-q5 two (chunk ids start with the article id)
            gold_chunk_ids=["a:0:0"] if i < 3 else ["a:0:0", "b:0:0"],
        )
        for i in range(6)
    ] + [Question(qid="u", question="u", qtype="unanswerable", answerable=False)]
    rets, ans, jud = [], [], []
    for s, good in (("S1", False), ("S2", True)):
        for q in qs:
            rets.append(
                RetrievalRow(
                    qid=q.qid,
                    system=s,
                    result=RetrievalResult(
                        chunk_ids=["a:0:0", "b:0:0"] if good else ["z:0:0"],
                        timings={"total_s": 0.1},
                    ),
                )
            )
            ga = GroundedAnswer(
                question=q.question,
                abstained=not q.answerable,
                sentences=[]
                if not q.answerable
                else [AnswerSentence(text="a", citations=["a:0:0"])],
                context_chunk_ids=["a:0:0"],
            )
            ans.append(AnswerRow(qid=q.qid, system=s, answer=ga))
            jud.append(
                JudgeRow(
                    qid=q.qid,
                    system=s,
                    verified=VerifiedAnswer(verdicts=["supported"], kept=[], abstained=False),
                    judge=JudgeOutput(
                        score=1.0 if (good or not q.answerable) else 0.0, key_facts_covered=[good]
                    ),
                )
            )
    rep = build_report(qs, rets, ans, jud)
    assert rep["table"]["S2"]["correctness"]["mean"] == 1.0
    assert (
        rep["table"]["S2"]["recall@8"]["mean"] == 1.0
        and rep["table"]["S1"]["recall@8"]["mean"] == 0.0
    )
    corr = next(t for t in rep["tests"] if t["metric"] == "correctness")
    assert corr["diff"] > 0 and corr["p"] < 0.05
    assert rep["table"]["S1"]["abstention_recall"]["mean"] == 1.0
    spans = rep["by_evidence_span"]
    assert spans["multi-article"]["S2"]["n"] == 3 and spans["single-article"]["S2"]["n"] == 4
    assert spans["multi-article"]["S2"]["recall@8"] == 1.0
    multi = next(t for t in rep["multi_article_tests"] if t["metric"] == "correctness")
    assert multi["n"] == 3 and multi["diff"] == 1.0
    assert rep["provenance"] == {"human_verified": 0, "auto_screen_passed": 0, "total": 7}
    md = render_markdown(rep)
    assert "| correctness |" in md and "| multi-article |" in md
    # Rows for questions outside the set (e.g. rejected in review) are ignored, not a KeyError.
    subset = build_report(qs[:2], rets, ans, jud)
    assert subset["table"]["S2"]["correctness"]["n"] == 2
    json.dumps(rep)  # serializable
