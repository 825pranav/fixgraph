"""Tests bench/screen.py: the deterministic leak check, verbatim-quote verification, the
single-article test for multi-article questions, and review ordering. Uses the fake LLM."""

import json

from fixgraph.bench.schema import Question, ScreenResult
from fixgraph.bench.screen import leak_overlap, review_order, screen_question
from fixgraph.llm import FakeLLMClient

CHUNKS = {
    "1:0:0": "If your AirPods won't connect, put both AirPods in the charging case for 30 seconds.",
    "2:0:0": "If the status light doesn't flash white, put both AirPods in the charging case.",
}
NODES = {"fix": "Put both AirPods in your charging case", "s1": "AirPods won't connect"}


def _q(question: str, chunks: list[str]) -> Question:
    return Question(
        qid="t0",
        question=question,
        qtype="multi_constraint",
        gold_answer="Put both AirPods in the charging case.",
        gold_chunk_ids=chunks,
        gold_seed_nodes=["s1"],
        gold_answer_nodes=["fix"],
    )


def _evidence_reply(quote: str, supported: bool = True) -> str:
    return json.dumps(
        {"answerable": True, "answer_supported": supported, "support_quote": quote, "reason": "ok"}
    )


def test_leak_overlap_ignores_topic_words_and_stems() -> None:
    assert leak_overlap("Could updating macOS fix both?", ["Update macOS"]) == 1.0
    assert (
        leak_overlap("My Apple Watch won't pair. What can I do?", ["Unpair your Apple Watch"]) == 0
    )
    # words from the question's own seed nodes are topic, not answer
    assert (
        leak_overlap("iMessage won't turn on", ["Turn on iMessage"], ["iMessage won't turn on"])
        == 0
    )


def test_screen_passes_with_real_quote_and_needs_both_articles() -> None:
    quote = "put both AirPods in the charging case for 30 seconds"
    fake = FakeLLMClient(
        scripted=[
            _evidence_reply(quote),
            '{"fully_answerable": false}',
            '{"fully_answerable": false}',
        ]
    )
    q = _q("My AirPods won't connect and the light doesn't flash. One thing to try?", list(CHUNKS))
    r = screen_question(fake, q, CHUNKS, "m", NODES)
    assert r.passed and r.quote_found and r.needs_multiple_articles is True


def test_screen_fails_on_invented_quote_and_single_source() -> None:
    fake = FakeLLMClient(
        scripted=[
            _evidence_reply("Reset the network settings on your iPhone"),
            '{"fully_answerable": true}',
        ]
    )
    q = _q("My AirPods won't connect and the light doesn't flash. One thing to try?", list(CHUNKS))
    r = screen_question(fake, q, CHUNKS, "m", NODES)
    assert not r.passed and not r.quote_found and r.needs_multiple_articles is False
    assert "no verbatim supporting quote" in r.reason


def test_screen_flags_leak_even_if_model_approves() -> None:
    fake = FakeLLMClient(scripted=[_evidence_reply("put both AirPods in the charging case")])
    q = _q("Should I put both AirPods in the charging case?", ["1:0:0"])
    r = screen_question(fake, q, CHUNKS, "m", NODES)
    assert r.leaks_answer and not r.passed and r.needs_multiple_articles is None


def test_review_order_puts_passed_multi_article_first() -> None:
    def q(qid: str, passed: bool | None, multi: bool | None = None) -> Question:
        screen = (
            None
            if passed is None
            else ScreenResult(
                passed=passed,
                reason="",
                answerable=True,
                answer_supported=True,
                leaks_answer=False,
                needs_multiple_articles=multi,
            )
        )
        return Question(qid=qid, question=qid, qtype="single_hop", screen=screen)

    ordered = review_order([q("a", None), q("b", False), q("c", True), q("d", True, True)])
    assert [x.qid for x in ordered] == ["d", "c", "a", "b"]
