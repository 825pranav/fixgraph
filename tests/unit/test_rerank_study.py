from fixgraph.bench.rerank_study import (
    StudyConfig,
    StudyRow,
    evaluate,
    grid,
    rank,
    score_slices,
    seconds,
    select,
)
from fixgraph.bench.schema import QType, Question


def _row(qid: str = "q") -> StudyRow:
    fused = [f"c{i}" for i in range(60)]
    # "a" ranks c0..c59 in order; "b" prefers c40 above everything
    a = {c: 1.0 - i * 0.01 for i, c in enumerate(fused)}
    b = {c: 0.0 for c in fused} | {"c40": 5.0}
    return StudyRow(
        qid=qid,
        fused=fused,
        rrf=[1.0 / (60 + i) for i in range(60)],
        search_s=0.1,
        scores={"a": a, "b": b},
        score_s={"a": {"30": 0.3, "50": 0.2, "100": 0.1}, "b": {"30": 1.0, "50": 1.0, "100": 1.0}},
    )


def test_rank_respects_pool_and_blend() -> None:
    row = _row()
    assert rank(row, StudyConfig(scorer="a", pool=30)) == [f"c{i}" for i in range(8)]
    assert "c40" not in rank(row, StudyConfig(scorer="b", pool=30))
    assert rank(row, StudyConfig(scorer="b", pool=50))[0] == "c40"
    blended = rank(row, StudyConfig(scorer="a", pool=50, second="b", w2=0.5))
    assert blended[0] == "c40" and len(blended) == 8


def test_seconds_prices_pool_slices_and_both_scorers() -> None:
    row = _row()
    assert abs(seconds(row, StudyConfig(scorer="a", pool=30)) - 0.4) < 1e-9
    assert abs(seconds(row, StudyConfig(scorer="a", pool=50)) - 0.6) < 1e-9
    both = seconds(row, StudyConfig(scorer="a", pool=30, second="b", w2=0.5))
    assert abs(both - 1.4) < 1e-9


def test_score_slices_scores_every_candidate_once() -> None:
    calls: list[int] = []

    def score(q: str, docs: list[str]) -> list[float]:
        calls.append(len(docs))
        return [float(len(d)) for d in docs]

    fused = [f"c{i}" for i in range(40)]
    out, secs = score_slices("q", fused, {c: c for c in fused}, score)
    assert calls == [30, 10] and len(out) == 40 and set(secs) == {"30", "50", "100"}


def _q(qid: str, qtype: QType, gold: list[str]) -> Question:
    return Question(qid=qid, question="q", qtype=qtype, gold_chunk_ids=gold)


def test_evaluate_and_select_prefer_strict_gains() -> None:
    qs = [_q("m1", "single_hop", ["c40"]), _q("m2", "single_hop", ["c1"])]
    rows = {"m1": _row("m1"), "m2": _row("m2")}
    base = StudyConfig(scorer="a", pool=30)
    cfgs = [base, StudyConfig(scorer="b", pool=50)]
    log = evaluate(qs, rows, cfgs, base)
    assert log[0]["all"]["recall@8"] == 0.5 and log[1]["all"]["recall@8"] == 1.0
    assert log[1]["recovered"] == 1 and log[1]["broken"] == 0
    assert select(log, base.name)["config"] == "b@50"
    assert select(log[:1], base.name)["config"] == base.name
    assert len(grid(["bge", "qwen"])) == 2 * 3 + 2 * 6 + 4


class _Fixed:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def score(self, query: str, docs: list[str]) -> list[float]:
        return [self.scores[d] for d in docs]

    def release(self) -> None:
        pass


def test_hybrid_rerank_blend_matches_study_ranking() -> None:
    from typing import Any, cast

    from fixgraph.retrieval.hybrid import HybridRetriever

    docs = {f"c{i}": f"c{i}" for i in range(10)}
    first = _Fixed({f"c{i}": float(10 - i) for i in range(10)})
    second = _Fixed({f"c{i}": 1.0 if i == 9 else 0.0 for i in range(10)})
    cands = [(f"c{i}", 1.0 / (60 + i)) for i in range(10)]
    none = cast(Any, None)
    plain = HybridRetriever(none, none, none, first, docs, rerank_top=10)
    assert [c for c, _ in plain.rerank("q", cands, 3)] == ["c0", "c1", "c2"]
    blend = HybridRetriever(none, none, none, first, docs, rerank_top=10, second=second, w2=0.9)
    assert blend.rerank("q", cands, 3)[0][0] == "c9"
    row = StudyRow(qid="q", fused=[c for c, _ in cands], rrf=[s for _, s in cands], search_s=0.0,
                   scores={"a": first.scores, "b": second.scores}, score_s={})  # fmt: skip
    cfg = StudyConfig(scorer="a", pool=10, second="b", w2=0.9, alpha=0.1)
    live = HybridRetriever(none, none, none, first, docs, rerank_top=10, second=second, w2=0.9,
                           alpha=0.1)  # fmt: skip
    assert [c for c, _ in live.rerank("q", cands, 8)] == rank(row, cfg)


def _entry(name: str, rec: float, lat: float) -> dict[str, object]:
    return {"config": name, "all": {"recall@8": rec}, "latency_s": {"mean": lat}}


def test_select_finalists_adds_a_latency_neutral_pick() -> None:
    from typing import Any, cast

    from fixgraph.bench.rerank_study import select_finalists

    log = cast(Any, [_entry("S1", 0.90, 1.0), _entry("slow", 0.95, 3.0),
                     _entry("cheap", 0.93, 1.05)])  # fmt: skip
    assert [f["config"] for f in select_finalists(log, "S1")] == ["slow", "cheap"]
    log = cast(Any, [_entry("S1", 0.90, 1.0), _entry("worse", 0.80, 1.0)])
    assert select_finalists(log, "S1") == []


def test_test_summary_reports_tests_and_latency() -> None:
    from fixgraph.bench.rerank_study import test_summary

    qs = [_q(f"m{i}", "single_hop", [f"g{i}"]) for i in range(4)]
    base = {q.qid: ["x"] for q in qs}
    new = {q.qid: [q.gold_chunk_ids[0]] for q in qs}
    rep = test_summary(qs, {"S1": base, "new": new}, {"S1": [1.0] * 4, "new": [2.0] * 4}, "S1")
    assert rep["table"]["main"]["new"]["recall@8"]["mean"] == 1.0
    assert rep["tests_vs_baseline"][0]["diff"] == 1.0
    assert rep["latency_s"]["new"]["p50"] == 2.0
