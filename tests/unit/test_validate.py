from pathlib import Path

from fixgraph.bench.schema import Question
from fixgraph.bench.validate import (
    agreement,
    label_loop,
    read_labels,
    sample_for_labeling,
    write_labels,
)


def _qs() -> dict[str, Question]:
    qs = [
        Question(
            qid=f"q{i}", question=f"q{i}", qtype="single_hop", gold_answer="a", key_facts=["a"]
        )
        for i in range(4)
    ]
    qs.append(Question(qid="u", question="u", qtype="unanswerable", answerable=False))
    return {q.qid: q for q in qs}


def test_sample_excludes_unanswerable_and_is_deterministic() -> None:
    keys = [(q, s) for q in _qs() for s in ("S1", "S2")]
    a = sample_for_labeling(keys, _qs(), n=5)
    assert a == sample_for_labeling(keys, _qs(), n=5) and len(a) == 5
    assert all(k[0] != "u" for k in a)


def test_label_loop_resumes_and_agreement(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    todo = [("q0", "S1"), ("q1", "S1"), ("q2", "S2")]
    keys = iter(["1", "x", "5", "q"])  # invalid key re-prompts; q stops before q2
    labels = label_loop(
        todo,
        _qs(),
        {},
        [],
        lambda x: write_labels(x, path),
        "dev",
        ask=lambda _: next(keys),
        show=lambda _: None,
    )
    assert [(x.qid, x.score) for x in read_labels(path)] == [("q0", 1.0), ("q1", 0.5)]
    keys2 = iter(["0"])
    labels = label_loop(
        todo,
        _qs(),
        {},
        labels,
        lambda x: write_labels(x, path),
        "dev",
        ask=lambda _: next(keys2),
        show=lambda _: None,
    )
    assert len(read_labels(path)) == 3
    ag = agreement(labels, {("q0", "S1"): 1.0, ("q1", "S1"): 0.5, ("q2", "S2"): 1.0})
    assert ag.n == 3 and ag.exact_agreement == round(2 / 3, 3)
    assert -1.0 <= ag.kappa <= 1.0
