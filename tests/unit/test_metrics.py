import math

from fixgraph.bench.metrics import (
    abstention_prf,
    citation_precision,
    citation_recall,
    nan_drop,
    percentile,
    recall_at_k,
    support_complete,
    unsupported_rate,
)


def test_retrieval_metrics() -> None:
    retrieved = ["a", "b", "c", "d"]
    assert recall_at_k(retrieved, ["a", "d"], 2) == 0.5
    assert recall_at_k(retrieved, ["a", "d"], 4) == 1.0
    assert support_complete(retrieved, ["a", "d"], 2) == 0.0
    assert support_complete(retrieved, ["a", "d"], 4) == 1.0
    assert math.isnan(recall_at_k(retrieved, [], 2))


def test_citation_metrics() -> None:
    assert citation_precision(["a", "x"], ["a", "b"]) == 0.5
    assert citation_recall(["a", "x"], ["a", "b"]) == 0.5
    assert math.isnan(citation_precision([], ["a"]))
    assert unsupported_rate(["supported", "unsupported", "partial", "unsupported"]) == 0.5


def test_abstention() -> None:
    r = abstention_prf([True, True, False, False], [True, False, True, False])
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 1)
    assert r["precision"] == 0.5 and r["recall"] == 0.5


def test_helpers() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert nan_drop([1.0, float("nan"), 2.0]) == [1.0, 2.0]
