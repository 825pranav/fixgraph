"""Per-question metrics (spec §11.3). Pure functions; aggregation + CIs live in bench.stats.

Definitions (also in DECISIONS.md):
- recall@k: share of gold support chunks in the top-k retrieved chunks.
- support-set completeness: 1 if *every* gold support chunk (all hops) is in the top-k.
- citation precision: share of cited chunk ids that are gold support chunks.
- citation recall: share of gold support chunks that the answer cites.
- unsupported-claim rate: share of answer sentences the verifier labels unsupported.
- abstention: precision/recall of "abstained" against "question is unanswerable".
"""

from collections.abc import Sequence

import numpy as np


def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    if not gold:
        return float("nan")
    top = set(retrieved[:k])
    return sum(g in top for g in set(gold)) / len(set(gold))


def support_complete(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    if not gold:
        return float("nan")
    return float(set(gold) <= set(retrieved[:k]))


def citation_precision(cited: Sequence[str], gold: Sequence[str]) -> float:
    if not cited:
        return float("nan")
    g = set(gold)
    return sum(c in g for c in set(cited)) / len(set(cited))


def citation_recall(cited: Sequence[str], gold: Sequence[str]) -> float:
    if not gold:
        return float("nan")
    c = set(cited)
    return sum(g in c for g in set(gold)) / len(set(gold))


def unsupported_rate(verdicts: Sequence[str]) -> float:
    if not verdicts:
        return float("nan")
    return sum(v == "unsupported" for v in verdicts) / len(verdicts)


def abstention_prf(abstained: Sequence[bool], unanswerable: Sequence[bool]) -> dict[str, float]:
    tp = sum(a and u for a, u in zip(abstained, unanswerable, strict=True))
    fp = sum(a and not u for a, u in zip(abstained, unanswerable, strict=True))
    fn = sum(u and not a for a, u in zip(abstained, unanswerable, strict=True))
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    return {"precision": p, "recall": r, "tp": tp, "fp": fp, "fn": fn}


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else float("nan")


def nan_drop(values: Sequence[float]) -> list[float]:
    return [v for v in values if not np.isnan(v)]
