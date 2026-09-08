"""Filtered ranking metrics (spec §9.4): each held-out (symptom, fix) is ranked against ALL
Fix nodes after removing the symptom's other known true fixes. Ties count half (expected rank
under random tie-breaking). AUROC (secondary) compares each positive with one sampled negative.
"""

from collections import defaultdict
from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor

ScoreFn = Callable[[Tensor], Tensor]  # symptom indices -> [len, num_fix] scores


def known_fixes(edge_index: Tensor) -> dict[int, set[int]]:
    known: dict[int, set[int]] = defaultdict(set)
    for s, f in edge_index.t().tolist():
        known[s].add(f)
    return dict(known)


def filtered_ranks(
    score_fn: ScoreFn, test_pos: Tensor, known: dict[int, set[int]]
) -> tuple[list[float], list[float], list[float]]:
    """Returns (ranks, positive scores, sampled-negative scores)."""
    if test_pos.size(1) == 0:
        return [], [], []
    symptoms = torch.unique(test_pos[0])
    scores = score_fn(symptoms).float()
    row_of = {int(s): i for i, s in enumerate(symptoms.tolist())}
    gen = np.random.default_rng(0)
    ranks: list[float] = []
    pos_scores: list[float] = []
    neg_scores: list[float] = []
    for s, f in test_pos.t().tolist():
        row = scores[row_of[s]].clone()
        others = [x for x in known.get(s, set()) if x != f]
        target = float(row[f])
        if others:
            row[others] = float("-inf")
        greater = int((row > target).sum())
        ties = int((row == target).sum()) - 1
        ranks.append(1.0 + greater + ties / 2.0)
        pos_scores.append(target)
        candidates = [x for x in range(row.numel()) if x not in known.get(s, set())]
        if candidates:
            neg_scores.append(float(scores[row_of[s], int(gen.choice(candidates))]))
    return ranks, pos_scores, neg_scores


def auroc(pos: list[float], neg: list[float]) -> float:
    """Mann-Whitney AUROC; 0.5 per tie."""
    if not pos or not neg:
        return float("nan")
    p = np.asarray(pos)[:, None]
    n = np.asarray(neg)[None, :]
    return float(((p > n).sum() + 0.5 * (p == n).sum()) / (p.size * n.size))


def ranking_metrics(ranks: list[float]) -> dict[str, float]:
    if not ranks:
        return {
            "mrr": float("nan"),
            "hits@1": float("nan"),
            "hits@3": float("nan"),
            "hits@10": float("nan"),
            "n": 0,
        }
    r = np.asarray(ranks)
    return {
        "mrr": float((1.0 / r).mean()),
        "hits@1": float((r <= 1).mean()),
        "hits@3": float((r <= 3).mean()),
        "hits@10": float((r <= 10).mean()),
        "n": int(r.size),
    }


def evaluate(score_fn: ScoreFn, test_pos: Tensor, known: dict[int, set[int]]) -> dict[str, float]:
    ranks, pos, neg = filtered_ranks(score_fn, test_pos, known)
    return {**ranking_metrics(ranks), "auroc": auroc(pos, neg)}


def aggregate(runs: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """mean and std (ddof=0) across seeds for every metric."""
    out: dict[str, dict[str, float]] = {}
    for key in runs[0] if runs else []:
        vals = np.asarray([r[key] for r in runs], dtype=np.float64)
        out[key] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
    return out
